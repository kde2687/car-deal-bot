import math
from datetime import datetime, timedelta

from flask import Flask, jsonify, render_template, request
from sqlalchemy import func

import config
from database import SessionLocal, Listing
from geo import haversine_km, city_to_coords, ORIGIN_LAT, ORIGIN_LON, ORIGIN_NAME, CITY_COORDS


def format_price(value) -> str:
    if value is None:
        return "N/A"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "N/A"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if value >= 1_000:
        return f"{value / 1_000:.0f}K"
    return f"{value:.0f}"


def time_ago(dt: datetime) -> str:
    if dt is None:
        return "never"
    now = datetime.utcnow()
    diff = now - dt
    total_seconds = int(diff.total_seconds())
    if total_seconds < 60:
        return f"{total_seconds} sec ago"
    minutes = total_seconds // 60
    if minutes < 60:
        return f"{minutes} min ago"
    hours = minutes // 60
    if hours < 24:
        return f"{hours} hours ago"
    days = hours // 24
    return f"{days} days ago"


def create_app() -> Flask:
    app = Flask(__name__, template_folder="templates")
    app.secret_key = config.FLASK_SECRET_KEY

    app.jinja_env.filters["format_price"] = format_price
    app.jinja_env.filters["time_ago"] = time_ago

    import os as _os
    _ADMIN_TOKEN = _os.getenv("ADMIN_TOKEN", "")

    def _check_admin():
        if not _ADMIN_TOKEN:
            return None  # not configured = allow (dev mode)
        token = request.args.get("token") or request.headers.get("X-Admin-Token", "")
        if token != _ADMIN_TOKEN:
            return jsonify({"error": "unauthorized"}), 401
        return None

    def _apply_filters(query, brand, model, city, source, origin_city, min_year, max_year, max_km, min_score, max_distance, new_today=False, since="", min_price_drops=None, sort=None):
        if brand:
            query = query.filter(Listing.brand.ilike(f"%{brand}%"))
        if model:
            query = query.filter(Listing.model.ilike(f"%{model}%"))
        if city:
            query = query.filter(Listing.seller_city.ilike(f"%{city}%"))
        if source:
            query = query.filter(Listing.source == source)
        if min_year:
            query = query.filter(Listing.year >= min_year)
        if max_year:
            query = query.filter(Listing.year <= max_year)
        if max_km:
            query = query.filter(Listing.km <= max_km)
        if min_score is not None:
            query = query.filter(Listing.score >= min_score)
        if new_today:
            today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
            query = query.filter(Listing.first_seen >= today_start)
        if since:
            now = datetime.utcnow()
            if since == "today":
                cutoff = now.replace(hour=0, minute=0, second=0, microsecond=0)
                query = query.filter(Listing.first_seen >= cutoff)
            elif since == "week":
                query = query.filter(Listing.first_seen >= now - timedelta(days=7))
            elif since == "month":
                query = query.filter(Listing.first_seen >= now - timedelta(days=30))
            elif since == "older":
                query = query.filter(Listing.first_seen < now - timedelta(days=30))
        if min_price_drops is not None:
            query = query.filter(Listing.price_changes_count >= min_price_drops)
        # Distance filter: SQL fast-path only when using the default origin (Darregueira)
        origin_lower = (origin_city or "").lower().strip()
        using_default_origin = not origin_lower or "darregueira" in origin_lower
        if max_distance is not None and using_default_origin:
            query = query.filter(Listing.distance_km <= max_distance)
        return query

    def _apply_distance_filter(listings, origin_city, max_distance):
        """Python-side distance filter for custom origin cities."""
        if max_distance is None:
            return listings
        origin_lower = (origin_city or "").lower().strip()
        if not origin_lower or "darregueira" in origin_lower:
            return listings   # already handled in SQL
        coords = city_to_coords(origin_lower)
        if not coords:
            return listings   # unknown city — don't filter
        olat, olon = coords
        result = []
        for lst in listings:
            if lst.seller_lat is not None and lst.seller_lon is not None:
                d = haversine_km(olat, olon, lst.seller_lat, lst.seller_lon)
                if d <= max_distance:
                    result.append(lst)
            # listings with no coords are excluded when a custom origin is active
        return result

    def _read_filters():
        return {
            "brand":        request.args.get("brand", "").strip(),
            "model":        request.args.get("model", "").strip(),
            "city":         request.args.get("city", "").strip(),
            "source":       request.args.get("source", "").strip(),
            "origin_city":  request.args.get("origin_city", "Darregueira").strip(),
            "min_year":     request.args.get("min_year", type=int),
            "max_year":     request.args.get("max_year", type=int),
            "max_km":       request.args.get("max_km", type=int),
            "min_score":    request.args.get("min_score", type=int),
            "max_distance": request.args.get("max_distance", type=int),
            "new_today":       bool(request.args.get("new_today")),
            "since":           request.args.get("since", "").strip(),
            "min_price_drops": request.args.get("min_price_drops", type=int),
            "sort":            request.args.get("sort", "score_desc"),
        }

    def _filters_for_template(f):
        return {k: (v if v is not None else "") for k, v in f.items()}

    @app.route("/")
    def index():
        session = SessionLocal()
        try:
            page = max(1, request.args.get("page", 1, type=int))
            f = _read_filters()

            query = session.query(Listing).filter(
                Listing.is_deal == True,
                Listing.status == "active",
                Listing.hidden != True,
                Listing.is_agency != True,
                Listing.discount_pct >= 0,
            )
            query = _apply_filters(query, **f)
            sort = f.get("sort", "score_desc")
            if sort == "discount_desc":
                query = query.order_by(Listing.discount_pct.desc().nullslast())
            elif sort == "price_asc":
                query = query.order_by(Listing.price_ars.asc().nullslast())
            elif sort == "price_desc":
                query = query.order_by(Listing.price_ars.desc())
            elif sort == "newest":
                query = query.order_by(Listing.first_seen.desc())
            else:
                query = query.order_by(Listing.score.desc())

            all_rows = query.all()
            all_rows = _apply_distance_filter(all_rows, f["origin_city"], f["max_distance"])
            total = len(all_rows)
            pages = max(1, math.ceil(total / 20))
            listings = all_rows[(page - 1) * 20: page * 20]

            last_updated = session.query(func.max(Listing.last_seen)).scalar()
            today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
            new_today = session.query(Listing).filter(
                Listing.is_deal == True, Listing.first_seen >= today_start).count()
            avg_score_row = session.query(func.avg(Listing.score)).filter(Listing.is_deal == True).scalar()
            avg_score = round(float(avg_score_row), 1) if avg_score_row else 0.0
            unique_brands = [r[0] for r in session.query(Listing.brand).filter(
                Listing.is_deal == True).distinct().order_by(Listing.brand).all() if r[0]]
            unique_cities = [r[0] for r in session.query(Listing.seller_city).filter(
                Listing.is_deal == True, Listing.seller_city != None, Listing.seller_city != ""
            ).distinct().order_by(Listing.seller_city).all() if r[0]]
            origin_coords = city_to_coords(f["origin_city"]) or (ORIGIN_LAT, ORIGIN_LON)

            return render_template("index.html",
                listings=listings, total=total, page=page, pages=pages,
                filters=_filters_for_template(f), last_updated=last_updated,
                new_today=new_today, avg_score=avg_score,
                sources=[], unique_brands=unique_brands, unique_cities=unique_cities,
                origin_name=f["origin_city"] or ORIGIN_NAME,
                origin_coords=origin_coords, all_origin_cities=sorted(CITY_COORDS.keys()),
                usd_rate=config.get_usd_mep_rate(),
                view="deals")
        finally:
            session.close()

    @app.route("/all")
    def all_listings():
        session = SessionLocal()
        try:
            page = max(1, request.args.get("page", 1, type=int))
            f = _read_filters()

            query = session.query(Listing).filter(
                Listing.status == "active",
                Listing.hidden != True,
                Listing.is_agency != True,
            )
            query = _apply_filters(query, **f)
            sort = f.get("sort", "score_desc")
            if sort == "discount_desc":
                query = query.order_by(Listing.discount_pct.desc().nullslast())
            elif sort == "price_asc":
                query = query.order_by(Listing.price_ars.asc().nullslast())
            elif sort == "price_desc":
                query = query.order_by(Listing.price_ars.desc())
            elif sort == "newest":
                query = query.order_by(Listing.first_seen.desc())
            else:
                query = query.order_by(Listing.score.desc())

            all_rows = query.all()
            all_rows = _apply_distance_filter(all_rows, f["origin_city"], f["max_distance"])
            total = len(all_rows)
            pages = max(1, math.ceil(total / 20))
            listings = all_rows[(page - 1) * 20: page * 20]

            last_updated = session.query(func.max(Listing.last_seen)).scalar()
            today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
            new_today = session.query(Listing).filter(
                Listing.status == "active", Listing.first_seen >= today_start).count()
            avg_score_row = session.query(func.avg(Listing.score)).filter(
                Listing.status == "active").scalar()
            avg_score = round(float(avg_score_row), 1) if avg_score_row else 0.0
            unique_brands = [r[0] for r in session.query(Listing.brand).filter(
                Listing.status == "active").distinct().order_by(Listing.brand).all() if r[0]]
            unique_cities = [r[0] for r in session.query(Listing.seller_city).filter(
                Listing.status == "active",
                Listing.seller_city != None, Listing.seller_city != ""
            ).distinct().order_by(Listing.seller_city).all() if r[0]]
            origin_coords = city_to_coords(f["origin_city"]) or (ORIGIN_LAT, ORIGIN_LON)

            return render_template("index.html",
                listings=listings, total=total, page=page, pages=pages,
                filters=_filters_for_template(f), last_updated=last_updated,
                new_today=new_today, avg_score=avg_score,
                sources=[], unique_brands=unique_brands, unique_cities=unique_cities,
                origin_name=f["origin_city"] or ORIGIN_NAME,
                origin_coords=origin_coords, all_origin_cities=sorted(CITY_COORDS.keys()),
                usd_rate=config.get_usd_mep_rate(),
                view="all")
        finally:
            session.close()

    @app.route("/hide/<path:listing_id>", methods=["POST"])
    def hide_listing(listing_id):
        session = SessionLocal()
        try:
            listing = session.query(Listing).filter_by(id=listing_id).first()
            if listing:
                listing.hidden = True
                listing.is_deal = False
                session.commit()
            return ("", 204)
        finally:
            session.close()

    @app.route("/mark_agency/<path:listing_id>", methods=["POST"])
    def mark_agency(listing_id):
        """Manually flag a listing as an agency — removes from deals permanently."""
        session = SessionLocal()
        try:
            listing = session.query(Listing).filter_by(id=listing_id).first()
            if listing:
                listing.is_agency = True
                listing.is_deal = False
                listing.deal_reason = "Marcado manualmente como agencia"
                session.commit()
            return ("", 204)
        finally:
            session.close()

    @app.route("/admin/ml_login")
    def ml_login():
        """Redirect to ML OAuth2 authorization page (authorization_code flow)."""
        err = _check_admin()
        if err: return err
        from ml_auth import get_authorization_url
        url = get_authorization_url()
        return (
            f'<h2>Autorizar CarDeal Bot en MercadoLibre</h2>'
            f'<p>Hacé clic para autorizar. Serás redirigido a ML y luego de vuelta aquí.</p>'
            f'<a href="{url}" style="padding:12px 24px;background:#3483FA;color:white;'
            f'text-decoration:none;border-radius:6px;font-size:16px;">Autorizar en ML</a>'
        )

    @app.route("/admin/ml_callback")
    def ml_callback():
        """Handle ML OAuth2 callback — exchange code for tokens."""
        err = _check_admin()
        if err: return err
        import asyncio, httpx, config as cfg
        code = request.args.get("code")
        if not code:
            return jsonify({"error": "no code in callback", "args": dict(request.args)}), 400

        result = {}
        async def _exchange():
            async with httpx.AsyncClient(follow_redirects=True) as client:
                resp = await client.post(
                    "https://api.mercadolibre.com/oauth/token",
                    data={
                        "grant_type": "authorization_code",
                        "client_id": cfg.ML_APP_ID,
                        "client_secret": cfg.ML_CLIENT_SECRET,
                        "code": code,
                        "redirect_uri": "https://cardeal.ar/admin/ml_callback",
                    },
                    headers={"Content-Type": "application/x-www-form-urlencoded"},
                    timeout=15.0,
                )
                result["status"] = resp.status_code
                data = resp.json()
                result["response"] = data
                if resp.status_code == 200 and "access_token" in data:
                    from ml_auth import _manager
                    _manager.set_tokens(
                        data["access_token"],
                        data.get("refresh_token", ""),
                        data.get("expires_in", 21600),
                    )
                    result["ok"] = True

        try:
            asyncio.run(_exchange())
        except RuntimeError:
            loop = asyncio.new_event_loop()
            loop.run_until_complete(_exchange())
            loop.close()

        if result.get("ok"):
            return (
                "<h2>✓ Autorización exitosa</h2>"
                "<p>El refresh_token fue guardado. ML search API funcionará en el próximo scan.</p>"
                "<a href='/'>Ir al dashboard</a>"
            )
        return jsonify({"error": "token exchange failed", "detail": result}), 400

    @app.route("/admin/ml_token_refresh", methods=["POST"])
    def ml_token_refresh():
        """Force ML OAuth token refresh to pick up new app permissions."""
        err = _check_admin()
        if err: return err
        import asyncio, httpx
        from ml_auth import _manager
        result = {}
        async def _run():
            _manager._token = None
            _manager._expires_at = 0.0
            async with httpx.AsyncClient(follow_redirects=True) as client:
                from ml_auth import get_auth_headers
                headers = await get_auth_headers(client)
                result["new_token_obtained"] = bool(headers)
        try:
            asyncio.run(_run())
        except RuntimeError:
            loop = asyncio.new_event_loop()
            loop.run_until_complete(_run())
            loop.close()
        return jsonify(result)

    @app.route("/admin/ml_debug")
    def ml_debug():
        """Diagnose ML scraper: test auth, run one sample query, show config."""
        err = _check_admin()
        if err: return err
        import asyncio, httpx, config as cfg
        result = {"ml_app_id_set": bool(cfg.ML_APP_ID), "ml_secret_set": bool(cfg.ML_CLIENT_SECRET)}

        async def _run():
            async with httpx.AsyncClient(follow_redirects=True) as client:
                from ml_auth import get_auth_headers
                headers = await get_auth_headers(client)
                result["auth_ok"] = bool(headers)
                if not headers:
                    result["mode"] = "HTML_FALLBACK (no API credentials)"
                    return
                # Test multiple endpoints to find what works
                test_urls = {
                    "search_with_auth": (
                        "https://api.mercadolibre.com/sites/MLA/search"
                        "?category=MLA1744&condition=used&q=toyota&limit=1"
                    ),
                    "search_no_auth": (
                        "https://api.mercadolibre.com/sites/MLA/search"
                        "?category=MLA1744&condition=used&q=toyota&limit=1"
                    ),
                    "item_with_auth": "https://api.mercadolibre.com/items/MLA2710424838",
                    "item_no_auth": "https://api.mercadolibre.com/items/MLA2710424838",
                }
                result["endpoint_tests"] = {}
                for name, url in test_urls.items():
                    use_auth = not name.endswith("_no_auth")
                    r = await client.get(url, headers=headers if use_auth else {}, timeout=10.0)
                    info = {"status": r.status_code, "auth_sent": use_auth}
                    if r.status_code == 200:
                        d = r.json()
                        info["total"] = d.get("paging", {}).get("total")
                        info["title"] = d.get("title")
                    else:
                        try:
                            info["error"] = r.json().get("message") or r.json().get("error")
                        except Exception:
                            pass
                    result["endpoint_tests"][name] = info
                search_ok = result["endpoint_tests"].get("search_with_auth", {}).get("status") == 200
                result["mode"] = "API_OAUTH2_WORKS" if search_ok else "API_AUTH_OK_BUT_SEARCH_FORBIDDEN"

        try:
            asyncio.run(_run())
        except RuntimeError:
            loop = asyncio.new_event_loop()
            loop.run_until_complete(_run())
            loop.close()
        return jsonify(result)

    @app.route("/admin/fix_overpriced", methods=["POST"])
    def fix_overpriced():
        """Clear is_deal flag for any listing with negative discount_pct."""
        err = _check_admin()
        if err: return err
        session = SessionLocal()
        try:
            updated = (
                session.query(Listing)
                .filter(Listing.is_deal == True, Listing.discount_pct < 0)
                .all()
            )
            count = 0
            for lst in updated:
                lst.is_deal = False
                count += 1
            session.commit()
            return jsonify({"fixed": count})
        finally:
            session.close()

    _scan_running = False

    @app.route("/admin/scan", methods=["POST"])
    def trigger_scan():
        """Trigger an immediate scan in a background thread."""
        err = _check_admin()
        if err: return err
        nonlocal _scan_running
        if _scan_running:
            return jsonify({"status": "already running"}), 409
        import threading
        def _run():
            nonlocal _scan_running
            _scan_running = True
            try:
                import asyncio
                from main import run_scan
                asyncio.run(run_scan())
            except Exception as e:
                import logging
                logging.getLogger(__name__).error(f"Manual scan failed: {e}", exc_info=True)
            finally:
                _scan_running = False
        t = threading.Thread(target=_run, daemon=True)
        t.start()
        return jsonify({"status": "scan started"})

    @app.route("/admin/scan_status")
    def scan_status():
        return jsonify({"running": _scan_running})

    @app.route("/unhide/<path:listing_id>", methods=["POST"])
    def unhide_listing(listing_id):
        session = SessionLocal()
        try:
            listing = session.query(Listing).filter_by(id=listing_id).first()
            if listing:
                listing.hidden = False
                listing.is_agency = False
                session.commit()
            return ("", 204)
        finally:
            session.close()

    @app.route("/hidden")
    def hidden_listings():
        session = SessionLocal()
        try:
            page = max(1, request.args.get("page", 1, type=int))
            query = session.query(Listing).filter(
                (Listing.hidden == True) | (Listing.is_agency == True)
            ).order_by(Listing.last_seen.desc())
            total = query.count()
            pages = max(1, math.ceil(total / 20))
            listings = query.offset((page - 1) * 20).limit(20).all()
            return render_template("index.html",
                listings=listings, total=total, page=page, pages=pages,
                filters={k: "" for k in ["brand","model","city","origin_city","min_year","max_year","max_km","min_score","max_distance"]},
                last_updated=None, new_today=0, avg_score=0,
                sources=[], unique_brands=[], unique_cities=[],
                origin_name=ORIGIN_NAME, origin_coords=(ORIGIN_LAT, ORIGIN_LON),
                all_origin_cities=sorted(CITY_COORDS.keys()), view="hidden")
        finally:
            session.close()

    @app.route("/listing/<path:listing_id>")
    def listing_detail(listing_id):
        session = SessionLocal()
        try:
            listing = session.query(Listing).filter_by(id=listing_id).first()
            if not listing:
                return render_template("detail.html", listing=None, error="Listing not found"), 404
            return render_template("detail.html", listing=listing, error=None)
        finally:
            session.close()

    @app.route("/api/comparables/<path:listing_id>")
    def api_comparables(listing_id):
        from database import PriceHistory
        from config import get_usd_mep_rate
        session = SessionLocal()
        try:
            listing = session.query(Listing).filter_by(id=listing_id).first()
            if not listing:
                return jsonify({"error": "not found"}), 404

            usd_rate = get_usd_mep_rate()
            year = listing.year or 2010
            brand = listing.brand or ""
            year_range = list(range(year - 2, year + 3))

            # Match comparables by base model + engine size when detectable.
            # This separates variants with very different price tiers
            # (e.g. Amarok 2.0 TDI ~$25k vs Amarok V6 3.0 ~$45k).
            import re as _re
            raw_model = (listing.model or "").strip()
            model_parts = raw_model.split()
            base_word = model_parts[0].lower() if model_parts else ""
            # Extract first displacement token like "2.0", "1.6", "3.0"
            engine = next(
                (tok for tok in model_parts[1:] if _re.match(r"^\d+\.\d+$", tok)),
                None,
            )
            from sqlalchemy import and_ as sa_and
            if base_word and engine:
                model_filter = sa_and(
                    Listing.model.ilike(f"%{base_word}%"),
                    Listing.model.ilike(f"%{engine}%"),
                )
            elif base_word:
                model_filter = Listing.model.ilike(f"%{base_word}%")
            else:
                model_filter = True

            rows = (
                session.query(Listing.price_usd_equiv, Listing.price_ars, Listing.year)
                .filter(
                    Listing.brand.ilike(f"%{brand}%"),
                    model_filter,
                    (Listing.price_ars > 0) | (Listing.price_usd_equiv > 0),
                    Listing.is_agency != True,
                    Listing.hidden != True,
                    Listing.status.in_(["active", "sold"]),
                    Listing.id != listing_id,
                    Listing.year.in_(year_range),
                )
                .all()
            )

            prices_usd = []
            for pusd, pars, yr in rows:
                usd = pusd if pusd and pusd > 0 else (pars / usd_rate if pars else None)
                if usd and usd > 500:
                    prices_usd.append(round(usd))

            cur_usd = listing.price_usd_equiv
            if not cur_usd and listing.price_ars:
                cur_usd = round(listing.price_ars / usd_rate)

            market_usd = None
            if listing.market_price_ars:
                market_usd = round(listing.market_price_ars / usd_rate)

            ph = [
                {
                    "date": h.recorded_at.isoformat(),
                    "price_ars": h.price_ars,
                    "event": h.event_type,
                }
                for h in listing.price_history
            ]

            return jsonify({
                "listing_price_usd": round(cur_usd) if cur_usd else None,
                "market_price_usd": market_usd,
                "comparables_usd": sorted(prices_usd),
                "sample_count": len(prices_usd),
                "ref_type": listing.ref_type,
                "confidence_index": listing.confidence_index,
                "percentile_rank": listing.percentile_rank,
                "discount_pct": listing.discount_pct,
                "price_history": ph,
            })
        finally:
            session.close()

    @app.route("/api/deals")
    def api_deals():
        session = SessionLocal()
        try:
            deals = (
                session.query(Listing)
                .filter(Listing.is_deal == True, Listing.status == "active")
                .order_by(Listing.score.desc())
                .limit(50)
                .all()
            )
            result = []
            for d in deals:
                result.append({
                    "id": d.id,
                    "source": d.source,
                    "title": d.title,
                    "brand": d.brand,
                    "model": d.model,
                    "year": d.year,
                    "km": d.km,
                    "price_ars": d.price_ars,
                    "price_usd": d.price_usd,
                    "fuel": d.fuel,
                    "transmission": d.transmission,
                    "condition": d.condition,
                    "url": d.url,
                    "thumbnail": d.thumbnail,
                    "score": d.score,
                    "discount_pct": d.discount_pct,
                    "is_deal": d.is_deal,
                    "deal_reason": d.deal_reason,
                    "first_seen": d.first_seen.isoformat() if d.first_seen else None,
                    "last_seen": d.last_seen.isoformat() if d.last_seen else None,
                })
            return jsonify(result)
        finally:
            session.close()

    @app.route("/api/stats")
    def api_stats():
        session = SessionLocal()
        try:
            from sqlalchemy import case
            rows = session.query(
                Listing.source,
                func.count(Listing.id).label("total"),
                func.sum(case((Listing.is_deal == True, 1), else_=0)).label("deals"),
                func.sum(case((Listing.ref_type == "cold", 1), else_=0)).label("cold"),
                func.sum(case((Listing.is_agency == True, 1), else_=0)).label("agency"),
            ).filter(Listing.status == "active").group_by(Listing.source).all()

            sources = [
                {"source": r.source, "total": r.total, "deals": r.deals or 0,
                 "cold": r.cold or 0, "agency": r.agency or 0}
                for r in rows
            ]
            total_active = sum(r["total"] for r in sources)
            total_deals  = sum(r["deals"] for r in sources)

            ref_dist = dict(session.query(Listing.ref_type, func.count(Listing.id))
                            .filter(Listing.status == "active")
                            .group_by(Listing.ref_type).all())

            return jsonify({
                "total_active": total_active,
                "total_deals": total_deals,
                "by_source": sources,
                "ref_type_distribution": ref_dist,
            })
        finally:
            session.close()

    return app
