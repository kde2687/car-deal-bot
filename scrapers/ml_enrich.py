"""
Post-scan enrichment for MercadoLibre listings.
Uses the public ML API to detect dealers via multiple signals:

  1. official_store_id  — non-null → always a store/agency
  2. active listing count — seller with >DEALER_THRESHOLD active car listings
  3. seller_reputation  — high transaction count + old account → likely dealer

ML API (no auth required):
  GET https://api.mercadolibre.com/items/{MLA_ID}
      → {seller_id, official_store_id, tags, ...}
  GET https://api.mercadolibre.com/users/{seller_id}/items/search?limit=1&category=MLA1743
      → {paging: {total: N}}   (MLA1743 = Autos y Camionetas, Argentina)
  GET https://api.mercadolibre.com/users/{seller_id}
      → {registration_date, seller_reputation: {transactions: {completed: N}}}

Price reference enrichment (HTML scraping — works from Railway):
  Fetches https://auto.mercadolibre.com.ar/{MLA_ID} and extracts ML's own
  "precio de referencia" from the embedded __NEXT_DATA__ JSON.
  Listings priced above ML's median reference are de-flagged as deals.
"""
import asyncio
import json
import logging
import random
import re
from datetime import datetime, timezone
from typing import Optional

import httpx

from database import SessionLocal, Listing
from ml_auth import get_auth_headers

logger = logging.getLogger(__name__)

ML_API = "https://api.mercadolibre.com"
ML_CARS_CATEGORY = "MLA1743"   # Autos y Camionetas — Argentina
DEALER_THRESHOLD = 5           # sellers with >5 active car listings = dealer
DEALER_COMPLETED_THRESHOLD = 20   # >20 completed car sales = professional dealer
DEALER_ACCOUNT_AGE_YEARS = 2      # account older than 2 years + high sales = dealer

# Keywords in seller nickname that indicate a dealership
_DEALER_NICKNAME_KEYWORDS = (
    "automotor", "automotores", "automotriz", "automoviles", "automóviles",
    "concesion", "concesionaria", "concesionario",
    "agencia", "agencias", "dealer", "dealers",
    "usados", "vehiculos", "vehículos", "cocheria", "cochería",
    "multimarca", "s.a.", "srl", "s.r.l", "sa.",
    "motors", "autos", "cars", "import",
)
CONCURRENCY = 8                # parallel API requests

# Browser-like headers for HTML listing page requests
_HTML_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/123.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "es-AR,es;q=0.9,en;q=0.7",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
}


def _get_proxy() -> Optional[str]:
    """Return a random proxy URL from config, or None."""
    try:
        import config as cfg
        pool = cfg.ML_PROXY_URLS or ([cfg.ML_PROXY_URL] if cfg.ML_PROXY_URL else [])
        return random.choice(pool) if pool else None
    except Exception:
        return None


async def _check_seller(
    client: httpx.AsyncClient, mla_id: str
) -> tuple[Optional[int], Optional[str]]:
    """
    Return (seller_id, reason) if seller is a dealer, else (None, None).
    reason is a human-readable string explaining why they were flagged.
    Returns (None, None) if not a dealer or on any API error.
    """
    try:
        auth = await get_auth_headers(client)

        # Step 1: get item data
        resp = await client.get(f"{ML_API}/items/{mla_id}", headers=auth, timeout=15.0)
        if resp.status_code != 200:
            return None, None
        data = resp.json()
        seller_id = data.get("seller_id")
        if not seller_id:
            return None, None

        # Signal 1: official store (always an agency)
        if data.get("official_store_id") is not None:
            return seller_id, f"tienda oficial ML (store_id={data['official_store_id']})"

        # Signal 1b: item-level tags (car_dealer, brand, etc.)
        item_tags = data.get("tags") or []
        if "car_dealer" in item_tags or "brand" in item_tags:
            return seller_id, f"item tag: {[t for t in item_tags if t in ('car_dealer','brand')]}"

        # Signal 1c: description text contains dealer keywords
        resp_desc = await client.get(
            f"{ML_API}/items/{mla_id}/descriptions", headers=auth, timeout=15.0
        )
        if resp_desc.status_code == 200:
            descs = resp_desc.json()
            if isinstance(descs, list):
                for desc in descs:
                    text = (desc.get("plain_text") or desc.get("text") or "").lower()
                    if any(kw in text for kw in ("concesionaria", "concesionario", "datos de la concesion", "agencia de autos")):
                        return seller_id, "descripción menciona concesionaria"

        # Signal 2: count active car listings
        resp2 = await client.get(
            f"{ML_API}/users/{seller_id}/items/search",
            params={"limit": 1, "category": ML_CARS_CATEGORY},
            headers=auth,
            timeout=15.0,
        )
        listing_count = 0
        if resp2.status_code == 200:
            listing_count = resp2.json().get("paging", {}).get("total", 0)
            if listing_count > DEALER_THRESHOLD:
                return seller_id, f"{listing_count} autos en venta activos"

        # Signal 3: seller reputation — high completed transactions + old account
        resp3 = await client.get(f"{ML_API}/users/{seller_id}", headers=auth, timeout=15.0)
        if resp3.status_code == 200:
            udata = resp3.json()

            # Signal 3a: seller has a company profile (business account)
            if udata.get("company"):
                return seller_id, f"cuenta empresa ML"

            # Signal 3b: seller tags include dealer indicators
            seller_tags = udata.get("tags") or []
            dealer_tags = [t for t in seller_tags if "dealer" in t.lower() or "car" in t.lower()]
            if dealer_tags:
                return seller_id, f"seller tags: {dealer_tags}"

            # Signal 3c: nickname contains dealer keywords
            nickname = (udata.get("nickname") or "").lower()
            matched_kw = next((kw for kw in _DEALER_NICKNAME_KEYWORDS if kw in nickname), None)
            if matched_kw:
                return seller_id, f"nickname contiene '{matched_kw}'"

            # Signal 3d: high transaction count + old account
            completed = (
                udata.get("seller_reputation", {})
                .get("transactions", {})
                .get("completed", 0) or 0
            )
            reg_date_str = udata.get("registration_date", "")
            account_age_years = 0
            if reg_date_str:
                try:
                    reg_dt = datetime.fromisoformat(reg_date_str.replace("Z", "+00:00"))
                    account_age_years = (
                        datetime.now(timezone.utc) - reg_dt
                    ).days / 365.25
                except Exception:
                    pass

            if (
                completed >= DEALER_COMPLETED_THRESHOLD
                and account_age_years >= DEALER_ACCOUNT_AGE_YEARS
            ):
                return seller_id, (
                    f"vendedor profesional: {completed} ventas, "
                    f"{account_age_years:.0f} años en ML"
                )

        return None, None

    except Exception as e:
        logger.debug(f"ML API error for {mla_id}: {e}")
        return None, None


# ---------------------------------------------------------------------------
# ML price reference — HTML scraping
# ---------------------------------------------------------------------------

def _extract_ref_price_from_obj(obj, _depth: int = 0) -> Optional[float]:
    """
    Recursively search a nested dict/list for ML's reference price value.
    Returns the first plausible ARS amount (1 000 000 – 2 000 000 000 range).
    """
    if _depth > 12:
        return None

    if isinstance(obj, dict):
        # Direct key matches for known ML price reference field names
        for key in (
            "reference_price", "suggested_price", "typical_price",
            "market_price", "price_reference", "median_price",
            "reference_value", "market_reference",
        ):
            val = obj.get(key)
            if isinstance(val, (int, float)) and 1_000_000 <= val <= 2_000_000_000:
                return float(val)
            # Some ML responses nest amount under {"amount": N, "currency_id": "ARS"}
            if isinstance(val, dict):
                amount = val.get("amount") or val.get("value")
                if isinstance(amount, (int, float)) and 1_000_000 <= amount <= 2_000_000_000:
                    currency = val.get("currency_id", "ARS")
                    if currency == "ARS":
                        return float(amount)

        for v in obj.values():
            result = _extract_ref_price_from_obj(v, _depth + 1)
            if result:
                return result

    elif isinstance(obj, list):
        for item in obj:
            result = _extract_ref_price_from_obj(item, _depth + 1)
            if result:
                return result

    return None


def _is_dealer_html(html: str) -> Optional[str]:
    """
    Check a full ML listing HTML page for dealer/concesionaria signals.
    Returns a reason string if dealer detected, else None.

    This is a second-pass filter — catches dealers that aren't flagged in
    search result cards (polycard format sometimes omits seller labels).
    """
    from bs4 import BeautifulSoup

    # Fast text-level check before parsing HTML
    html_lower = html.lower()
    _DEALER_TEXT_SIGNALS = (
        "concesionaria", "concesionario",
        "agencia de autos", "agencia de vehículos",
        "agencia de vehiculos",
        "tienda oficial",
        "car_dealer",          # JSON field in embedded scripts
        "\"seller_type\":\"car_dealer\"",
        "\"is_car_dealer\":true",
        "\"power_seller_status\"",
    )
    for sig in _DEALER_TEXT_SIGNALS:
        if sig in html_lower:
            return f"página menciona '{sig}'"

    # Structured check on the page DOM for seller-type labels
    try:
        soup = BeautifulSoup(html, "lxml")
        _SELLER_LABEL_SELECTORS = [
            ".ui-seller-info__subtitle-label",
            ".ui-seller-info__status-label",
            "[class*=seller-info__subtitle]",
            "[class*=seller-info__status]",
            "[class*=seller-info__title]",
            "[class*=seller-type]",
        ]
        for sel in _SELLER_LABEL_SELECTORS:
            el = soup.select_one(sel)
            if el:
                text = el.get_text(strip=True).lower()
                if any(kw in text for kw in ("concesionaria", "concesionario", "agencia", "dealer", "tienda")):
                    return f"etiqueta vendedor: '{el.get_text(strip=True)}'"

        # "Ver más vehículos de" link — only dealers have this
        for a in soup.find_all("a", href=True):
            txt = a.get_text(strip=True).lower()
            if any(p in txt for p in ("más vehículos", "mas vehiculos", "más autos", "mas autos",
                                       "ver más del vendedor", "ver mas del vendedor")):
                return f"link multi-vehículo: '{a.get_text(strip=True)}'"
    except Exception:
        pass

    return None


async def _fetch_listing_page_data(
    client: httpx.AsyncClient, mla_id: str
) -> tuple[Optional[float], Optional[str]]:
    """
    Fetch an ML listing HTML page and extract:
      - ML's own reference price (ARS) from the price distribution widget
      - Dealer reason string (or None if not a dealer)

    Returns (reference_price_ars, dealer_reason).
    Both can be None if unavailable.
    """
    url = f"https://auto.mercadolibre.com.ar/{mla_id}"
    try:
        resp = await client.get(url, timeout=15.0)
        if resp.status_code != 200:
            logger.debug(f"Reference price: HTTP {resp.status_code} for {mla_id}")
            return None, None
        html = resp.text

        # Check for dealer signals first (cheap text scan)
        dealer_reason = _is_dealer_html(html)

        ref_price: Optional[float] = None

        # Strategy 1: __NEXT_DATA__ (Next.js full page state — most reliable)
        m = re.search(
            r'<script[^>]*id="__NEXT_DATA__"[^>]*>\s*(\{.*?\})\s*</script>',
            html, re.DOTALL
        )
        if m:
            try:
                data = json.loads(m.group(1))
                ref_price = _extract_ref_price_from_obj(data)
                if ref_price:
                    logger.debug(f"Reference price for {mla_id}: ${ref_price:,.0f} ARS (from __NEXT_DATA__)")
            except (json.JSONDecodeError, Exception):
                pass

        # Strategy 2: window.__PRELOADED_STATE__ or similar inline state vars
        if not ref_price:
            for state_pattern in (
                r'window\.__PRELOADED_STATE__\s*=\s*(\{.+?\});\s*(?:window|</script>)',
                r'window\.__INITIAL_STATE__\s*=\s*(\{.+?\});\s*(?:window|</script>)',
            ):
                m2 = re.search(state_pattern, html, re.DOTALL)
                if m2:
                    try:
                        data = json.loads(m2.group(1))
                        ref_price = _extract_ref_price_from_obj(data)
                        if ref_price:
                            logger.debug(f"Reference price for {mla_id}: ${ref_price:,.0f} ARS (from preloaded state)")
                            break
                    except (json.JSONDecodeError, Exception):
                        pass

        # Strategy 3: bare regex scan for known field names with ARS-range integers
        if not ref_price:
            for field_pattern in (
                r'"reference_price"\s*:\s*(\d{7,11})',
                r'"suggested_price"\s*:\s*(\d{7,11})',
                r'"typical_price"\s*:\s*(\d{7,11})',
                r'"market_price"\s*:\s*(\d{7,11})',
                r'"median_price"\s*:\s*(\d{7,11})',
            ):
                m3 = re.search(field_pattern, html)
                if m3:
                    val = float(m3.group(1))
                    if 1_000_000 <= val <= 2_000_000_000:
                        logger.debug(f"Reference price for {mla_id}: ${val:,.0f} ARS (from regex scan)")
                        ref_price = val
                        break

        if not ref_price:
            logger.debug(f"Reference price: no data found for {mla_id}")

        return ref_price, dealer_reason

    except Exception as e:
        logger.debug(f"Listing page fetch error for {mla_id}: {e}")
        return None, None


# ---------------------------------------------------------------------------
# Main enrichment entry point
# ---------------------------------------------------------------------------

async def enrich_ml_new_listings(listing_ids: list[str]) -> int:
    """
    Check seller data for new ML listings via the public API.
    Also fetches ML's own reference price from HTML listing pages and
    de-flags deal listings that are priced above ML's market median.

    Marks is_agency=True and is_deal=False when a dealer signal is found.
    Updates market_price_ars when ML's reference price is available.
    Returns count of agencies found.
    """
    if not listing_ids:
        return 0

    # Extract raw MLA IDs
    pairs: list[tuple[str, str]] = []  # (listing_db_id, mla_id)
    for lid in listing_ids:
        mla_id = lid.replace("meli:", "")
        if mla_id.startswith("MLA"):
            pairs.append((lid, mla_id))

    if not pairs:
        return 0

    sem = asyncio.Semaphore(CONCURRENCY)
    agencies_found: list[tuple[str, int, str]] = []  # (lid, seller_id, reason)

    async def check_one(lid: str, mla_id: str):
        async with sem:
            seller_id, reason = await _check_seller(api_client, mla_id)
            if seller_id is not None and reason:
                agencies_found.append((lid, seller_id, reason))
                logger.debug(
                    f"ML enrich: {mla_id} seller {seller_id} → dealer ({reason})"
                )
            await asyncio.sleep(0.2)

    async with httpx.AsyncClient(follow_redirects=True) as api_client:
        await asyncio.gather(*[check_one(lid, mla_id) for lid, mla_id in pairs])

    if agencies_found:
        session = SessionLocal()
        try:
            for lid, seller_id, reason in agencies_found:
                listing = session.query(Listing).filter_by(id=lid).first()
                if listing and not listing.is_agency:
                    listing.is_agency = True
                    listing.is_deal = False
                    listing.deal_reason = (
                        f"Filtrado: agencia ML (vendedor {seller_id} — {reason})"
                    )
                    logger.info(
                        f"ML enrich: {lid} → agencia (seller {seller_id}, {reason})"
                    )
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"ML enrich DB write failed: {e}")
        finally:
            session.close()

    logger.info(
        f"ML enrich: checked {len(pairs)} listings, found {len(agencies_found)} agencies"
    )

    # --- Phase 2: reference price enrichment via HTML scraping ---
    agency_lids = {lid for lid, _, _ in agencies_found}
    # Only check listings not already flagged as agencies (no point enriching hidden listings)
    ref_pairs = [(lid, mla_id) for lid, mla_id in pairs if lid not in agency_lids]

    if ref_pairs:
        await _enrich_reference_prices(ref_pairs)

    return len(agencies_found)


async def _enrich_reference_prices(pairs: list[tuple[str, str]]) -> None:
    """
    For each listing, fetch ML's listing HTML page and:
      - Extract ML's own reference price → update market_price_ars
      - De-flag deals priced above ML's market median
      - Detect concesionaria signals missed in search results → mark as agency
    """
    proxy = _get_proxy()
    client_kwargs: dict = {
        "headers": _HTML_HEADERS,
        "follow_redirects": True,
    }
    if proxy:
        client_kwargs["proxy"] = proxy

    sem = asyncio.Semaphore(4)  # conservative HTML concurrency
    # (lid, ref_price_or_None, dealer_reason_or_None)
    results: list[tuple[str, Optional[float], Optional[str]]] = []

    async def fetch_one(lid: str, mla_id: str):
        async with sem:
            ref_price, dealer_reason = await _fetch_listing_page_data(html_client, mla_id)
            results.append((lid, ref_price, dealer_reason))
            await asyncio.sleep(0.5)

    async with httpx.AsyncClient(**client_kwargs) as html_client:
        await asyncio.gather(*[fetch_one(lid, mla_id) for lid, mla_id in pairs])

    price_data = [(lid, p, d) for lid, p, d in results if p is not None or d is not None]
    if not price_data:
        logger.debug("Reference price enrichment: no data retrieved")
        return

    logger.info(
        f"Reference price enrichment: got data for {len(price_data)}/{len(pairs)} listings"
    )

    session = SessionLocal()
    overpriced_count = 0
    updated_count = 0
    late_agency_count = 0
    try:
        for lid, ref_price, dealer_reason in price_data:
            listing = session.query(Listing).filter_by(id=lid).first()
            if not listing or listing.hidden:
                continue

            # Late agency detection — concesionaria signal found on listing page
            if dealer_reason and not listing.is_agency:
                listing.is_agency = True
                listing.is_deal = False
                listing.deal_reason = f"Filtrado: agencia (HTML) — {dealer_reason}"
                logger.info(f"ML enrich HTML: {lid} → agencia ({dealer_reason})")
                late_agency_count += 1
                continue  # no point updating market price for an agency

            if listing.is_agency:
                continue

            if ref_price:
                # Always update market_price_ars with ML's reference (better data source)
                listing.market_price_ars = ref_price
                updated_count += 1

                listing_price = listing.price_ars or 0.0
                if listing_price > 0 and listing_price > ref_price:
                    pct_above = (listing_price - ref_price) / ref_price * 100
                    if listing.is_deal:
                        listing.is_deal = False
                        listing.deal_reason = (
                            f"Sobre precio de referencia ML: "
                            f"${listing_price/1e6:.1f}M > ref ${ref_price/1e6:.1f}M "
                            f"(+{pct_above:.0f}%)"
                        )
                        logger.info(
                            f"ML ref price: {lid} de-flagged — "
                            f"price ${listing_price/1e6:.1f}M > ref ${ref_price/1e6:.1f}M "
                            f"(+{pct_above:.0f}%)"
                        )
                        overpriced_count += 1

        session.commit()
        logger.info(
            f"Reference price enrichment: {updated_count} market prices updated, "
            f"{overpriced_count} overpriced deals removed, "
            f"{late_agency_count} late agency detections"
        )
    except Exception as e:
        session.rollback()
        logger.error(f"Reference price DB write failed: {e}")
    finally:
        session.close()
