"""
Post-scan enrichment for MercadoLibre listings.
Uses the public ML API to detect dealers via multiple signals:

  1. official_store_id  — non-null → always a store/agency
  2. active listing count — seller with >=DEALER_THRESHOLD active car listings
  3. seller_reputation  — high transaction count + old account → likely dealer

ML API (authenticated):
  GET https://api.mercadolibre.com/items/{MLA_ID}
      → {seller_id, official_store_id, tags, original_price, sale_price, ...}
  GET https://api.mercadolibre.com/items/{MLA_ID}/prices
      → {prices: [{type: "reference", amount: N, currency_id: "ARS"}, ...]}
      ← This is the same market median ML shows on the listing's bell-curve graph.
  GET https://api.mercadolibre.com/users/{seller_id}/items/search?limit=1&category=MLA1743
      → {paging: {total: N}}
  GET https://api.mercadolibre.com/users/{seller_id}
      → {registration_date, seller_reputation: {transactions: {completed: N}}}

Price reference enrichment (API-based — no HTML scraping needed):
  _get_ml_ref_price() is called inside _check_seller() using the already-fetched
  item data + one extra /prices call. Listings priced above ML's median reference
  are de-flagged as deals.
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
DEALER_THRESHOLD = 3           # sellers with >=3 active car listings = dealer
DEALER_COMPLETED_THRESHOLD = 20   # >20 completed car sales = professional dealer
DEALER_ACCOUNT_AGE_YEARS = 2      # account older than 2 years + high sales = dealer

# Keywords in seller nickname that indicate a dealership.
# IMPORTANT: all entries are matched as substrings of the lowercased nickname.
# Keep entries specific enough to avoid false positives:
#   - "auto" was REMOVED — it matches "automático", "autónomo", "autobiografía", etc.
#     Use "autos" (plural) and "automotor*" variants instead.
#   - "import" kept — "importadora", "importados" are dealer-specific in AR context.
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


async def _get_ml_ref_price(
    client: httpx.AsyncClient,
    mla_id: str,
    item_data: dict,
    auth: dict,
) -> Optional[float]:
    """
    Extract ML's own market reference price (the bell-curve median shown on listing pages).

    Tries in order:
      1. GET /items/{id}/prices  →  entry with type="reference" and currency_id="ARS"
         This is the same number ML displays in its price-distribution widget.
      2. item_data["original_price"]  — ML's "before discount" price when seller set one.
      3. sale_price.regular_amount vs sale_price.amount difference.
      4. Recursive key search through item_data for known reference field names.

    Returns ARS float or None if unavailable.
    """
    # Strategy 1: /prices endpoint — "reference" type is ML's market median
    try:
        resp = await client.get(
            f"{ML_API}/items/{mla_id}/prices", headers=auth, timeout=10.0
        )
        if resp.status_code == 200:
            for entry in (resp.json().get("prices") or []):
                if entry.get("type") == "reference" and entry.get("currency_id") == "ARS":
                    amt = entry.get("amount")
                    if isinstance(amt, (int, float)) and 1_000_000 <= amt <= 2_000_000_000:
                        logger.debug(f"ML ref price {mla_id}: ${amt:,.0f} ARS (from /prices)")
                        return float(amt)
    except Exception:
        pass

    # Strategy 2: original_price field (set when seller marks a "before" price)
    op = item_data.get("original_price")
    if isinstance(op, (int, float)) and 1_000_000 <= op <= 2_000_000_000:
        logger.debug(f"ML ref price {mla_id}: ${op:,.0f} ARS (from original_price)")
        return float(op)

    # Strategy 3: sale_price.regular_amount when it differs from current amount
    sp = item_data.get("sale_price") or {}
    regular = sp.get("regular_amount")
    current = sp.get("amount") or item_data.get("price")
    if (
        isinstance(regular, (int, float))
        and isinstance(current, (int, float))
        and regular > current
        and 1_000_000 <= regular <= 2_000_000_000
    ):
        logger.debug(f"ML ref price {mla_id}: ${regular:,.0f} ARS (from sale_price.regular_amount)")
        return float(regular)

    # Strategy 4: recursive search for known reference field names in item data
    found = _extract_ref_price_from_obj(item_data)
    if found:
        logger.debug(f"ML ref price {mla_id}: ${found:,.0f} ARS (from recursive key scan)")
    return found


async def _check_seller(
    client: httpx.AsyncClient, mla_id: str
) -> tuple[Optional[int], Optional[str], Optional[float]]:
    """
    Return (seller_id, reason, ref_price_ars) — seller_id/reason are None if not a dealer.
    ref_price_ars is ML's market median for this listing (None if unavailable).
    Returns (None, None, None) on any API error.
    """
    try:
        auth = await get_auth_headers(client)

        # Step 1: get item data
        resp = await client.get(f"{ML_API}/items/{mla_id}", headers=auth, timeout=15.0)
        if resp.status_code != 200:
            return None, None, None
        data = resp.json()
        seller_id = data.get("seller_id")
        if not seller_id:
            return None, None, None

        # Extract ML's market reference price now (reuses item data, +1 /prices call)
        ref_price = await _get_ml_ref_price(client, mla_id, data, auth)

        # Signal 0: check BlockedSeller table — manually flagged agencies.
        # This runs before all other signals to short-circuit expensive API calls.
        # The seller nickname is matched case-insensitively against the stored
        # normalised-lowercase names.  We defer the user fetch to Signal 3 so we
        # first try a cheap DB hit using the seller_id from item data.
        try:
            from database import SessionLocal as _SL, BlockedSeller as _BS
            _session = _SL()
            try:
                # We don't have the nickname yet — use the /users/{id} response cached
                # later in Signal 3.  Do a quick pre-check using seller_id as a string
                # in case the operator blocked by numeric ID stored as a string.
                seller_id_str = str(seller_id)
                _blocked_by_id = _session.query(_BS).filter(
                    _BS.seller_name == seller_id_str
                ).first()
                if _blocked_by_id:
                    return seller_id, f"vendedor bloqueado manualmente (id={seller_id})", ref_price
            finally:
                _session.close()
        except Exception:
            pass

        # Signal 1: official store (always an agency)
        if data.get("official_store_id") is not None:
            return seller_id, f"tienda oficial ML (store_id={data['official_store_id']})", ref_price

        # Signal 1-permalink: permalink contains "/tienda/" or "/official-stores/"
        # ML uses these URL patterns exclusively for official store / dealer listings.
        permalink = (data.get("permalink") or "").lower()
        if "/tienda/" in permalink or "/official-stores/" in permalink:
            return seller_id, f"permalink de tienda oficial ('{permalink[:60]}')", ref_price

        # Signal 1b: item-level tags (car_dealer, brand, etc.)
        item_tags = data.get("tags") or []
        if "car_dealer" in item_tags or "brand" in item_tags:
            return seller_id, f"item tag: {[t for t in item_tags if t in ('car_dealer','brand')]}", ref_price

        # Signal 1c: item title contains explicit dealer keywords.
        # Uses a SEPARATE (stricter) keyword set — not _DEALER_NICKNAME_KEYWORDS —
        # because title keywords must be unambiguous as standalone words in context:
        #   - "import" is excluded: "Toyota Hilux importada" would be a false positive.
        #   - "autos" is excluded: "compra venta de autos usados" — too generic.
        #   - "s.a.", "srl", etc. are excluded: too short/ambiguous as title substrings.
        #   - "usados" IS included as a phrase-level check (see below).
        # All checks are word-boundary aware to reduce substring false positives.
        title_lower = (data.get("title") or "").lower()
        _TITLE_DEALER_KEYWORDS = (
            "concesionaria", "concesionario", "concesion",
            "agencia", "agencias",
            "automotores", "automotriz",
            "cocheria", "cochería",
            "multimarca",
            "dealers", "dealer",
        )
        if any(kw in title_lower for kw in _TITLE_DEALER_KEYWORDS):
            matched_kw = next(kw for kw in _TITLE_DEALER_KEYWORDS if kw in title_lower)
            return seller_id, f"título contiene '{matched_kw}'", ref_price

        # Signal 1e: dealer store-name prefix BEFORE the brand in title
        # (e.g. "LOB Peugeot 2008..." or "ABC Toyota Corolla...").
        # Private sellers almost always start with brand or model; dealers prefix
        # their store code/abbreviation.
        # Heuristic: first word is a short alphabetic token (2-5 chars), not a brand,
        # not a common Spanish adjective/descriptor, and a car brand appears in words 2-4.
        # MAX LENGTH IS 5 (not 6) — "Toyota" and "Suzuki" are 6 chars; we must not
        # accidentally match brand names that appear at position 0 due to a data anomaly.
        # We also require the first word to be FULLY alphabetic (no digits, no punctuation)
        # so that "4x4 Toyota..." or "2020 Toyota..." do not trigger.
        try:
            import config as _cfg
            _brands_lower = {b.lower() for b in _cfg.BRANDS_LIST}
            # Words that private sellers commonly put at the start of titles.
            # MUST cover all Spanish descriptors ≤5 chars to avoid false positives.
            _common_prefix_words = {
                # condition / quality descriptors
                "usado", "usada", "nueva", "nuevo", "unico", "única", "único",
                "impecable", "ideal", "excelente", "hermoso", "precioso", "perfecto",
                "oportunidad", "urgente",
                # car descriptors ≤5 chars (the key false-positive candidates)
                "full",   # "Full Toyota..." — private seller highlighting trim level
                "semi",   # "Semi Toyota..."
                "gran",   # "Gran Toyota..." (uncommon but possible)
                "doble",  # "Doble tracción Toyota..."
                "turbo",  # "Turbo Fiat..."
                "nafta",  # "Nafta Ford..."
                "diesel", # "Diesel VW..."
                "color",  # "Color Toyota..."
                "super",  # "Super Chevrolet..."
                "extra",  # "Extra Toyota..."
                "motor",  # "Motor Toyota..." (unlikely but safe to block)
                "techo",  # "Techo Toyota..."
                "gnc",    # "GNC Fiat..." — fuel type prefix
                "titular", # "Titular Toyota..." — private seller note
                "con",    # "Con GNC Toyota..."
                "sin",    # "Sin deuda Ford..."
                "año",    # "Año 2020 Toyota..."
                "mas",    # "Mas equipado Ford..."
                "muy",    # "Muy buen Toyota..."
                "buen",   # "Buen Toyota..."
                "bien",   # "Bien Toyota..."
                "vendo",  # "Vendo Toyota..."
                "venta",  # "Venta Toyota..."
                "oferta", # "Oferta Toyota..."
                "precio", # "Precio Toyota..."
                "bajo",   # "Bajo km Toyota..."
                "poco",   # "Poco km Toyota..."
                "unico",  # duplicate without accent
                "tomo",   # "Tomo auto Toyota..." (trade-in offer)
                "permuto",# "Permuto Toyota..."
            }
            title_words = title_lower.split()
            if len(title_words) >= 2:
                first = title_words[0]
                # Conditions for a dealer prefix:
                #   1. Fully alphabetic (no digits, no punctuation like "/", ".")
                #   2. 2–5 chars (6 would include brand names like Toyota/Suzuki)
                #   3. Not a known car brand
                #   4. Not a common descriptor private sellers use
                #   5. A known brand appears in positions 2–4 of the title
                if (
                    first.isalpha()
                    and 2 <= len(first) <= 5
                    and first not in _brands_lower
                    and first not in _common_prefix_words
                    and any(w in _brands_lower for w in title_words[1:4])
                ):
                    return seller_id, f"título con prefijo de agencia antes de la marca ('{first}')", ref_price
        except Exception:
            pass

        # Signal 1d: description text contains dealer keywords
        resp_desc = await client.get(
            f"{ML_API}/items/{mla_id}/descriptions", headers=auth, timeout=15.0
        )
        if resp_desc.status_code == 200:
            descs = resp_desc.json()
            if isinstance(descs, list):
                for desc in descs:
                    text = (desc.get("plain_text") or desc.get("text") or "").lower()
                    _DESC_DEALER_KW = (
                        "concesionaria", "concesionario", "agencia de autos",
                        "agencia de vehiculos", "agencia de vehículos",
                        "somos una agencia", "somos concesionarios",
                        "datos de la concesion", "stock de vehiculos",
                        "ventas de vehiculos", "automotores",
                    )
                    if any(kw in text for kw in _DESC_DEALER_KW):
                        matched = next(kw for kw in _DESC_DEALER_KW if kw in text)
                        return seller_id, f"descripción menciona '{matched}'", ref_price

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
            if listing_count >= DEALER_THRESHOLD:
                return seller_id, f"{listing_count} autos en venta activos", ref_price

        # Signal 3: seller reputation — high completed transactions + old account
        resp3 = await client.get(f"{ML_API}/users/{seller_id}", headers=auth, timeout=15.0)
        if resp3.status_code == 200:
            udata = resp3.json()

            # Signal 3a: seller has a company profile (business account)
            if udata.get("company"):
                return seller_id, "cuenta empresa ML", ref_price

            # Signal 3b: seller tags include dealer indicators
            seller_tags = udata.get("tags") or []
            dealer_tags = [t for t in seller_tags if "dealer" in t.lower() or "car" in t.lower()]
            if dealer_tags:
                return seller_id, f"seller tags: {dealer_tags}", ref_price

            # Signal 3c: nickname contains dealer keywords
            nickname = (udata.get("nickname") or "").lower()
            matched_kw = next((kw for kw in _DEALER_NICKNAME_KEYWORDS if kw in nickname), None)
            if matched_kw:
                return seller_id, f"nickname contiene '{matched_kw}'", ref_price

            # Signal 3c-bis: nickname matches a manually blocked seller (case-insensitive).
            # The BlockedSeller table stores normalised lowercase names, so we compare
            # the already-lowercased nickname directly.
            if nickname:
                try:
                    from database import SessionLocal as _SL2, BlockedSeller as _BS2
                    _session2 = _SL2()
                    try:
                        _blocked = _session2.query(_BS2).filter(
                            _BS2.seller_name == nickname
                        ).first()
                        if _blocked:
                            return seller_id, f"vendedor bloqueado manualmente (nick='{nickname}')", ref_price
                    finally:
                        _session2.close()
                except Exception:
                    pass

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
                ), ref_price

        return None, None, ref_price

    except Exception as e:
        logger.debug(f"ML API error for {mla_id}: {e}")
        return None, None, None


# ---------------------------------------------------------------------------
# ML price reference — HTML scraping
# ---------------------------------------------------------------------------

def _extract_ref_price_from_obj(obj, _depth: int = 0) -> Optional[float]:
    """
    Recursively search a nested dict/list for ML's reference price value.
    Returns the first plausible ARS amount (1 000 000 – 2 000 000 000 range).

    Skips sub-trees that are known to contain unrelated prices (seller info,
    shipping costs, installment details) to avoid returning false reference prices.
    """
    if _depth > 12:
        return None

    # Sub-tree keys that are known to hold non-market prices.  Skipping them
    # prevents shipping costs, installment amounts, or seller reputation values
    # from being mistaken for ML's market reference price.
    _SKIP_KEYS = frozenset({
        "seller", "shipping", "installments", "fees", "buyer_fees",
        "seller_fees", "taxes", "cost", "rate", "deals", "promotions",
    })

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

        for k, v in obj.items():
            if k in _SKIP_KEYS:
                continue  # do not descend into non-price sub-trees
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
    agencies_found: list[tuple[str, int, str]] = []        # (lid, seller_id, reason)
    ref_prices_found: list[tuple[str, float]] = []         # (lid, ref_price_ars)

    async def check_one(lid: str, mla_id: str):
        async with sem:
            seller_id, reason, ref_price = await _check_seller(api_client, mla_id)
            if seller_id is not None and reason:
                agencies_found.append((lid, seller_id, reason))
                logger.debug(
                    f"ML enrich: {mla_id} seller {seller_id} → dealer ({reason})"
                )
            if ref_price is not None:
                ref_prices_found.append((lid, ref_price))
                logger.debug(f"ML enrich: {mla_id} ref price ${ref_price/1e6:.1f}M ARS")
            await asyncio.sleep(0.2)

    async with httpx.AsyncClient(follow_redirects=True) as api_client:
        await asyncio.gather(*[check_one(lid, mla_id) for lid, mla_id in pairs])

    agency_lids: set[str] = set()
    session = SessionLocal()
    try:
        # Phase 1: mark agencies
        for lid, seller_id, reason in agencies_found:
            listing = session.query(Listing).filter_by(id=lid).first()
            if listing and not listing.is_agency:
                listing.is_agency = True
                listing.is_deal = False
                listing.deal_reason = (
                    f"Filtrado: agencia ML (vendedor {seller_id} — {reason})"
                )
                agency_lids.add(lid)
                logger.info(f"ML enrich: {lid} → agencia (seller {seller_id}, {reason})")

        # Phase 2: apply ML's reference prices (API-based — no HTML scraping needed)
        ref_updated = 0
        ref_deflags = 0
        for lid, ref_price in ref_prices_found:
            if lid in agency_lids:
                continue   # already handled as agency
            listing = session.query(Listing).filter_by(id=lid).first()
            if not listing or listing.hidden:
                continue

            listing.market_price_ars = ref_price   # always update with ML's data

            # De-flag if listing price is above ML's market median (with 2% tolerance)
            listing_price = listing.price_ars or 0
            if listing_price > 0 and listing_price > ref_price * 1.02:
                pct_above = (listing_price - ref_price) / ref_price * 100
                if listing.is_deal:
                    listing.is_deal = False
                    listing.deal_reason = (
                        f"Sobre precio ML: ${listing_price/1e6:.1f}M > "
                        f"ref ${ref_price/1e6:.1f}M (+{pct_above:.0f}%)"
                    )
                    ref_deflags += 1
                    logger.info(
                        f"ML ref price: {lid} de-flagged — "
                        f"${listing_price/1e6:.1f}M > ref ${ref_price/1e6:.1f}M (+{pct_above:.0f}%)"
                    )
            ref_updated += 1

        session.commit()
        logger.info(
            f"ML enrich: {len(pairs)} checked | {len(agencies_found)} agencias | "
            f"{ref_updated} ref prices applied | {ref_deflags} de-flagged as overpriced"
        )
    except Exception as e:
        session.rollback()
        logger.error(f"ML enrich DB write failed: {e}")
    finally:
        session.close()

    # Secondary HTML pass: scan listing page text for concesionaria signals and
    # extract ML's reference price from the price-distribution widget.
    # Only run for non-agency listings to avoid wasting bandwidth.
    agency_lids_set = {lid for lid, _, _ in agencies_found}
    html_pairs = [(lid, mla_id) for lid, mla_id in pairs if lid not in agency_lids_set]
    if html_pairs:
        try:
            await _enrich_reference_prices(html_pairs)
        except Exception as e:
            logger.warning(f"ML enrich HTML pass failed: {e}")

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


# ---------------------------------------------------------------------------
# Active listing status verification
# ---------------------------------------------------------------------------

async def check_ml_listing_statuses() -> int:
    """
    Batch-query ML API for the status of all active ML listings in the DB.
    Marks listings with status != 'active' (e.g. 'closed', 'paused') as sold.

    Fixes the case where ML's search API still returns a listing that is no
    longer available on the listing page — the scraper keeps refreshing
    last_seen so _mark_sold_listings never fires.

    ML batch endpoint: GET /items?ids=MLA1,MLA2,...  (max 20 per request)
    Response: [{code: 200, body: {id, status, ...}}, ...]
    Returns count of listings marked sold.
    """
    from database import PriceHistory

    session = SessionLocal()
    try:
        rows = session.query(Listing.id).filter(
            Listing.source == "mercadolibre",
            Listing.status == "active",
            Listing.hidden != True,
        ).all()
        db_ids = [r.id for r in rows]
    finally:
        session.close()

    if not db_ids:
        return 0

    pairs = []
    for lid in db_ids:
        mla_id = lid.replace("meli:", "")
        if mla_id.startswith("MLA"):
            pairs.append((lid, mla_id))

    if not pairs:
        return 0

    BATCH = 20
    inactive: list[tuple[str, str]] = []  # (db_id, ml_status)

    async with httpx.AsyncClient(follow_redirects=True) as client:
        for i in range(0, len(pairs), BATCH):
            batch = pairs[i:i + BATCH]
            mla_ids_str = ",".join(mla_id for _, mla_id in batch)
            # Refresh auth headers per-batch so an expired token is renewed automatically
            # via MLTokenManager rather than failing silently for all remaining batches.
            try:
                auth = await get_auth_headers(client)
            except Exception:
                auth = {}
            try:
                resp = await client.get(
                    f"{ML_API}/items",
                    params={"ids": mla_ids_str, "attributes": "id,status"},
                    headers=auth,
                    timeout=15.0,
                )
                if resp.status_code != 200:
                    logger.debug(f"ML status batch HTTP {resp.status_code}")
                    continue
                results = resp.json()
                # ML returns one result entry per requested ID, in the same order
                # as the comma-separated ids parameter.  We use index-based mapping
                # as a fallback for 404 entries where body.id may be absent.
                results_list = results if isinstance(results, list) else []
                id_to_status: dict[str, str] = {}
                for idx, item in enumerate(results_list):
                    if not isinstance(item, dict):
                        continue
                    item_code = item.get("code")
                    body = item.get("body") or {}
                    item_id = body.get("id")
                    # Try to get item_id from positional mapping when body lacks "id"
                    # (ML returns empty/error body for 404 items in some API versions).
                    if not item_id and idx < len(batch):
                        item_id = batch[idx][1]  # mla_id at same position
                    if not item_id:
                        continue
                    if item_code == 404:
                        # 404 = listing deleted/not found on ML — treat as inactive.
                        id_to_status[item_id] = "not_found"
                    else:
                        status = body.get("status")
                        if status:
                            id_to_status[item_id] = status
                for db_id, mla_id in batch:
                    status = id_to_status.get(mla_id)
                    if status and status != "active":
                        inactive.append((db_id, status))
            except Exception as e:
                logger.debug(f"ML status batch error: {e}")
            await asyncio.sleep(0.3)

    if not inactive:
        logger.debug(f"ML status check: all {len(pairs)} listings active")
        return 0

    session = SessionLocal()
    try:
        now = datetime.utcnow()
        marked = 0
        for db_id, ml_status in inactive:
            listing = session.query(Listing).filter_by(id=db_id).first()
            if not listing or listing.status != "active":
                continue
            listing.status = "sold"
            listing.sold_at = now
            listing.is_deal = False
            session.add(PriceHistory(
                listing_id=db_id,
                price_ars=listing.price_ars,
                price_usd_equiv=listing.price_usd_equiv,
                recorded_at=now,
                days_on_market=(now - listing.first_seen).days if listing.first_seen else 0,
                event_type="sold",
            ))
            marked += 1
            logger.info(f"ML status check: {db_id} → '{ml_status}' (marked sold)")
        if marked:
            session.commit()
        logger.info(f"ML status check: {marked} inactive listings marked sold out of {len(pairs)} checked")
        return marked
    except Exception as e:
        session.rollback()
        logger.error(f"ML status check DB write failed: {e}")
        return 0
    finally:
        session.close()
