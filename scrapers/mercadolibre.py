"""
MercadoLibre scraper — uses the official ML Search API with OAuth2.
Falls back to HTML scraping if API credentials are not configured.

API advantages over HTML scraping:
- No cookies needed (tokens auto-refresh every 6h)
- dealers fetched too (is_agency flag set) — they serve as price comparables
- Structured JSON data — no brittle HTML parsing
- 18,000 requests/hour with auth (vs very low without)
"""
import asyncio
import json
import logging
import os
import re
from datetime import datetime
from typing import Optional

import httpx
from bs4 import BeautifulSoup
import config

logger = logging.getLogger(__name__)

ML_API = "https://api.mercadolibre.com"
ML_SITE = "MLA"
ML_CATEGORY = "MLA1744"   # Autos y Camionetas — Argentina

ML_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "es-AR,es;q=0.9",
    "Referer": "https://www.mercadolibre.com.ar/",
}

_COOKIES_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "ml_cookies.json")


def _load_cookies() -> dict:
    """Load ML session cookies from env var ML_COOKIES_JSON or ml_cookies.json file."""
    raw = None
    env_json = os.environ.get("ML_COOKIES_JSON", "")
    if env_json:
        try:
            raw = json.loads(env_json)
        except Exception:
            pass
    if raw is None:
        try:
            with open(_COOKIES_FILE) as f:
                raw = json.load(f)
        except Exception:
            return {}
    try:
        return {c["name"]: c["value"] for c in raw if "mercadolibre.com.ar" in c.get("domain", "")}
    except Exception:
        return {}


# HTML scraper constants (fallback when no API credentials)
PAGES_PER_BRAND = 12         # 12 pages × 48 cards = 576 listings max per brand (matches Autocosmos)
HTML_CONCURRENCY = 4         # concurrent brand fetches
HTML_PAGE_DELAY  = 0.5       # seconds between pages within one brand (halved — scans complete in ~70s)
# Regional pages: 2 pages per major city to capture province-level inventory
# Actual scans complete in ~70s vs 420s budget — plenty of headroom for regional pass
REGIONAL_CITY_SLUGS: list[str] = ["buenos-aires", "cordoba", "rosario", "mendoza"]
REGIONAL_PAGES = 2
CATEGORY_PAGES = 20          # increased from 8 — captures more non-brand listings


class MercadoLibreScraper:
    BASE_URL = "https://autos.mercadolibre.com.ar"

    def __init__(self, brands: list, min_year: int, max_km: int):
        self.brands = brands
        self.min_year = min_year
        self.max_km = max_km

    # ------------------------------------------------------------------
    # API-based scraping (primary — requires ML_APP_ID + ML_CLIENT_SECRET)
    # ------------------------------------------------------------------

    def _api_result_to_listing(self, item: dict) -> Optional[dict]:
        """Convert ML API search result to our listing dict format."""
        try:
            item_id = item.get("id", "")
            if not item_id:
                return None

            title = item.get("title", "")
            price_value = item.get("price")
            currency = item.get("currency_id", "ARS")
            permalink = item.get("permalink", "")
            thumbnail = item.get("thumbnail", "")

            price_ars = price_value if currency == "ARS" else None
            price_usd = price_value if currency == "USD" else None

            # Price sanity
            if currency == "ARS":
                if not price_ars or price_ars < config.MIN_PRICE_ARS or price_ars > config.MAX_PRICE_ARS:
                    return None
            elif currency == "USD":
                if not price_usd or price_usd < config.MIN_PRICE_USD or price_usd > config.MAX_PRICE_USD:
                    return None
            else:
                return None  # unknown currency — skip

            # Attributes: year, km, fuel, transmission
            year = None
            km = None
            fuel = ""
            transmission = ""
            brand_attr = ""
            model_attr = ""

            for attr in item.get("attributes", []):
                attr_id = attr.get("id", "")
                val = attr.get("value_name") or ""
                if attr_id == "VEHICLE_YEAR":
                    try:
                        y = int(val)
                        if 1990 <= y <= datetime.utcnow().year + 2:
                            year = y
                    except (ValueError, TypeError):
                        pass
                elif attr_id == "KILOMETERS":
                    try:
                        km = int(re.sub(r"[^\d]", "", val))
                    except (ValueError, TypeError):
                        pass
                elif attr_id == "FUEL_TYPE":
                    fuel = val
                elif attr_id == "TRANSMISSION":
                    transmission = val
                elif attr_id == "BRAND":
                    brand_attr = val
                elif attr_id == "MODEL":
                    model_attr = val

            # Filters
            if not year:
                return None  # No year in API response — can't score reliably
            if year < self.min_year:
                return None
            if km is not None and km > self.max_km:
                return None
            if km is not None and km < config.MIN_KM:
                return None

            # Year-based minimum price
            if currency == "ARS" and price_ars and year:
                if price_ars < config.min_price_for_year(year):
                    return None

            # Location
            location = item.get("location") or {}
            city = ""
            if isinstance(location, dict):
                city_obj = location.get("city") or {}
                city = city_obj.get("name", "") if isinstance(city_obj, dict) else ""

            # Brand / model
            brand = brand_attr or (next(iter(title.split()), "") if title else "")
            model = model_attr or title
            if brand and model.lower().startswith(brand.lower()):
                model = model[len(brand):].strip()
            from scorer import _normalize_model as _nm
            # _nm may return "" for purely numeric models (e.g. Peugeot "208");
            # fall back to first token of the raw model string so the field is never empty.
            if model:
                nm_result = _nm(model)
                model = (nm_result or model.split()[0]).title()
            # Guard: if model is still empty use first title token as last resort
            if not model and title:
                model = title.split()[0].title()

            # Capture ML's original_price (before seller discount) and sale_price
            # as early market reference signals before enrichment runs
            original_price = item.get("original_price")
            sale_price_obj = item.get("sale_price") or {}
            ml_ref_price_hint = None
            if isinstance(original_price, (int, float)) and original_price > 0:
                ml_ref_price_hint = float(original_price)
            elif (sale_price_obj.get("regular_amount") and sale_price_obj.get("amount")
                  and sale_price_obj["regular_amount"] > sale_price_obj["amount"]):
                ml_ref_price_hint = float(sale_price_obj["regular_amount"])

            return {
                "id": f"meli:{item_id}",
                "source": "mercadolibre",
                "title": title,
                "brand": brand.title(),
                "model": model,
                "year": year,
                "km": km,
                "price_ars": price_ars,
                "price_usd": price_usd,
                "fuel": fuel,
                "transmission": transmission,
                "condition": "used" if (km or 0) > 0 else "new",
                "url": permalink.split("?")[0] if permalink else "",
                "thumbnail": thumbnail,
                "seller_city": city,
                "raw_data": {
                    "title": title,
                    "price_ars": price_ars,
                    "price_usd": price_usd,
                    "currency": currency,
                    "year": year,
                    "km": km,
                    "location": city,
                    "url": permalink,
                    "ml_original_price": original_price,
                    "ml_ref_price_hint": ml_ref_price_hint,
                },
            }
        except Exception as e:
            logger.debug(f"ML API result parse error: {e}")
            return None

    async def _fetch_api_listings(self, client: httpx.AsyncClient, auth_headers: dict) -> Optional[list]:
        """Fetch listings using the ML Search API.
        Returns None if the API is inaccessible (403/401) so caller can fall back to HTML.
        """
        from ml_auth import get_auth_headers
        listings = []
        seen_ids: set = set()
        _api_accessible: Optional[bool] = None  # None = not yet tested

        # Split into year windows to beat the 1000-result API cap.
        # Without this, sort=date_desc fills the cap with all years and
        # local year/km filtering leaves very few listings.
        current_year = datetime.utcnow().year
        year_windows = [
            (self.min_year, self.min_year + 2),
            (self.min_year + 3, self.min_year + 5),
            (self.min_year + 6, current_year + 1),
        ]

        # Build search queries: one per brand per year window + one general sweep
        queries = []
        for brand in self.brands:
            q = f'q={brand.lower().replace(" ", "+")}'
            for yr_from, yr_to in year_windows:
                queries.append((f"{brand} {yr_from}-{yr_to}", q, yr_from, yr_to))
        # General sweep for unlisted brands — full year range
        queries.append(("(all)", "", self.min_year, current_year + 1))

        for brand_name, q_param, yr_from, yr_to in queries:
            brand_count = 0
            offset = 0
            limit = 50
            max_results = 1000  # ML API hard cap per query
            transient_errors = 0  # guard against infinite retry on persistent server errors
            MAX_TRANSIENT_ERRORS = 3

            window_filters = (
                f"&VEHICLE_YEAR-from={yr_from}"
                f"&VEHICLE_YEAR-to={yr_to}"
            )

            while offset < max_results:
                params = (
                    f"category={ML_CATEGORY}"
                    f"&condition=used"
                    f"&sort=date_desc"
                    f"{window_filters}"
                    f"&limit={limit}"
                    f"&offset={offset}"
                )
                if q_param:
                    params += f"&{q_param}"

                url = f"{ML_API}/sites/{ML_SITE}/search?{params}"
                try:
                    resp = await client.get(url, headers=auth_headers, timeout=20.0)
                    if resp.status_code == 429:
                        logger.warning("ML API rate limited, waiting 30s")
                        await asyncio.sleep(30)
                        continue
                    if resp.status_code in (401, 403):
                        logger.warning(
                            f"ML API {resp.status_code} — app lacks search permissions, "
                            f"switching to HTML fallback"
                        )
                        return None  # signal to caller: fall back to HTML
                    if resp.status_code in (500, 502, 503, 504):
                        # Transient server-side error — retry after short delay instead of
                        # aborting the entire page window (which would silently drop all
                        # listings at this offset and beyond).
                        transient_errors += 1
                        if transient_errors > MAX_TRANSIENT_ERRORS:
                            logger.warning(
                                f"ML API {resp.status_code} persists for {brand_name} "
                                f"after {MAX_TRANSIENT_ERRORS} retries — skipping window"
                            )
                            break
                        logger.warning(
                            f"ML API {resp.status_code} (transient) for {brand_name} "
                            f"offset {offset} — retrying in 10s ({transient_errors}/{MAX_TRANSIENT_ERRORS})"
                        )
                        await asyncio.sleep(10)
                        continue
                    if resp.status_code != 200:
                        logger.warning(f"ML API {resp.status_code} for {brand_name} offset {offset}")
                        break
                    data = resp.json()
                    transient_errors = 0  # reset on successful response
                except Exception as e:
                    transient_errors += 1
                    if transient_errors > MAX_TRANSIENT_ERRORS:
                        logger.warning(
                            f"ML API network error persists for {brand_name} "
                            f"after {MAX_TRANSIENT_ERRORS} retries: {e} — skipping window"
                        )
                        break
                    logger.warning(f"ML API error for {brand_name}: {e} — retrying in 5s ({transient_errors}/{MAX_TRANSIENT_ERRORS})")
                    await asyncio.sleep(5)
                    continue

                results = data.get("results", [])
                paging = data.get("paging", {})
                total = paging.get("total", 0)

                if offset == 0:
                    logger.info(
                        f"ML API [{brand_name} {yr_from}-{yr_to}]: "
                        f"{total} total in API, fetching up to {min(total, max_results)}"
                    )

                if not results:
                    break

                for item in results:
                    # Detect agencies via API seller data — mark them, don't skip.
                    # Agencies are stored as market comparables (real price data) but
                    # excluded from deal detection via is_agency=True flag.
                    is_agency = False
                    seller = item.get("seller", {}) or {}
                    seller_type = item.get("seller_type") or seller.get("seller_type") or ""
                    eshop = seller.get("eshop")
                    car_dealer = seller.get("car_dealer")
                    if seller_type in ("car_dealer", "real_estate_agency") or eshop or car_dealer:
                        is_agency = True
                    listing = self._api_result_to_listing(item)
                    if listing and listing["id"] not in seen_ids:
                        listing["is_agency"] = is_agency
                        seen_ids.add(listing["id"])
                        listings.append(listing)
                        brand_count += 1

                offset += limit
                if offset >= min(total, max_results):
                    break

                await asyncio.sleep(0.3)   # polite delay — well within 18k/hr limit

            logger.info(f"ML API [{brand_name} {yr_from}-{yr_to}]: {brand_count} passed local filters")

        agencies = sum(1 for l in listings if l.get("is_agency"))
        private = len(listings) - agencies
        logger.info(
            f"ML API TOTAL: {len(listings)} listings "
            f"({private} private, {agencies} agencies) "
            f"from {len(queries)} queries"
        )
        return listings

    # ------------------------------------------------------------------
    # HTML scraping (fallback — used when no API credentials)
    # ------------------------------------------------------------------

    def _brand_slug(self, brand: str) -> str:
        return brand.lower().replace(" ", "-").replace("á","a").replace("é","e").replace("í","i").replace("ó","o").replace("ú","u").replace("ü","u").replace("ë","e").replace("ñ","n")

    def _page_url(self, brand: str, offset: int) -> str:
        slug = self._brand_slug(brand)
        if offset == 0:
            return f"{self.BASE_URL}/{slug}/_VendedorTipo_U_" if slug else f"{self.BASE_URL}/_VendedorTipo_U_"
        return f"{self.BASE_URL}/{slug}/_VendedorTipo_U__Desde_{offset + 1}" if slug else f"{self.BASE_URL}/_VendedorTipo_U__Desde_{offset + 1}"

    def _regional_url(self, brand: str, city_slug: str, page: int = 0) -> str:
        slug = self._brand_slug(brand)
        if page == 0:
            return f"{self.BASE_URL}/{slug}/{city_slug}/_VendedorTipo_U_"
        offset = page * 48
        return f"{self.BASE_URL}/{slug}/{city_slug}/_VendedorTipo_U__Desde_{offset + 1}"

    def _parse_price(self, text: str) -> Optional[float]:
        if not text:
            return None
        digits = re.sub(r"[^\d]", "", text)
        return float(digits) if digits else None

    def _detect_currency(self, price_container) -> str:
        if not price_container:
            return "ARS"
        amount_el = price_container.select_one("[aria-label]")
        if amount_el:
            aria = amount_el.get("aria-label", "").lower()
            if "dólar" in aria or "dollar" in aria or "usd" in aria:
                return "USD"
        symbol_el = price_container.select_one(".andes-money-amount__currency-symbol")
        if symbol_el and "us" in symbol_el.get_text(strip=True).lower():
            return "USD"
        return "ARS"

    _AGENCY_KEYWORDS = (
        "automotor", "automotores", "automotriz",
        "concesion", "concesionaria", "concesionario",
        "agencia", "dealer",
        "usados certificados",
        "vehículo validado", "vehiculo validado",
        "tienda oficial", "tienda",
        "s.a.", "s.r.l.", "s.r.l", " srl ", "s.a.s",
        "multimarca", "cochería", "cocheria",
        "plan de ahorro",
        "grupo ", "group ", "motors ", "autos ",
        "certificado", "garantía de fábrica", "garantia de fabrica",
        "ver más de este vendedor", "ver mas de este vendedor",
        "ver vehículos", "ver vehiculos",
    )

    def _is_agency(self, li_el) -> bool:
        # Any dedicated seller element = dealer (private sellers don't have this)
        seller_el = li_el.select_one(
            ".poly-component__seller, [class*=seller-info], [class*=seller__], "
            "[class*=seller-name], [data-testid*=seller]"
        )
        if seller_el and seller_el.get_text(strip=True):
            return True
        # Official store SVG icon
        for svg in li_el.find_all("svg"):
            if "oficial" in (svg.get("aria-label") or "").lower():
                return True
        # Full-text keyword scan
        card_text = li_el.get_text(" ", strip=True).lower()
        if any(kw in card_text for kw in self._AGENCY_KEYWORDS):
            return True
        # Any badge/tag/pill mentioning "validado", "garantia", "oficial", "certificado"
        for badge in li_el.select("[class*=badge],[class*=pill],[class*=label],[class*=tag]"):
            t = badge.get_text(strip=True).lower()
            if any(w in t for w in ("validado", "garantia", "garantía", "oficial", "certificado")):
                return True
        # "Ver más vehículos de" link = seller with multiple listings = dealer
        for a in li_el.find_all("a", href=True):
            txt = a.get_text(strip=True).lower()
            if "más vehículos" in txt or "mas vehiculos" in txt or "más autos" in txt:
                return True
        return False

    def _parse_km(self, text: str) -> Optional[int]:
        if not text:
            return None
        digits = re.sub(r"[^\d]", "", text)
        return int(digits) if digits else None

    def _extract_item_id(self, href: str) -> Optional[str]:
        m = re.search(r"MLA-?(\d+)", href)
        return f"MLA{m.group(1)}" if m else None

    _FINANCING_KEYWORDS = (
        "financiado", "facilidades de pago",
        "plan de ahorro", "plan ahorro", "financiaci",
        "precio anticipo", "valor anticipo",
    )

    def _parse_card(self, li_el, brand: str) -> Optional[dict]:
        try:
            if config.ONLY_PRIVATE_SELLERS and self._is_agency(li_el):
                return None
            card_text_lower = li_el.get_text(" ", strip=True).lower()
            if any(kw in card_text_lower for kw in self._FINANCING_KEYWORDS):
                return None
            title_el = li_el.select_one(".poly-component__title, .ui-search-item__title, h2")
            title = title_el.get_text(strip=True) if title_el else ""
            price_container = li_el.select_one(".poly-component__price, .ui-search-price")
            price_el = li_el.select_one(".andes-money-amount__fraction, .price-tag-fraction")
            price_raw = price_el.get_text(strip=True) if price_el else ""
            price_value = self._parse_price(price_raw)
            currency = self._detect_currency(price_container or li_el)
            price_ars = price_value if currency == "ARS" else None
            price_usd = price_value if currency == "USD" else None
            if currency == "ARS":
                if price_ars is None or price_ars < config.MIN_PRICE_ARS or price_ars > config.MAX_PRICE_ARS:
                    return None
            elif currency == "USD":
                if price_usd is None or price_usd < config.MIN_PRICE_USD or price_usd > config.MAX_PRICE_USD:
                    return None
            link_el = li_el.select_one("a[href*='mercadolibre']")
            href = link_el.get("href", "") if link_el else ""
            item_id = self._extract_item_id(href)
            if not item_id:
                return None
            img_el = li_el.select_one("img")
            thumbnail = ""
            if img_el:
                thumbnail = img_el.get("data-src") or img_el.get("src") or ""
            attr_items = li_el.select(".poly-attributes_list__item, .poly-component__attributes-list li")
            attr_texts = [el.get_text(strip=True) for el in attr_items]
            year = None
            km = None
            for txt in attr_texts:
                # Match standalone 4-digit year OR year followed by extra text ("2020 modelo")
                year_m = re.search(r'\b(19\d{2}|20[0-3]\d)\b', txt)
                if year_m and not year:
                    try:
                        y = int(year_m.group(1))
                        if 1990 <= y <= datetime.utcnow().year + 2:
                            year = y
                    except ValueError:
                        pass
                if "km" in txt.lower():
                    km = self._parse_km(txt)
            if not year:
                # Last-resort: try to extract year from card title text
                title_el = li_el.select_one(".poly-component__title, h2, [class*=title]")
                title_txt = title_el.get_text(strip=True) if title_el else ""
                year_m2 = re.search(r'\b(19\d{2}|20[0-3]\d)\b', title_txt)
                if year_m2:
                    try:
                        y = int(year_m2.group(1))
                        if 1990 <= y <= datetime.utcnow().year + 2:
                            year = y
                    except ValueError:
                        pass
            if not year:
                return None  # no year = can't score reliably (matches API scraper behaviour)
            if year < self.min_year:
                return None
            if km is not None and km > self.max_km:
                return None
            if km is not None and km < config.MIN_KM:
                return None
            if currency == "ARS" and price_ars and year:
                if price_ars < config.min_price_for_year(year):
                    return None
            loc_el = li_el.select_one(".poly-component__location, [class*=location], [class*=city]")
            location_text = loc_el.get_text(strip=True) if loc_el else ""
            city = location_text.split(" - ")[0].strip() if location_text else ""

            # Capture seller display name — present on dealer/agency cards, absent on private
            seller_el = li_el.select_one(
                ".poly-component__seller, [class*=seller-info], [class*=seller__], "
                "[class*=seller-name], [data-testid*=seller]"
            )
            seller_name = seller_el.get_text(strip=True) if seller_el else ""

            model = title
            brand_lower = brand.lower()
            if brand_lower and model.lower().startswith(brand_lower):
                model = model[len(brand_lower):].strip()
            from scorer import _normalize_model as _nm
            # _nm may return "" for purely numeric models (e.g. Peugeot "208");
            # fall back to first token of the raw model string so the field is never empty.
            if model:
                nm_result = _nm(model)
                model = (nm_result or model.split()[0]).title()
            # Guard: if model is still empty use first title token as last resort
            if not model and title:
                model = title.split()[0].title()
            return {
                "id": f"meli:{item_id}",
                "source": "mercadolibre",
                "title": title,
                "brand": brand.title(),
                "model": model,
                "year": year,
                "km": km,
                "price_ars": price_ars,
                "price_usd": price_usd,
                "fuel": "",
                "transmission": "",
                "condition": "used" if (km or 0) > 0 else "new",
                "url": href.split("#")[0],
                "thumbnail": thumbnail,
                "seller_city": city,
                "raw_data": {
                    "title": title, "price_ars": price_ars, "price_usd": price_usd,
                    "currency": currency, "year": year, "km": km,
                    "location": location_text, "url": href,
                    "seller_name": seller_name,
                },
            }
        except Exception as e:
            logger.debug(f"Error parsing ML card: {e}")
            return None

    async def _fetch_page(self, client: httpx.AsyncClient, url: str, retries: int = 2) -> Optional[str]:
        delay = 1.5
        for attempt in range(retries):
            try:
                resp = await client.get(url, timeout=15.0)
                if resp.status_code == 429:
                    wait = min(delay * (2 ** attempt), 60.0)
                    logger.warning(f"ML rate limited, waiting {wait:.0f}s")
                    await asyncio.sleep(wait)
                    continue
                if resp.status_code in (403, 404):
                    return None  # don't retry blocks or missing pages
                resp.raise_for_status()
                return resp.text
            except httpx.HTTPStatusError as e:
                logger.warning(f"ML HTTP {e.response.status_code} for {url}")
                if attempt < retries - 1:
                    await asyncio.sleep(delay * (2 ** attempt))
            except Exception as e:
                logger.warning(f"ML request error for {url}: {e}")
                if attempt < retries - 1:
                    await asyncio.sleep(delay * (2 ** attempt))
        return None

    async def _fetch_html_listings(self, client: httpx.AsyncClient) -> list:
        """HTML scraping fallback — parallel brand fetching with concurrency limit."""
        listings: list = []
        seen_ids: set = set()
        lock = asyncio.Lock()
        sem = asyncio.Semaphore(HTML_CONCURRENCY)

        async def _scrape_brand(brand: str):
            brand_listings = []
            for page in range(PAGES_PER_BRAND):
                offset = page * 48
                url = self._page_url(brand, offset)
                html = await self._fetch_page(client, url)
                if not html:
                    break
                soup = BeautifulSoup(html, "lxml")
                cards = soup.select("li.ui-search-layout__item")
                if not cards:
                    break
                for card in cards:
                    listing = self._parse_card(card, brand)
                    if listing:
                        brand_listings.append(listing)
                await asyncio.sleep(HTML_PAGE_DELAY)

            for city_slug in REGIONAL_CITY_SLUGS:
                for rpage in range(REGIONAL_PAGES):
                    url = self._regional_url(brand, city_slug, rpage)
                    html = await self._fetch_page(client, url)
                    if not html:
                        break
                    soup = BeautifulSoup(html, "lxml")
                    cards = soup.select("li.ui-search-layout__item")
                    if not cards:
                        break
                    for card in cards:
                        listing = self._parse_card(card, brand)
                        if listing:
                            brand_listings.append(listing)
                    await asyncio.sleep(HTML_PAGE_DELAY)

            async with lock:
                added = 0
                for lst in brand_listings:
                    if lst["id"] not in seen_ids:
                        seen_ids.add(lst["id"])
                        listings.append(lst)
                        added += 1
            logger.info(f"ML HTML {brand}: {added} new listings")

        async def _scrape_brand_throttled(brand: str):
            async with sem:
                await _scrape_brand(brand)

        await asyncio.gather(*[_scrape_brand_throttled(b) for b in self.brands])

        # Category sweep
        for page in range(CATEGORY_PAGES):
            offset = page * 48
            url = f"{self.BASE_URL}/_VendedorTipo_U_" if offset == 0 else f"{self.BASE_URL}/_VendedorTipo_U__Desde_{offset + 1}"
            html = await self._fetch_page(client, url)
            if not html:
                break
            soup = BeautifulSoup(html, "lxml")
            cards = soup.select("li.ui-search-layout__item")
            if not cards:
                break
            for card in cards:
                title_el = card.select_one(".poly-component__title, .ui-search-item__title, h2")
                brand_guess = next(iter(title_el.get_text(strip=True).split()), "unknown") if title_el else "unknown"
                listing = self._parse_card(card, brand_guess)
                if listing and listing["id"] not in seen_ids:
                    seen_ids.add(listing["id"])
                    listings.append(listing)
            await asyncio.sleep(HTML_PAGE_DELAY)

        logger.info(f"ML HTML total: {len(listings)} listings")
        return listings

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def fetch_listings(self) -> list:
        from ml_auth import get_auth_headers

        import random
        if config.ML_PROXY_URLS:
            proxy = random.choice(config.ML_PROXY_URLS)
            logger.info(f"ML scraper: using proxy (pool of {len(config.ML_PROXY_URLS)})")
        elif config.ML_PROXY_URL:
            proxy = config.ML_PROXY_URL
            logger.info("ML scraper: using outbound proxy")
        else:
            proxy = None
        client_kwargs = dict(follow_redirects=True, proxy=proxy)

        async with httpx.AsyncClient(**client_kwargs) as client:
            auth_headers = await get_auth_headers(client)

            if auth_headers:
                logger.info(
                    f"ML scraper: trying API with OAuth2 | "
                    f"brands={len(self.brands)} min_year={self.min_year} max_km={self.max_km}"
                )
                api_result = await self._fetch_api_listings(client, auth_headers)
                if api_result is not None:
                    return api_result
                logger.warning("ML scraper: API returned 403 — falling back to HTML scraping")
            else:
                logger.warning(
                    f"ML scraper: NO API credentials (ML_APP_ID set={bool(config.ML_APP_ID)}) "
                    f"— falling back to HTML scraping"
                )

        # HTML fallback — runs whether API had credentials or not
        cookies = _load_cookies()
        if cookies:
            logger.info(f"ML scraper: using session cookies ({len(cookies)} loaded)")
        else:
            logger.info("ML scraper: HTML fallback (no cookies)")
        html_client = httpx.AsyncClient(
            headers=ML_HEADERS, cookies=cookies, follow_redirects=True, proxy=proxy
        )
        async with html_client:
            return await self._fetch_html_listings(html_client)
