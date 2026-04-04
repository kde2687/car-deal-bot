"""
MercadoLibre scraper — uses the official ML Search API with OAuth2.
Falls back to HTML scraping if API credentials are not configured.

API advantages over HTML scraping:
- No cookies needed (tokens auto-refresh every 6h)
- seller_type=private filters dealers at the source
- Structured JSON data — no brittle HTML parsing
- 18,000 requests/hour with auth (vs very low without)
"""
import asyncio
import json
import logging
import os
import re
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
PAGES_PER_BRAND = 12
REGIONAL_CITY_SLUGS: list[str] = [
    # Near Darregueira (<250km)
    "bahia-blanca", "punta-alta", "coronel-suarez", "santa-rosa",
    "tres-arroyos", "general-pico", "tandil", "olavarria", "azul",
    "pehuajo", "bolivar", "pigüe", "carhue", "guamini",
    "salliqueloo", "trenque-lauquen", "nueve-de-julio", "lincoln",
    # Major cities (for comparables)
    "mar-del-plata", "la-plata", "rosario", "cordoba", "mendoza",
    "neuquen", "san-luis", "rio-cuarto", "san-nicolas-de-los-arroyos",
    "junin", "pergamino", "venado-tuerto", "villa-maria",
]
REGIONAL_PAGES = 3
CATEGORY_PAGES = 12


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
                        if 1990 <= y <= 2030:
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
            if year and year < self.min_year:
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
            brand = brand_attr or (title.split()[0] if title else "")
            model = model_attr or title
            if model.lower().startswith(brand.lower()):
                model = model[len(brand):].strip()
            from scorer import _normalize_model as _nm
            model = _nm(model).title() if model else model

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
                },
            }
        except Exception as e:
            logger.debug(f"ML API result parse error: {e}")
            return None

    async def _fetch_api_listings(self, client: httpx.AsyncClient, auth_headers: dict) -> list:
        """Fetch listings using the ML Search API."""
        from ml_auth import get_auth_headers
        listings = []
        seen_ids: set = set()

        # Pre-filters applied at API level — each 1000-result cap contains only relevant listings
        api_attr_filters = (
            f"&VEHICLE_YEAR-from={self.min_year}"
            f"&KILOMETERS-from={config.MIN_KM}"
            f"&KILOMETERS-to={self.max_km}"
        )

        # Build search queries: one per brand + one general sweep
        queries = [(brand, f'q={brand.lower().replace(" ", "+")}') for brand in self.brands]
        queries.append(("(all)", ""))   # general sweep — catches unlisted brands

        for brand_name, q_param in queries:
            brand_count = 0
            offset = 0
            limit = 50
            max_results = 1000  # ML API hard cap per query

            while offset < max_results:
                params = (
                    f"category={ML_CATEGORY}"
                    f"&condition=used"
                    f"&sort=date_desc"
                    f"{api_attr_filters}"
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
                    if resp.status_code != 200:
                        logger.warning(f"ML API {resp.status_code} for {brand_name} offset {offset}")
                        break
                    data = resp.json()
                except Exception as e:
                    logger.warning(f"ML API error for {brand_name}: {e}")
                    break

                results = data.get("results", [])
                if not results:
                    break

                paging = data.get("paging", {})
                total = paging.get("total", 0)

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

            if brand_count:
                logger.info(f"ML API {brand_name}: {brand_count} listings")

        logger.info(f"ML API total: {len(listings)} listings")
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
        "tienda oficial",
        "s.a.", "s.r.l.", "s.r.l", " srl ",
        "multimarca", "cochería", "cocheria",
        "plan de ahorro",
    )

    def _is_agency(self, li_el) -> bool:
        seller_el = li_el.select_one(".poly-component__seller, [class*=seller-info]")
        if seller_el and seller_el.get_text(strip=True):
            return True
        for svg in li_el.find_all("svg"):
            if "oficial" in (svg.get("aria-label") or "").lower():
                return True
        card_text = li_el.get_text(" ", strip=True).lower()
        if any(kw in card_text for kw in self._AGENCY_KEYWORDS):
            return True
        for badge in li_el.select("[class*=badge],[class*=pill],[class*=label],[class*=tag]"):
            if "validado" in badge.get_text(strip=True).lower():
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
            href = link_el["href"] if link_el else ""
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
                if re.match(r"^\d{4}$", txt):
                    try:
                        y = int(txt)
                        if 1990 <= y <= 2030:
                            year = y
                    except ValueError:
                        pass
                elif "km" in txt.lower():
                    km = self._parse_km(txt)
            if year and year < self.min_year:
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
            model = title
            brand_lower = brand.lower()
            if model.lower().startswith(brand_lower):
                model = model[len(brand_lower):].strip()
            from scorer import _normalize_model as _nm
            model = _nm(model).title() if model else model
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
                "raw_data": {"title": title, "price_ars": price_ars, "price_usd": price_usd,
                             "currency": currency, "year": year, "km": km, "location": location_text, "url": href},
            }
        except Exception as e:
            logger.debug(f"Error parsing ML card: {e}")
            return None

    async def _fetch_page(self, client: httpx.AsyncClient, url: str, retries: int = 3) -> Optional[str]:
        delay = 2.0
        for attempt in range(retries):
            try:
                resp = await client.get(url, timeout=30.0)
                if resp.status_code == 429:
                    wait = min(delay * (2 ** attempt), 60.0)
                    logger.warning(f"ML rate limited, waiting {wait:.0f}s")
                    await asyncio.sleep(wait)
                    continue
                if resp.status_code == 404:
                    return None
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
        """HTML scraping fallback — used when API credentials are not set."""
        listings = []
        seen_ids: set = set()

        for brand in self.brands:
            brand_count = 0
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
                    if listing and listing["id"] not in seen_ids:
                        seen_ids.add(listing["id"])
                        listings.append(listing)
                        brand_count += 1
                await asyncio.sleep(2.5)
            logger.info(f"ML HTML {brand}: {brand_count} listings")

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
                        if listing and listing["id"] not in seen_ids:
                            seen_ids.add(listing["id"])
                            listings.append(listing)
                    await asyncio.sleep(2.0)

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
                brand_guess = title_el.get_text(strip=True).split()[0] if title_el else "unknown"
                listing = self._parse_card(card, brand_guess)
                if listing and listing["id"] not in seen_ids:
                    seen_ids.add(listing["id"])
                    listings.append(listing)
            await asyncio.sleep(2.5)

        logger.info(f"ML HTML total: {len(listings)} listings")
        return listings

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def fetch_listings(self) -> list:
        from ml_auth import get_auth_headers

        async with httpx.AsyncClient(follow_redirects=True) as client:
            auth_headers = await get_auth_headers(client)

            if auth_headers:
                logger.info("ML scraper: using API with OAuth2 (no cookies needed)")
                return await self._fetch_api_listings(client, auth_headers)
            else:
                logger.warning("ML scraper: no API credentials — falling back to HTML scraping with cookies")
                cookies = _load_cookies()
                if cookies:
                    logger.info(f"ML scraper: using session cookies ({len(cookies)} loaded)")
                else:
                    logger.warning("ML scraper: no cookies either — requests may be blocked")
                html_client = httpx.AsyncClient(
                    headers=ML_HEADERS, cookies=cookies, follow_redirects=True
                )
                async with html_client:
                    return await self._fetch_html_listings(html_client)
