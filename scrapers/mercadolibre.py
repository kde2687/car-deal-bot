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
    # 1. Try environment variable first (works on Railway / cloud deployments)
    env_json = os.environ.get("ML_COOKIES_JSON", "")
    if env_json:
        try:
            raw = json.loads(env_json)
        except Exception:
            pass
    # 2. Fall back to local file
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

# Pages per brand (48 items/page). 12 pages = ~576 listings per brand → richer comparables DB.
PAGES_PER_BRAND = 12

# Regional city slugs for ML — scrape these in addition to the national search.
# Covers cities near Darregueira + major population centers for comparables.
REGIONAL_CITY_SLUGS: list[str] = [
    # Near Darregueira (<250km)
    "bahia-blanca",      # ~130km
    "punta-alta",        # ~150km
    "coronel-suarez",    # ~95km
    "santa-rosa",        # ~166km (La Pampa capital)
    "tres-arroyos",      # ~200km
    "general-pico",      # ~236km
    "tandil",            # ~240km
    "olavarria",         # ~210km
    "azul",              # ~195km
    "pehuajo",           # ~229km
    "bolivar",           # ~175km
    "pigüe",             # ~90km
    "carhue",            # ~80km
    "guamini",           # ~60km
    "salliqueloo",       # ~115km
    "trenque-lauquen",   # ~160km
    "nueve-de-julio",    # ~200km
    "lincoln",           # ~210km
    # Major cities (for comparables — more listings = better market reference)
    "mar-del-plata",     # ~330km
    "la-plata",          # ~480km
    "rosario",           # ~500km
    "cordoba",           # ~600km
    "mendoza",           # ~830km
    "neuquen",           # ~830km
    "san-luis",          # ~620km
    "rio-cuarto",        # ~490km
    "san-nicolas-de-los-arroyos",  # ~520km
    "junin",             # ~360km
    "pergamino",         # ~430km
    "venado-tuerto",     # ~430km
    "villa-maria",       # ~450km
]
REGIONAL_PAGES = 3   # pages per city-brand combo (3 × 48 = up to 144 per city)


class MercadoLibreScraper:
    BASE_URL = "https://autos.mercadolibre.com.ar"

    def __init__(self, brands: list, min_year: int, max_km: int):
        self.brands = brands
        self.min_year = min_year
        self.max_km = max_km

    def _brand_slug(self, brand: str) -> str:
        return brand.lower().replace(" ", "-").replace("á","a").replace("é","e").replace("í","i").replace("ó","o").replace("ú","u").replace("ü","u").replace("ë","e").replace("ñ","n")

    def _page_url(self, brand: str, offset: int) -> str:
        slug = self._brand_slug(brand)
        # All pages keep _VendedorTipo_U_ to exclude dealers at URL level
        if offset == 0:
            return f"{self.BASE_URL}/{slug}/_VendedorTipo_U_"
        return f"{self.BASE_URL}/{slug}/_VendedorTipo_U__Desde_{offset + 1}"

    def _regional_url(self, brand: str, city_slug: str, page: int = 0) -> str:
        """Return URL for regional city search with private-seller filter."""
        slug = self._brand_slug(brand)
        if page == 0:
            return f"{self.BASE_URL}/{slug}/{city_slug}/_VendedorTipo_U_"
        offset = page * 48
        return f"{self.BASE_URL}/{slug}/{city_slug}/_VendedorTipo_U__Desde_{offset + 1}"

    def _parse_price(self, text: str) -> Optional[float]:
        """Parse '$45.000.000' or '45000000' → 45000000.0"""
        if not text:
            return None
        digits = re.sub(r"[^\d]", "", text)
        return float(digits) if digits else None

    def _detect_currency(self, price_container) -> str:
        """Return 'USD' or 'ARS' by reading aria-label or currency symbol."""
        if not price_container:
            return "ARS"
        # aria-label is most reliable: "28500 dólares" vs "45000000 pesos argentinos"
        amount_el = price_container.select_one("[aria-label]")
        if amount_el:
            aria = amount_el.get("aria-label", "").lower()
            if "dólar" in aria or "dollar" in aria or "usd" in aria:
                return "USD"
        # Fallback: currency symbol element
        symbol_el = price_container.select_one(".andes-money-amount__currency-symbol")
        if symbol_el and "us" in symbol_el.get_text(strip=True).lower():
            return "USD"
        return "ARS"

    # Strong dealer indicators — keywords that appear in agency names or ML dealer badges
    # NOTE: keep these specific to avoid false positives on private sellers.
    # Removed: "garage" (private sellers mention "tiene garage"), standalone "certificados"
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
        """Return True if the listing belongs to a dealer/store, not a private seller."""
        # 1. poly-component__seller — present only on dealer cards, not private sellers
        seller_el = li_el.select_one(".poly-component__seller, [class*=seller-info]")
        if seller_el and seller_el.get_text(strip=True):
            return True

        # 2. "Tienda oficial" SVG aria-label anywhere in card
        for svg in li_el.find_all("svg"):
            if "oficial" in (svg.get("aria-label") or "").lower():
                return True

        # 3. Dealer keyword in card text
        card_text = li_el.get_text(" ", strip=True).lower()
        if any(kw in card_text for kw in self._AGENCY_KEYWORDS):
            return True

        # 4. "VALIDADO" certification badge (dealers only)
        for badge in li_el.select("[class*=badge],[class*=pill],[class*=label],[class*=tag]"):
            if "validado" in badge.get_text(strip=True).lower():
                return True

        return False

    def _parse_km(self, text: str) -> Optional[int]:
        """Parse '23.000 Km' → 23000"""
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
            # --- Agency filter ---
            if config.ONLY_PRIVATE_SELLERS and self._is_agency(li_el):
                return None

            # --- Financing/anticipo filter ---
            card_text_lower = li_el.get_text(" ", strip=True).lower()
            if any(kw in card_text_lower for kw in self._FINANCING_KEYWORDS):
                return None

            # Title
            title_el = li_el.select_one(
                ".poly-component__title, .ui-search-item__title, h2"
            )
            title = title_el.get_text(strip=True) if title_el else ""

            # Price + currency
            price_container = li_el.select_one(".poly-component__price, .ui-search-price")
            price_el = li_el.select_one(".andes-money-amount__fraction, .price-tag-fraction")
            price_raw = price_el.get_text(strip=True) if price_el else ""
            price_value = self._parse_price(price_raw)
            currency = self._detect_currency(price_container or li_el)

            price_ars = price_value if currency == "ARS" else None
            price_usd = price_value if currency == "USD" else None

            # --- Price sanity filter ---
            if currency == "ARS":
                if price_ars is None:
                    return None
                if price_ars < config.MIN_PRICE_ARS or price_ars > config.MAX_PRICE_ARS:
                    return None
            elif currency == "USD":
                if price_usd is None:
                    return None
                if price_usd < config.MIN_PRICE_USD or price_usd > config.MAX_PRICE_USD:
                    return None

            # Link & item ID
            link_el = li_el.select_one("a[href*='mercadolibre']")
            href = link_el["href"] if link_el else ""
            item_id = self._extract_item_id(href)
            if not item_id:
                return None

            # Thumbnail
            img_el = li_el.select_one("img")
            thumbnail = ""
            if img_el:
                thumbnail = img_el.get("data-src") or img_el.get("src") or ""

            # Attributes list: year + km
            attr_items = li_el.select(
                ".poly-attributes_list__item, .poly-component__attributes-list li"
            )
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

            # Year / km filters
            if year and year < self.min_year:
                return None
            if km is not None and km > self.max_km:
                return None
            if km is not None and km < config.MIN_KM:
                return None

            # Year-based minimum price (anticipo/plan de ahorro trap)
            if currency == "ARS" and price_ars is not None and year:
                if price_ars < config.min_price_for_year(year):
                    return None

            # Location
            loc_el = li_el.select_one(
                ".poly-component__location, [class*=location], [class*=city]"
            )
            location_text = loc_el.get_text(strip=True) if loc_el else ""
            city = location_text.split(" - ")[0].strip() if location_text else ""

            # Model from title (strip brand prefix, then normalize to base model)
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
                "raw_data": {
                    "title": title,
                    "price_ars": price_ars,
                    "price_usd": price_usd,
                    "currency": currency,
                    "year": year,
                    "km": km,
                    "location": location_text,
                    "url": href,
                },
            }
        except Exception as e:
            logger.debug(f"Error parsing ML card: {e}")
            return None

    async def _fetch_page(
        self, client: httpx.AsyncClient, url: str, retries: int = 3
    ) -> Optional[str]:
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

    async def fetch_listings(self) -> list:
        listings = []
        seen_ids = set()

        cookies = _load_cookies()
        if cookies:
            logger.info("ML scraper: using session cookies (%d cookies loaded)", len(cookies))
        else:
            logger.warning("ML scraper: no session cookies found, requests may be blocked")

        async with httpx.AsyncClient(
            headers=ML_HEADERS, cookies=cookies, follow_redirects=True
        ) as client:
            for brand in self.brands:
                brand_count = 0
                for page in range(PAGES_PER_BRAND):
                    offset = page * 48
                    url = self._page_url(brand, offset)
                    logger.info(f"ML scraping {brand} page {page + 1}: {url}")

                    html = await self._fetch_page(client, url)
                    if not html:
                        logger.warning(f"ML: no HTML for {brand} page {page + 1}")
                        break

                    soup = BeautifulSoup(html, "lxml")
                    cards = soup.select("li.ui-search-layout__item")
                    if not cards:
                        logger.info(f"ML: no cards for {brand} page {page + 1}, stopping")
                        break

                    for card in cards:
                        listing = self._parse_card(card, brand)
                        if listing and listing["id"] not in seen_ids:
                            seen_ids.add(listing["id"])
                            listings.append(listing)
                            brand_count += 1

                    await asyncio.sleep(2.5)  # polite delay between pages

                logger.info(f"ML {brand}: {brand_count} listings")

                # Regional pass (inside brand loop) — city-specific pages for each brand
                for city_slug in REGIONAL_CITY_SLUGS:
                    city_count = 0
                    for rpage in range(REGIONAL_PAGES):
                        url = self._regional_url(brand, city_slug, rpage)
                        logger.info(f"ML regional {brand}/{city_slug} p{rpage+1}: {url}")
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
                                city_count += 1
                        await asyncio.sleep(2.0)
                    if city_count:
                        logger.info(f"ML regional {brand}/{city_slug}: +{city_count} new listings")

        # --- Full-category sweep (all brands, private sellers only) ---
        # Catches any brand not in self.brands (Citroën, Nissan, Jeep, Kia, etc.)
        # Uses "unknown" as brand — _parse_card extracts brand from title.
        CATEGORY_PAGES = 12
        cat_count = 0
        for page in range(CATEGORY_PAGES):
            offset = page * 48
            url = self._page_url("", offset)   # empty brand → category root
            # Override: category URL has no brand slug
            if offset == 0:
                url = f"{self.BASE_URL}/_VendedorTipo_U_"
            else:
                url = f"{self.BASE_URL}/_VendedorTipo_U__Desde_{offset + 1}"
            logger.info(f"ML category sweep page {page + 1}: {url}")
            html = await self._fetch_page(client, url)
            if not html:
                break
            soup = BeautifulSoup(html, "lxml")
            cards = soup.select("li.ui-search-layout__item")
            if not cards:
                break
            for card in cards:
                # Extract brand from title (first word before space)
                title_el = card.select_one(".poly-component__title, .ui-search-item__title, h2")
                brand_guess = title_el.get_text(strip=True).split()[0] if title_el else "unknown"
                listing = self._parse_card(card, brand_guess)
                if listing and listing["id"] not in seen_ids:
                    seen_ids.add(listing["id"])
                    listings.append(listing)
                    cat_count += 1
            await asyncio.sleep(2.5)
        if cat_count:
            logger.info(f"ML category sweep: +{cat_count} additional listings")

        logger.info(f"MercadoLibre total: {len(listings)} listings")
        return listings
