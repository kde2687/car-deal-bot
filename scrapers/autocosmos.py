import asyncio
import logging
import re
from typing import Optional

import httpx
from bs4 import BeautifulSoup

import config

logger = logging.getLogger(__name__)

AC_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "es-AR,es;q=0.9",
    "Referer": "https://www.autocosmos.com.ar/",
}

PAGES_PER_BRAND = 8

# Regional province IDs for AC — scrape these after the national pass to capture
# sellers in nearby provinces that fall outside the first PAGES_PER_BRAND pages.
# pr=301 = Buenos Aires province (non-AMBA), pr=310 = La Pampa
REGIONAL_PROVINCE_IDS: list[int] = [301, 310]
REGIONAL_PROVINCE_PAGES = 4   # 4 × 48 = up to 192 listings per brand per province


class AutocosmosScraper:
    BASE_URL = "https://www.autocosmos.com.ar"

    def __init__(self, brands: list, min_year: int, max_km: int):
        self.brands = brands
        self.min_year = min_year
        self.max_km = max_km

    def _brand_slug(self, brand: str) -> str:
        replacements = {"á": "a", "é": "e", "í": "i", "ó": "o", "ú": "u", "ü": "u", "ë": "e", "ñ": "n", "ã": "a", " ": "-"}
        slug = brand.lower()
        for k, v in replacements.items():
            slug = slug.replace(k, v)
        return slug

    def _page_url(self, brand: str, page: int) -> str:
        slug = self._brand_slug(brand)
        # pvem=0 = private sellers only
        return f"{self.BASE_URL}/auto/usado/{slug}?pidx={page}&pvem=0"

    def _regional_url(self, brand: str, province_id: int, page: int) -> str:
        """Province-filtered URL — captures regional cities buried past page 8 nationally."""
        slug = self._brand_slug(brand)
        return f"{self.BASE_URL}/auto/usado/{slug}?pidx={page}&pvem=0&pr={province_id}"

    def _parse_price(self, text: str) -> Optional[float]:
        if not text:
            return None
        digits = re.sub(r"[^\d]", "", text)
        return float(digits) if digits else None

    def _parse_card(self, a_tag, brand: str) -> Optional[dict]:
        try:
            href = a_tag.get("href", "").split("?")[0]  # strip query string
            if not href or "/auto/usado/" not in href:
                return None

            # Build a unique ID from the URL slug
            # href like /auto/usado/volkswagen/gol-trend/5p.../UUID (6 slashes)
            # Model-index pages like /auto/usado/toyota/86 only have 4 slashes — reject
            if href.count("/") < 5:
                return None
            parts = href.strip("/").split("/")
            listing_id = parts[-1] if parts else ""
            # UUID is 32 hex chars; model slugs ("86", "avensis") are much shorter
            if not listing_id or len(listing_id) < 20:
                return None

            # Title from <a title="...">
            title_attr = a_tag.get("title", "")

            # Reject financing/plan-de-ahorro listings immediately — they show partial prices
            _FINANCING_KEYWORDS = (
                "financiado", "anticipo", "cuota", "plan de ahorro",
                "plan ahorro", "financiaci",
            )
            title_lower = title_attr.lower()
            if any(kw in title_lower for kw in _FINANCING_KEYWORDS):
                return None

            # Content divs: brand, model, "year | km", "city | province", price
            divs = a_tag.find_all(["div", "strong", "span"], recursive=False)
            # Also try deeper
            if len(divs) < 3:
                divs = a_tag.find_all(["div", "strong", "span"])

            text_nodes = [d.get_text(strip=True) for d in divs if d.get_text(strip=True)]

            # Brand always comes from the search URL — never trust HTML extraction
            brand_name = brand.title()
            model_name = ""
            year = None
            km = None
            location_text = ""
            price_ars = None

            for i, txt in enumerate(text_nodes):
                txt_lower = txt.lower()

                # Skip any node that mentions financing/installment pricing
                if any(kw in txt_lower for kw in _FINANCING_KEYWORDS):
                    return None

                # Year | KM pattern: "2020 | 20000 km"
                yk_match = re.match(r"(\d{4})\s*[|·]\s*([\d.,]+)\s*km", txt, re.IGNORECASE)
                if yk_match:
                    y = int(yk_match.group(1))
                    if 1990 <= y <= 2030:
                        year = y
                    km_raw = re.sub(r"[^\d]", "", yk_match.group(2))
                    km = int(km_raw) if km_raw else None
                    # Model is the text node just before year|km.
                    # If it starts with a digit it's a trim/engine spec ("1.5L S", "2.0 TDi"),
                    # not a model name — try the node before that.
                    if i >= 1:
                        candidate = text_nodes[i - 1]
                        if re.match(r"^\d", candidate) and i >= 2:
                            candidate = text_nodes[i - 2]
                        model_name = candidate
                    continue

                # Location: "City | Province" or "City | Region"
                if "|" in txt and year is not None and not location_text:
                    location_text = txt
                    continue

                if "$" in txt or re.search(r"\d{6,}", txt):
                    price_val = self._parse_price(txt)
                    if price_val and price_val > 100_000:
                        price_ars = price_val

            # Primary model source: URL slug (most reliable for AC).
            # URL: /auto/usado/brand/MODEL/trim/UUID — segment index 3 is always the model.
            # Preferred over HTML extraction which returns trim codes
            # (e.g. "Advance Aut" instead of "Versa", "Dynamique" instead of "Duster").
            url_parts = href.strip("/").split("/")
            if len(url_parts) >= 4:
                slug_model = url_parts[3].replace("-", " ").strip()
                if slug_model and slug_model.lower() != brand.lower() and len(slug_model) > 1:
                    model_name = slug_model.title()

            # Title fallback: only when URL gives nothing and model is still empty/digit-start
            if (not model_name or re.match(r"^\d", model_name)) and title_attr:
                m = re.match(r"^\w[\w-]* (.+?) usado", title_attr, re.IGNORECASE)
                if m:
                    model_name = m.group(1)

            # Apply filters
            if year and year < self.min_year:
                return None
            if km is not None and km > self.max_km:
                return None
            if km is not None and km < config.MIN_KM:
                return None
            if price_ars is not None:
                if price_ars < config.MIN_PRICE_ARS or price_ars > config.MAX_PRICE_ARS:
                    return None
                if year and price_ars < config.min_price_for_year(year):
                    return None

            # Thumbnail from <img>
            img = a_tag.find("img")
            thumbnail = ""
            if img:
                thumbnail = img.get("src") or img.get("data-src") or ""

            # City from location
            # location_text: "Barrio |Provincia$precio" or "Ciudad |Provincia$precio"
            # If the province part indicates CABA, the left part is a neighborhood, not a city —
            # override to "Buenos Aires" so geo lookup resolves to CABA coordinates.
            if location_text:
                parts = location_text.split("|")
                city = parts[0].strip()
                if len(parts) > 1:
                    province_raw = parts[1].split("$")[0].strip().lower()
                    if any(kw in province_raw for kw in (
                        "capital federal", "ciudad autónoma", "ciudad autonoma", "caba"
                    )):
                        city = "Buenos Aires"
            else:
                city = ""

            url = f"{self.BASE_URL}{href}"

            return {
                "id": f"autocosmos:{listing_id}",
                "source": "autocosmos",
                "title": title_attr or f"{brand_name} {model_name} {year or ''}".strip(),
                "brand": brand_name or brand.title(),
                "model": model_name,
                "year": year,
                "km": km,
                "price_ars": price_ars,
                "price_usd": None,
                "fuel": "",
                "transmission": "",
                "condition": "used",
                "url": url,
                "thumbnail": thumbnail,
                "seller_city": city,
                "raw_data": {
                    "title": title_attr,
                    "price_ars": price_ars,
                    "location": location_text,
                    "url": url,
                },
            }
        except Exception as e:
            logger.debug(f"Autocosmos card parse error: {e}")
            return None

    async def _fetch_page(self, client: httpx.AsyncClient, url: str) -> Optional[str]:
        for attempt in range(3):
            try:
                resp = await client.get(url, timeout=25.0)
                if resp.status_code == 404:
                    return None
                if resp.status_code == 429:
                    await asyncio.sleep(min(10 * (2 ** attempt), 60))
                    continue
                resp.raise_for_status()
                return resp.text
            except Exception as e:
                logger.debug(f"Autocosmos fetch error {url}: {e}")
                if attempt < 2:
                    await asyncio.sleep(3 * (attempt + 1))
        return None

    async def fetch_listings(self) -> list:
        listings = []
        seen_ids = set()

        async with httpx.AsyncClient(headers=AC_HEADERS, follow_redirects=True) as client:
            for brand in self.brands:
                brand_count = 0
                for page in range(1, PAGES_PER_BRAND + 1):
                    url = self._page_url(brand, page)
                    logger.info(f"Autocosmos scraping {brand} page {page}: {url}")

                    html = await self._fetch_page(client, url)
                    if not html:
                        logger.info(f"Autocosmos: no HTML for {brand} page {page}, stopping")
                        break

                    soup = BeautifulSoup(html, "lxml")
                    # Cards are <a> tags pointing to /auto/usado/brand/model/...
                    cards = soup.select('a[href*="/auto/usado/"][href*="/"]')
                    # Filter out model-index nav links (≤4 slashes); real listings have ≥5
                    cards = [c for c in cards if c.get("href", "").split("?")[0].count("/") >= 5]

                    if not cards:
                        logger.info(f"Autocosmos: no cards for {brand} page {page}, stopping")
                        break

                    for card in cards:
                        listing = self._parse_card(card, brand)
                        if listing and listing["id"] not in seen_ids:
                            seen_ids.add(listing["id"])
                            listings.append(listing)
                            brand_count += 1

                    await asyncio.sleep(2.0)

                logger.info(f"Autocosmos {brand}: {brand_count} listings")

                # Regional pass — province-filtered to surface listings from
                # Buenos Aires interior + La Pampa buried past page 8 nationally
                for pr_id in REGIONAL_PROVINCE_IDS:
                    reg_count = 0
                    for page in range(1, REGIONAL_PROVINCE_PAGES + 1):
                        url = self._regional_url(brand, pr_id, page)
                        logger.info(f"Autocosmos regional {brand} pr={pr_id} p{page}: {url}")
                        html = await self._fetch_page(client, url)
                        if not html:
                            break
                        soup = BeautifulSoup(html, "lxml")
                        cards = soup.select('a[href*="/auto/usado/"][href*="/"]')
                        cards = [c for c in cards if c.get("href", "").split("?")[0].count("/") >= 5]
                        if not cards:
                            break
                        for card in cards:
                            listing = self._parse_card(card, brand)
                            if listing and listing["id"] not in seen_ids:
                                seen_ids.add(listing["id"])
                                listings.append(listing)
                                reg_count += 1
                        await asyncio.sleep(1.5)
                    if reg_count:
                        logger.info(f"Autocosmos regional {brand} pr={pr_id}: +{reg_count} new listings")

        logger.info(f"Autocosmos total: {len(listings)} listings")
        return listings
