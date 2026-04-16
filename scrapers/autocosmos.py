import asyncio
import logging
import re
from typing import Optional

import httpx
from bs4 import BeautifulSoup

import config

logger = logging.getLogger(__name__)


def _normalize(model: str) -> str:
    """Lazy import of scorer._normalize_model to avoid circular imports at module load."""
    try:
        from scorer import _normalize_model
        # _normalize_model() returns lowercase; .title() capitalises each word.
        # We call it once here — do NOT call .title() again on the result after
        # this function returns, to avoid corrupting acronyms like HRV → Hrv.
        return _normalize_model(model).title() if model else model
    except Exception:
        return model


# Financing/partial-price keywords — module-level constant so it is not
# reconstructed on every card parse.
_FINANCING_KEYWORDS = (
    "financiado", "anticipo", "cuota", "plan de ahorro",
    "plan ahorro", "financiaci", "entrega", "seña", "señ",
)

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

PAGES_PER_BRAND = 12

# Regional province IDs for AC — scrape these after the national pass to capture
# sellers in nearby provinces.
# 301=Buenos Aires interior, 303=Córdoba, 305=Mendoza, 306=Entre Ríos,
# 310=La Pampa, 318=San Luis, 319=Santa Fe, 322=Tucumán
# Mendoza (305) and Tucumán (322) added — 3rd and 4th most populous provinces
# with substantial used-car inventory previously missed entirely.
REGIONAL_PROVINCE_IDS: list[int] = [301, 303, 305, 306, 310, 318, 319, 322]
REGIONAL_PROVINCE_PAGES = 5   # 5 × 48 = up to 240 listings per brand per province


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
        # pvem=0 = private sellers only; pvem=1 = all (private + dealers).
        # Respect the global ONLY_PRIVATE_SELLERS config flag.
        pvem = "0" if config.ONLY_PRIVATE_SELLERS else "1"
        return f"{self.BASE_URL}/auto/usado/{slug}?pidx={page}&pvem={pvem}"

    def _regional_url(self, brand: str, province_id: int, page: int) -> str:
        """Province-filtered URL — captures regional cities buried past page 8 nationally."""
        slug = self._brand_slug(brand)
        pvem = "0" if config.ONLY_PRIVATE_SELLERS else "1"
        return f"{self.BASE_URL}/auto/usado/{slug}?pidx={page}&pvem={pvem}&pr={province_id}"

    def _parse_price(self, text: str) -> tuple[Optional[float], bool]:
        """Parse a price string.  Returns (ars_value, is_usd).

        is_usd=True means the original price was in USD and has been converted
        to ARS using the current MEP rate.  is_usd=False means native ARS.

        Autocosmos uses two formats:
          ARS  →  "$12.500.000"  or  "$ 12.500.000"
          USD  →  "U$S 12.500"  or  "USD 12.500"  or  "US$ 12.500"
        """
        if not text:
            return None, False
        is_usd = bool(re.search(r"u\$s|us\$|usd", text, re.IGNORECASE))
        digits = re.sub(r"[^\d]", "", text)
        if not digits:
            return None, False
        value = float(digits)
        if is_usd:
            # Convert USD → ARS at the current MEP rate
            value = value * config.get_usd_mep_rate()
        return value, is_usd

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

            # Reject financing/plan-de-ahorro listings immediately — they show partial prices.
            # _FINANCING_KEYWORDS is defined at module level for efficiency.
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
            price_usd = None

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

                # Price node detection: ARS "$12.500.000", USD "U$S 12.500" / "US$ 12.500"
                # Also match raw 7-digit numbers (ARS without symbol in some AC card layouts).
                # USD prices may only have 4-6 digits (e.g. 12500 for U$S 12.500) so we
                # cannot rely on \d{7,} alone — we also check for the USD currency marker.
                is_price_node = (
                    "$" in txt
                    or re.search(r"u\$s|us\$|usd", txt, re.IGNORECASE)
                    or re.search(r"\d{7,}", txt)
                )
                if is_price_node:
                    price_val, is_usd = self._parse_price(txt)
                    if price_val is not None:
                        if is_usd:
                            # Already converted to ARS in _parse_price; keep USD amount too
                            usd_digits = re.sub(r"[^\d]", "", txt)
                            price_usd = float(usd_digits) if usd_digits else None
                            # Accept any converted ARS amount ≥ 1M (sanity floor)
                            if price_val >= 1_000_000:
                                price_ars = price_val
                        else:
                            # Native ARS — apply 1M sanity floor (fragments like "100.000" are noise)
                            if price_val >= 1_000_000:
                                price_ars = price_val

            # Primary model source: URL slug (most reliable for AC).
            # URL: /auto/usado/brand/MODEL/trim/UUID — segment index 3 is always the model.
            # Preferred over HTML extraction which returns trim codes
            # (e.g. "Advance Aut" instead of "Versa", "Dynamique" instead of "Duster").
            url_parts = href.strip("/").split("/")
            if len(url_parts) >= 4:
                slug_model = url_parts[3].replace("-", " ").strip()
                # Do NOT call .title() here — _normalize() will do it once internally.
                if slug_model and slug_model.lower() != brand.lower() and len(slug_model) > 1:
                    model_name = slug_model

            # Title fallback: only when URL gives nothing and model is still empty/digit-start
            if (not model_name or re.match(r"^\d", model_name)) and title_attr:
                m = re.match(r"^\w[\w-]* (.+?) usado", title_attr, re.IGNORECASE)
                if m:
                    model_name = m.group(1)

            # Normalize model to match scorer's _normalize_model() so market reference
            # lookups find the same base model name (e.g. "Corolla 2.0 XEI" → "Corolla").
            # _normalize() applies .title() internally — do not .title() the slug before this.
            model_name = _normalize(model_name)

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

            # City + province from location_text.
            # Formats seen on AC:
            #   "Ciudad | Provincia"
            #   "Barrio | Capital Federal"   (CABA)
            #   "Ciudad | Provincia$12.500.000"  (price concatenated — strip it)
            # Province is stored separately so downstream geo-lookup doesn't have
            # to guess the province from city name alone (common names exist in
            # multiple provinces: "San Martín", "Villa del Parque", etc.).
            city = ""
            province = ""
            if location_text:
                loc_parts = location_text.split("|")
                city = loc_parts[0].strip()
                if len(loc_parts) > 1:
                    province_raw = loc_parts[1].split("$")[0].strip()
                    province = province_raw
                    prov_lower = province_raw.lower()
                    if any(kw in prov_lower for kw in (
                        "capital federal", "ciudad autónoma", "ciudad autonoma", "caba"
                    )):
                        city = "Buenos Aires"
                        province = "Ciudad Autónoma de Buenos Aires"

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
                "price_usd": price_usd,
                "fuel": "",
                "transmission": "",
                "condition": "used",
                "url": url,
                "thumbnail": thumbnail,
                "seller_city": city,
                "seller_province": province,
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
                    wait = min(10 * (2 ** attempt), 60)
                    logger.warning(f"Autocosmos 429 rate-limit on {url} — waiting {wait}s (attempt {attempt+1}/3)")
                    await asyncio.sleep(wait)
                    continue
                if resp.status_code >= 500:
                    logger.warning(f"Autocosmos HTTP {resp.status_code} on {url} (attempt {attempt+1}/3)")
                    resp.raise_for_status()
                resp.raise_for_status()
                return resp.text
            except httpx.HTTPStatusError:
                # Already logged above for 5xx; re-raise to outer except for retry logic
                if attempt < 2:
                    await asyncio.sleep(3 * (attempt + 1))
            except Exception as e:
                logger.debug(f"Autocosmos fetch error {url}: {e}")
                if attempt < 2:
                    await asyncio.sleep(3 * (attempt + 1))
        return None

    async def fetch_listings(self) -> list:
        listings = []
        seen_ids = set()
        # Secondary dedup: same title + same price = same physical car published twice.
        # Autocosmos allows duplicate ads (different UUID, identical content) — we keep only
        # the first occurrence so the dashboard doesn't show the same car twice.
        # NOTE: fingerprints with price_ars=None are NOT deduplicated against each other
        # because two different cars can share the same title with no parsed price — using
        # (title, None) as a key would silently discard one of them.
        seen_fingerprints: set[tuple] = set()

        def _fingerprint(lst: dict) -> tuple:
            """(title, price_ars) key — robust to minor km rounding differences.
            Returns None when price is unknown so the caller can skip dedup."""
            title = lst.get("title", "").strip().lower()
            price = lst.get("price_ars")
            if not title or price is None:
                return None  # type: ignore[return-value]
            return (title, price)

        def _add_listing(lst: dict) -> bool:
            if lst["id"] in seen_ids:
                return False
            fp = _fingerprint(lst)
            # Only deduplicate when both title and price are known (fp is not None)
            if fp is not None and fp in seen_fingerprints:
                logger.debug(f"Autocosmos dedup (same title+price): skipping {lst['id']}")
                return False
            seen_ids.add(lst["id"])
            if fp is not None:
                seen_fingerprints.add(fp)
            listings.append(lst)
            return True

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

                    # Stop only when Autocosmos itself returns no listing anchors (genuine
                    # end-of-results), not when our parser happens to filter all of them out.
                    # Stopping on 0 *parsed* listings was causing premature exit when all
                    # cards on a page were filtered by year/km/price but more pages existed.
                    if not cards:
                        logger.info(f"Autocosmos: no listing anchors for {brand} page {page}, stopping")
                        break

                    page_added = 0
                    for card in cards:
                        listing = self._parse_card(card, brand)
                        if listing and _add_listing(listing):
                            brand_count += 1
                            page_added += 1
                    logger.debug(f"Autocosmos {brand} page {page}: {len(cards)} anchors, {page_added} added")

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
                        # Stop only on genuinely empty result pages, not on filtered-out pages
                        if not cards:
                            break
                        for card in cards:
                            listing = self._parse_card(card, brand)
                            if listing and _add_listing(listing):
                                reg_count += 1
                        await asyncio.sleep(1.5)
                    if reg_count:
                        logger.info(f"Autocosmos regional {brand} pr={pr_id}: +{reg_count} new listings")

        logger.info(f"Autocosmos total: {len(listings)} listings")
        return listings
