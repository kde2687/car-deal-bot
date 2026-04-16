import asyncio
import base64
import json
import logging
import re
from typing import Optional

import httpx
import config

_KAVAK_IMG_BASE = "https://images.prd.kavak.io"


def _kavak_img_url(image_path: str) -> str:
    """Convert a Kavak image path to a full CDN URL."""
    if not image_path:
        return ""
    params = {
        "bucket": "kavak-images",
        "key": image_path,
        "edits": {"resize": {"width": 400, "height": 280, "fit": "cover"}},
    }
    encoded = base64.b64encode(json.dumps(params, separators=(",", ":")).encode()).decode()
    return f"{_KAVAK_IMG_BASE}/{encoded}"

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# RSC parsing patterns
# ---------------------------------------------------------------------------
# Kavak renders via Next.js React Server Components (RSC). The HTML embeds
# inline script tags of the form:
#   self.__next_f.push([1,"<escaped JSON payload>"])
#
# The payload is a single-escaped JSON string, so actual quotes inside are \\".
# After one round of unescape (replace('\\"', '"')), we get something like:
#
#   "title":"Toyota • Corolla",
#   "subtitle":"2019 • 45.000 km • 1.8 XEI CVT • Automático",
#   "mainPrice":"32.710.000",
#   "footerInfo":"Buenos Aires",
#   "car_id":"496897",
#   "car_year":"2019"
#
# The pattern below is intentionally tolerant: [\s\S]*? (dot-all equivalent)
# instead of [^}]*? so that nested objects between the matched fields don't
# break the match. We use non-greedy [\s\S]*? anchored by the known field
# names, which is safer than relying on "no } appears in between".
#
# IMPORTANT: if Kavak renames any of these keys the pattern silently stops
# matching. The page_parsed==0 guard in fetch_listings() will fire a WARNING
# on the first page so operators know immediately.
_CAR_PATTERN = re.compile(
    r'"title":"([^"]+)"'
    r'[\s\S]*?"subtitle":"([^"]+)"'
    r'[\s\S]*?"mainPrice":"([^"]+)"'
    r'[\s\S]*?"footerInfo":"([^"]+)"'
    r'[\s\S]*?"car_id":"([^"]+)"'
    r'(?:[\s\S]*?"car_year":"([^"]*)")?',
    re.DOTALL,
)

# Image pattern: matches entries like "id":"496897"..."image":"path/to/img.jpg"
# Kavak's RSC sometimes includes the image right next to the car_id block.
# We key this map by car_id string (same value used in _CAR_PATTERN group 5).
_IMG_PATTERN = re.compile(
    r'"car_id"\s*:\s*"(\d+)"[\s\S]*?"image"\s*:\s*"([^"]*)"',
    re.DOTALL,
)

# Transmission values as they appear in Kavak subtitles (lowercased for comparison)
_TRANSMISSION_VALUES = {"manual", "automático", "automatico", "cvt", "automática", "automatica"}

# Fuel keywords found in subtitle engine strings
_FUEL_KEYWORDS = {
    "nafta": "Nafta",
    "diesel": "Diesel",
    "diésel": "Diesel",
    "híbrido": "Híbrido",
    "hibrido": "Híbrido",
    "eléctrico": "Eléctrico",
    "electrico": "Eléctrico",
    "gnc": "GNC",
}


class KavakScraper:
    BASE_URL = "https://www.kavak.com/ar/usados"
    # Kavak is always a dealer/concesionaria — every listing is agency
    IS_AGENCY = True

    def __init__(self, brands: list, min_year: int, max_km: int):
        self.brands = [b.lower() for b in brands]
        self.min_year = min_year
        self.max_km = max_km

    def _parse_rsc_chunk(self, chunk: str) -> list:
        """Extract car dicts from a Kavak RSC script chunk."""
        # Chunk uses \\" for actual quote chars — unescape once
        unescaped = chunk.replace('\\"', '"')
        # Build car_id→image_path lookup
        img_map = {m.group(1): m.group(2) for m in _IMG_PATTERN.finditer(unescaped)}
        cars = []
        for m in _CAR_PATTERN.finditer(unescaped):
            title_raw   = m.group(1)   # "Toyota • Corolla"
            subtitle    = m.group(2)   # "2019 • 45.000 km • 1.8 XEI CVT • Automático"
            price_raw   = m.group(3)   # "32.710.000"
            city        = m.group(4)   # "Buenos Aires"
            car_id      = m.group(5)   # "496897"

            # Note: labelTop check removed — Kavak now shows "Precio desde" on all
            # listings (not just financed ones). Price sanity (MIN_PRICE_ARS) handles
            # any installment-style low prices instead.

            # Brand / model from title "Toyota • Corolla"
            title_parts = [p.strip() for p in title_raw.split("•")]
            brand = title_parts[0] if title_parts else ""
            model = title_parts[1] if len(title_parts) > 1 else ""

            # Year / km / transmission / fuel from subtitle
            # e.g. "2019 • 45.000 km • 1.8 XEI CVT • Automático"
            sub_parts = [p.strip() for p in subtitle.split("•")]
            year = None
            km = None
            transmission = ""
            fuel = ""

            # Priority 1: car_year field from RSC (most reliable)
            year_from_rsc = m.group(6) if m.lastindex and m.lastindex >= 6 else None
            if year_from_rsc:
                try:
                    y = int(year_from_rsc)
                    if 1990 <= y <= 2030:
                        year = y
                except (ValueError, TypeError):
                    pass

            # Priority 2: fall back to subtitle parsing
            for part in sub_parts:
                part_lower = part.lower()
                if not year and re.match(r"^\d{4}$", part):
                    try:
                        y = int(part)
                        if 1990 <= y <= 2030:
                            year = y
                    except ValueError:
                        pass
                elif "km" in part_lower:
                    digits = re.sub(r"[^\d]", "", part)
                    km = int(digits) if digits else None
                elif part_lower in _TRANSMISSION_VALUES:
                    transmission = part
                else:
                    # Try to detect fuel type from engine string like "1.8 XEI Nafta"
                    for keyword, label in _FUEL_KEYWORDS.items():
                        if keyword in part_lower:
                            fuel = label
                            break

            # Price — Kavak Argentina publishes in ARS
            digits = re.sub(r"[^\d]", "", price_raw)
            try:
                price_ars = float(digits) if digits else None
            except ValueError:
                price_ars = None

            cars.append({
                "id": f"kavak:{car_id}",
                "car_id": car_id,
                "brand": brand,
                "model": model,
                "year": year,
                "km": km,
                "price_ars": price_ars,
                "city": city,
                "transmission": transmission,
                "fuel": fuel,
                "title": title_raw.replace("•", "").strip(),
                "subtitle": subtitle,
                "image_path": img_map.get(car_id, ""),
            })
        return cars

    def _normalize_car(self, car: dict) -> Optional[dict]:
        try:
            car_id = car.get("car_id", "")
            brand  = car.get("brand", "").strip()
            model  = car.get("model", "").strip()
            year   = car.get("year")
            km     = car.get("km")
            price_ars = car.get("price_ars")
            city   = car.get("city", "")

            if not car_id:
                return None

            if price_ars is not None:
                if price_ars < config.MIN_PRICE_ARS or price_ars > config.MAX_PRICE_ARS:
                    return None
                if year and price_ars < config.min_price_for_year(year):
                    return None

            # Compute price_usd from ARS at current MEP rate
            price_usd = None
            if price_ars is not None:
                try:
                    rate = config.get_usd_mep_rate()
                    if rate and rate > 0:
                        price_usd = round(price_ars / rate, 2)
                except Exception:
                    pass

            title = f"{brand} {model} {year or ''}".strip()
            url = f"https://www.kavak.com/ar/usados/{car_id}"

            return {
                "id": f"kavak:{car_id}",
                "source": "kavak",
                "title": title,
                "brand": brand,
                "model": model,
                "year": year,
                "km": km,
                "price_ars": price_ars,
                # price_usd populated from ARS→MEP conversion (Kavak lists in ARS)
                "price_usd": price_usd,
                "fuel": car.get("fuel", ""),
                "transmission": car.get("transmission", ""),
                "condition": "used",
                "url": url,
                "thumbnail": _kavak_img_url(car.get("image_path", "")),
                "seller_city": city,
                # Kavak is always a dealer/concesionaria
                "is_agency": True,
                "raw_data": car,
            }
        except Exception as e:
            logger.debug(f"Kavak normalize error: {e}")
            return None

    def _passes_filters(self, listing: dict) -> bool:
        # No brand filter — Kavak has all brands and we want full coverage for comparables
        year = listing.get("year")
        if year and year < self.min_year:
            return False
        km = listing.get("km")
        if km is not None and km > self.max_km:
            return False
        if km is not None and km < config.MIN_KM:
            return False
        return True

    _HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "es-AR,es;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
    }

    # Maximum consecutive page failures before giving up
    _MAX_CONSECUTIVE_ERRORS = 3

    async def fetch_listings(self) -> list:
        # Kavak is always a dealer. When ONLY_PRIVATE_SELLERS is set, skip entirely
        # rather than ingesting listings that will all be flagged is_agency=True anyway.
        if config.ONLY_PRIVATE_SELLERS:
            logger.info("Kavak scraper skipped: ONLY_PRIVATE_SELLERS=True (Kavak is always a dealer)")
            return []

        listings = []
        seen_ids = set()
        consecutive_errors = 0

        async with httpx.AsyncClient(
            headers=self._HEADERS, follow_redirects=True, timeout=30.0
        ) as client:
            for page_num in range(1, 20):
                url = f"{self.BASE_URL}?page={page_num}"
                logger.info(f"Kavak page {page_num}: {url}")
                try:
                    resp = await client.get(url)
                    if resp.status_code == 404:
                        logger.info(f"Kavak: 404 on page {page_num} — end of pagination")
                        break
                    resp.raise_for_status()
                    html = resp.text
                    consecutive_errors = 0  # reset on success
                except httpx.HTTPStatusError as e:
                    consecutive_errors += 1
                    logger.warning(
                        f"Kavak page {page_num} HTTP error {e.response.status_code}: {e} "
                        f"(consecutive errors: {consecutive_errors}/{self._MAX_CONSECUTIVE_ERRORS})"
                    )
                    if consecutive_errors >= self._MAX_CONSECUTIVE_ERRORS:
                        logger.error(f"Kavak: {self._MAX_CONSECUTIVE_ERRORS} consecutive HTTP errors — stopping scrape")
                        break
                    await asyncio.sleep(3.0)
                    continue
                except Exception as e:
                    consecutive_errors += 1
                    logger.warning(
                        f"Kavak page {page_num} error: {e} "
                        f"(consecutive errors: {consecutive_errors}/{self._MAX_CONSECUTIVE_ERRORS})"
                    )
                    if consecutive_errors >= self._MAX_CONSECUTIVE_ERRORS:
                        logger.error(f"Kavak: {self._MAX_CONSECUTIVE_ERRORS} consecutive errors — stopping scrape")
                        break
                    await asyncio.sleep(3.0)
                    continue

                chunks = re.findall(r'self\.__next_f\.push\(\[1,"(.*?)"\]\)', html, re.DOTALL)
                page_new = 0
                page_parsed = 0  # total car records parsed (regardless of filters)
                for chunk in chunks:
                    if "mainPrice" not in chunk:
                        continue
                    for car in self._parse_rsc_chunk(chunk):
                        page_parsed += 1
                        normalized = self._normalize_car(car)
                        if normalized and normalized["id"] not in seen_ids and self._passes_filters(normalized):
                            seen_ids.add(normalized["id"])
                            listings.append(normalized)
                            page_new += 1

                logger.info(f"Kavak page {page_num}: {page_new} new listings ({page_parsed} parsed, total: {len(listings)})")

                if page_parsed == 0:
                    # Truly empty page — either end of pagination or RSC parse failure
                    if page_num == 1:
                        logger.warning(
                            "Kavak: no RSC car data found on first page — "
                            "site format may have changed (check _CAR_PATTERN and RSC field names)"
                        )
                    else:
                        logger.info(f"Kavak: no cars parsed on page {page_num} — assuming end of results")
                    break
                # page_new==0 but page_parsed>0 means all were filtered (seen or below min_year/max_km)
                # — continue scraping, more unique listings may appear on later pages

                await asyncio.sleep(2.0)

        logger.info(f"Kavak total: {len(listings)} listings")
        return listings
