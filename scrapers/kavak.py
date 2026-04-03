import asyncio
import logging
import re
from typing import Optional

import config

logger = logging.getLogger(__name__)

# Pattern to extract car data from Kavak's RSC (React Server Components) payload
_CAR_PATTERN = re.compile(
    r'"title":"([^"]+)","subtitle":"([^"]+)"[^}]*?"mainPrice":"([^"]+)"'
    r'[^}]*?"footerInfo":"([^"]+)"[^}]*?"car_id":"([^"]+)"'
    r'(?:[^}]*?"car_year":"([^"]*)")?'
)


class KavakScraper:
    BASE_URL = "https://www.kavak.com/ar/usados"

    def __init__(self, brands: list, min_year: int, max_km: int):
        self.brands = [b.lower() for b in brands]
        self.min_year = min_year
        self.max_km = max_km

    def _parse_rsc_chunk(self, chunk: str) -> list:
        """Extract car dicts from a Kavak RSC script chunk."""
        # Chunk uses \\" for actual quote chars — unescape once
        unescaped = chunk.replace('\\"', '"')
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

            # Year / km / transmission from subtitle "2019 • 45.000 km • 1.8 XEI CVT • Automático"
            sub_parts = [p.strip() for p in subtitle.split("•")]
            year = None
            km = None
            transmission = ""
            for part in sub_parts:
                if re.match(r"^\d{4}$", part):
                    try:
                        y = int(part)
                        if 1990 <= y <= 2030:
                            year = y
                    except ValueError:
                        pass
                elif "km" in part.lower():
                    digits = re.sub(r"[^\d]", "", part)
                    km = int(digits) if digits else None
                elif part.lower() in ("manual", "automático", "automatico", "cvt"):
                    transmission = part

            # Price
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
                "title": title_raw.replace("•", "").strip(),
                "subtitle": subtitle,
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
                "price_usd": None,
                "fuel": "",
                "transmission": car.get("transmission", ""),
                "condition": "used",
                "url": url,
                "thumbnail": "",
                "seller_city": city,
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

    async def fetch_listings(self) -> list:
        try:
            from playwright.async_api import async_playwright
        except ImportError:
            logger.warning("playwright not installed — skipping Kavak. Run: pip3 install playwright && playwright install chromium")
            return []

        listings = []
        seen_ids = set()

        async with async_playwright() as pw:
            browser = await pw.chromium.launch(headless=True)
            context = await browser.new_context(
                locale="es-AR",
                user_agent=(
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0.0.0 Safari/537.36"
                ),
            )
            page = await context.new_page()

            for page_num in range(1, 26):
                url = f"{self.BASE_URL}?page={page_num}"
                logger.info(f"Kavak (Playwright) page {page_num}: {url}")
                try:
                    await page.goto(url, wait_until="networkidle", timeout=35000)
                except Exception as e:
                    logger.warning(f"Kavak page {page_num} load error: {e}")
                    try:
                        await page.goto(url, wait_until="domcontentloaded", timeout=20000)
                        await asyncio.sleep(3)
                    except Exception:
                        break

                html = await page.content()

                # Extract RSC chunks containing car data
                chunks = re.findall(r'self\.__next_f\.push\(\[1,"(.*?)"\]\)', html, re.DOTALL)
                page_new = 0
                for chunk in chunks:
                    if "mainPrice" not in chunk:
                        continue
                    for car in self._parse_rsc_chunk(chunk):
                        normalized = self._normalize_car(car)
                        if normalized and normalized["id"] not in seen_ids and self._passes_filters(normalized):
                            seen_ids.add(normalized["id"])
                            listings.append(normalized)
                            page_new += 1

                logger.info(f"Kavak page {page_num}: {page_new} new listings (total: {len(listings)})")

                if page_new == 0:
                    if page_num == 1:
                        logger.warning("Kavak: no cars on first page, stopping")
                    break

                await asyncio.sleep(2)

            await browser.close()

        logger.info(f"Kavak total: {len(listings)} listings")
        return listings
