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
"""
import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional

import httpx

from database import SessionLocal, Listing
from ml_auth import get_auth_headers

logger = logging.getLogger(__name__)

ML_API = "https://api.mercadolibre.com"
ML_CARS_CATEGORY = "MLA1743"   # Autos y Camionetas — Argentina
DEALER_THRESHOLD = 2           # sellers with >2 active car listings = dealer
DEALER_COMPLETED_THRESHOLD = 50   # >50 completed sales = almost certainly a dealer
DEALER_ACCOUNT_AGE_YEARS = 5      # account older than 5 years + high sales = dealer
CONCURRENCY = 8                # parallel API requests


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
                    f"vendedor profesional: {completed} ventas completadas, "
                    f"{account_age_years:.0f} años en ML"
                )

        return None, None

    except Exception as e:
        logger.debug(f"ML API error for {mla_id}: {e}")
        return None, None


async def enrich_ml_new_listings(listing_ids: list[str]) -> int:
    """
    Check seller data for new ML listings via the public API.
    Marks is_agency=True and is_deal=False when a dealer signal is found.
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
            seller_id, reason = await _check_seller(client, mla_id)
            if seller_id is not None and reason:
                agencies_found.append((lid, seller_id, reason))
                logger.debug(
                    f"ML enrich: {mla_id} seller {seller_id} → dealer ({reason})"
                )
            await asyncio.sleep(0.2)

    async with httpx.AsyncClient(follow_redirects=True) as client:
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
    return len(agencies_found)
