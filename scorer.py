import logging
import math
import re
import statistics
from datetime import datetime, timedelta
from typing import Optional

import config
from database import SessionLocal, Listing, MarketReference, SegmentVelocity
from geo import coords_from_listing_dict, distance_from_darregueira, ORIGIN_NAME

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def format_ars(price: Optional[float]) -> str:
    if price is None:
        return "$0 ARS"
    if price >= 1_000_000:
        return f"${price / 1_000_000:.1f}M ARS"
    if price >= 1_000:
        return f"${price / 1_000:.0f}K ARS"
    return f"${price:.0f} ARS"


# Words that are version/trim/engine info — strip these for model normalisation
_STRIP_WORDS = re.compile(
    r"\b(\d[\w./]*"           # any token starting with a digit (1.8, 2.0T, 150cv…)
    r"|cvt|mt|at|dct|aut|man"
    r"|turbo|tdi|tdci|tfsi|fsi|gdi|mpi|hev|phev|bev|ev"
    r"|nafta|diesel|gnc|híbrido|hibrido|híbrida|hibrida|electrico|eléctrico"
    r"|ecvt|tsi|mpi|sdi"
    r"|xei|xls|xlt|xl|ls|lt|ltz|lte|lx|ex|se|le|sr|srx"
    r"|trendline|comfortline|highline|sportline|luxury|active|elegance|allure|intens|zen|life|play|feel|gt|gts|rs|r"
    r"|pack|hatch|sedan|suv|pickup|pick-up|cabina|cd|cs|ds"
    r"|4x4|4x2|awd|fwd|rwd|rr|4wd"
    r"|5p|3p|4p|2p|p"
    r"|plus|pro|max|prime|full|base|entry|sport|limited|premium|prestige|exclusive|signature"
    r"|compact|gti|gli|seg|hv|sw|privilege|advance|sense|iconic|dynamique|feel|zen|stepway"
    r"|nuevo|usado|certificado)\b",
    re.IGNORECASE,
)


def _normalize_model(model: str) -> str:
    """
    Extract the base model name, stripping trim/engine/version suffixes.
    'Corolla 1.8 SEG CVT' → 'corolla'
    'Corolla Cross 2.0 XEI CVT' → 'corolla cross'
    'Hilux Pick-up 2.4 CD SR 4x4' → 'hilux'
    'Gol Trend 5P Trendline' → 'gol trend'
    'Etios XLS 1.5' → 'etios'
    '208 1.6 Allure' → '208'  (Peugeot-style numeric model names)
    """
    if not model:
        return ""
    cleaned = _STRIP_WORDS.sub(" ", model)
    # Collapse whitespace and take first 1-3 meaningful words.
    # Keep single uppercase letters (model prefix codes like "T" in VW T Cross).
    words = [w for w in cleaned.split() if len(w) >= 2 or (len(w) == 1 and w.isupper())]
    if words:
        return " ".join(words[:3]).lower().strip().replace("-", " ")
    # Fallback: stripping removed everything (e.g. Peugeot "208 1.6 Allure").
    # Return the first token of the original as-is — it IS the model name.
    first = model.split()[0]
    return first.lower().replace("-", " ")


# ---------------------------------------------------------------------------
# Market reference
# ---------------------------------------------------------------------------

MIN_SAMPLE_SIZE = 2   # minimum comparables needed for a market reference

# Confidence index saturation point — n=30 gives full size_score=1.0
# Based on SE_median ∝ 1/√n (asymptotic theory of quantile estimators)
_CI_N_SAT = 30.0

# Reference type weight: proxy for omitted-variable bias
# exact=no bias, brand_fallback=high bias (ignores model, km, year precision)
_CI_TYPE_WEIGHT = {
    "exact":          1.00,
    "exact_nokm":     0.85,
    "broad":          0.60,
    "curve":          0.35,
    "brand_fallback": 0.20,
    "ml_model":       0.45,
    "cold":           0.00,
}


def _confidence_index(ref_type: str, n: int, prices: list[float]) -> int:
    """
    0–100 confidence index for the benchmark price estimate.

    CI = type_weight × sqrt(n / 30) × exp(-3 × QCD)

    Components:
      type_weight : proxy for specification bias (exact ref = low bias)
      sqrt(n/30)  : precision factor — from SE_median ∝ 1/√n (CLT for quantile estimators)
      QCD         : Quartile Coefficient of Dispersion = (Q3-Q1)/(Q3+Q1)
                    Robust, scale-invariant measure of market homogeneity.
                    Preferred over CoV because median+IQR are robust estimators.
    """
    type_w = _CI_TYPE_WEIGHT.get(ref_type, 0.0)
    if type_w == 0.0 or n == 0:
        return 0

    # Precision: saturates at n_sat via square-root (CLT)
    size_score = min(1.0, math.sqrt(n / _CI_N_SAT))

    # Market homogeneity via QCD
    if prices and len(prices) >= 4:
        sp = sorted(prices)
        n = len(sp)
        q1 = sp[max(0, (n - 1) // 4)]
        q3 = sp[min(n - 1, 3 * (n - 1) // 4)]
        qcd = (q3 - q1) / (q3 + q1) if (q3 + q1) > 0 else 0.5
    elif prices and len(prices) >= 2:
        mean = sum(prices) / len(prices)
        std = (sum((p - mean) ** 2 for p in prices) / len(prices)) ** 0.5
        qcd = std / mean if mean > 0 else 0.5  # CoV fallback for n<4
    else:
        qcd = 0.5  # unknown dispersion — moderate penalty

    homogeneity_score = math.exp(-3.0 * qcd)

    return round(type_w * size_score * homogeneity_score * 100)


def _ars_to_usd(price_ars: float, usd_rate: float) -> float:
    return price_ars / usd_rate if usd_rate else price_ars


def _current_year() -> int:
    return datetime.utcnow().year


def _weighted_median(values: list[float], weights: list[float]) -> float:
    """Weighted median with linear interpolation at the 50% boundary."""
    if not values or not weights:
        return 0.0
    pairs = sorted(zip(values, weights), key=lambda x: x[0])
    total = sum(weights)
    cumulative = 0.0
    prev_val = pairs[0][0]
    for val, w in pairs:
        cumulative += w
        if cumulative >= total / 2:
            if cumulative > total / 2 and w > 0:
                # Interpolate back from val towards prev_val
                over = cumulative - total / 2
                return val - (over / w) * (val - prev_val)
            return (prev_val + val) / 2.0
        prev_val = val
    return pairs[-1][0]


_DECAY_LAMBDA: float = 0.0  # precomputed at first use


def _decay_weight(first_seen: datetime, half_life_days: int) -> float:
    """Exponential decay: weight = e^(-λ * days_old), λ = ln(2) / half_life."""
    global _DECAY_LAMBDA
    if not _DECAY_LAMBDA:
        _DECAY_LAMBDA = math.log(2) / half_life_days
    days_old = max(0.0, (datetime.utcnow() - first_seen).total_seconds() / 86400)
    return math.exp(-_DECAY_LAMBDA * days_old)


def _fit_depreciation_curve(
    ages: list[float], prices_usd: list[float], weights: list[float]
) -> Optional[tuple[float, float]]:
    """
    Fit price_usd = P0 * exp(-alpha * antigüedad_años) via weighted least squares
    on log-transformed prices. Returns (P0, alpha) or None if fit fails.
    """
    try:
        import numpy as np
        valid = [(a, p, w) for a, p, w in zip(ages, prices_usd, weights)
                 if p > 0 and a >= 0 and w > 0]
        if len(valid) < 3:
            return None
        ages_arr   = np.array([v[0] for v in valid], dtype=float)
        log_prices = np.log([v[1] for v in valid])
        w_arr      = np.array([v[2] for v in valid], dtype=float)
        # Weighted polyfit: log(p) = log(P0) - alpha * age
        coeffs = np.polyfit(ages_arr, log_prices, 1, w=w_arr)
        alpha = float(-coeffs[0])
        P0    = float(math.exp(coeffs[1]))
        if alpha < 0 or P0 <= 0:
            return None
        return P0, alpha
    except Exception:
        return None


def _percentile_rank(price_usd: float, comparables: list[float]) -> int:
    """What % of comparables cost LESS than this listing (lower = cheaper = better deal)."""
    if not comparables:
        return 50
    below = sum(1 for p in comparables if p < price_usd)
    return round(below / len(comparables) * 100)


def calculate_market_reference(
    session, brand: str, model: str, year: int,
    listing_km: Optional[int] = None, exclude_id: Optional[str] = None,
    listing_price_usd: Optional[float] = None,
) -> tuple[Optional[float], Optional[float], int, str, Optional[int], int]:
    """
    Return (median_usd, median_ars_current, sample_count, ref_type, percentile_rank, confidence_index).
    percentile_rank: 0–100, lower = cheaper relative to comparables. None when cold.
    confidence_index: 0–100, quality of the benchmark estimate.
    Uses 12-month history in USD with exponential decay weighting (half-life configurable).
    ref_type: 'exact' | 'exact_nokm' | 'broad' | 'curve' | 'brand_fallback' | 'ml_model' | 'cold'
    """
    from config import get_usd_mep_rate
    usd_rate     = get_usd_mep_rate()
    half_life    = config.MARKET_HALF_LIFE_DAYS
    km_tolerance = config.MARKET_KM_TOLERANCE
    cutoff       = datetime.utcnow() - timedelta(days=config.MARKET_HISTORY_DAYS)

    base_model = _normalize_model(model)

    def _fetch(year_filter, km_filter: bool) -> tuple[list[float], list[float]]:
        """Return (usd_prices, decay_weights) for matching listings."""
        q = (
            session.query(
                Listing.id, Listing.price_ars, Listing.price_usd_equiv,
                Listing.km, Listing.first_seen, Listing.model  # model included — no N+1
            )
            .filter(
                Listing.brand.ilike(f"%{brand}%"),
                (Listing.price_ars > 0) | (Listing.price_usd_equiv > 0),
                Listing.first_seen >= cutoff,
                # Include agency listings: dealers set market prices and are real data points.
                # is_agency listings are excluded from deal detection separately.
                Listing.hidden != True,
                # Include active + sold listings; sold = confirmed clearing price
                Listing.status.in_(["active", "sold"]),
            )
        )
        if year_filter:
            q = q.filter(Listing.year.in_(year_filter))
        rows = q.all()

        prices, weights = [], []
        for lid, price_ars, price_usd_equiv, km, first_seen, row_model in rows:
            if exclude_id and lid == exclude_id:
                continue

            if km_filter and listing_km and km:
                if abs(km - listing_km) > km_tolerance:
                    continue

            nm = _normalize_model(row_model or "")
            if base_model not in nm and nm not in base_model:
                continue

            usd = price_usd_equiv if price_usd_equiv and price_usd_equiv > 0 \
                  else _ars_to_usd(price_ars, usd_rate)
            w = _decay_weight(first_seen, half_life) if first_seen else 1.0
            prices.append(usd)
            weights.append(w)

        return prices, weights

    if not base_model:
        return None, None, 0, "cold", None, 0

    year_exact  = [year - 1, year, year + 1]
    year_broad  = list(range(year - 3, year + 4))

    def _pct(prices):
        return _percentile_rank(listing_price_usd, prices) if listing_price_usd and prices else None

    # Pass 1: exact year ±1, km filtered
    prices, weights = _fetch(year_exact, km_filter=True)
    if len(prices) >= MIN_SAMPLE_SIZE:
        median_usd = _weighted_median(prices, weights)
        median_ars = median_usd * usd_rate
        _save_market_ref(session, brand, model, year, median_ars,
                         median_usd, usd_rate, len(prices))
        return median_usd, median_ars, len(prices), "exact", _pct(prices), _confidence_index("exact", len(prices), prices)

    # Pass 2: exact year ±1, no km filter
    prices, weights = _fetch(year_exact, km_filter=False)
    if len(prices) >= MIN_SAMPLE_SIZE:
        median_usd = _weighted_median(prices, weights)
        median_ars = median_usd * usd_rate
        _save_market_ref(session, brand, model, year, median_ars,
                         median_usd, usd_rate, len(prices))
        return median_usd, median_ars, len(prices), "exact_nokm", _pct(prices), _confidence_index("exact_nokm", len(prices), prices)

    # Pass 3: broad year ±3, no km filter
    prices, weights = _fetch(year_broad, km_filter=False)
    if len(prices) >= MIN_SAMPLE_SIZE:
        median_usd = _weighted_median(prices, weights)
        median_ars = median_usd * usd_rate
        return median_usd, median_ars, len(prices), "broad", _pct(prices), _confidence_index("broad", len(prices), prices)

    # Pass 4: depreciation curve — same model, ALL years, interpolate at target antigüedad
    curve_result = _depreciation_curve_estimate(
        session, brand, base_model, year, usd_rate, half_life, cutoff, exclude_id
    )
    if curve_result is not None:
        est_usd, n_samples = curve_result
        return est_usd, est_usd * usd_rate, n_samples, "curve", None, _confidence_index("curve", n_samples, [])

    # Pass 5: brand fallback — same brand, similar antigüedad (±2 años), any model
    brand_result = _brand_age_fallback(
        session, brand, year, usd_rate, half_life, cutoff, exclude_id
    )
    if brand_result is not None:
        est_usd, n_samples, bf_prices = brand_result
        return est_usd, est_usd * usd_rate, n_samples, "brand_fallback", None, _confidence_index("brand_fallback", n_samples, bf_prices)

    # Pass 6: LightGBM hedonic model (activates automatically when ≥300 listings)
    try:
        from pricing_model.pipeline import get_pipeline
        pipeline = get_pipeline()
        if pipeline.is_ready():
            est_usd = pipeline.predict(brand, model, year, listing_km or 0)
            if est_usd and est_usd > 0:
                return est_usd, est_usd * usd_rate, 0, "ml_model", None, _confidence_index("ml_model", 0, [])
    except Exception as e:
        logger.debug(f"ML model fallback failed: {e}")

    return None, None, 0, "cold", None, 0


def _depreciation_curve_estimate(
    session, brand, base_model, target_year, usd_rate, half_life, cutoff, exclude_id
) -> Optional[tuple[float, int]]:
    """
    Fetch all years of the same model, fit depreciation curve, estimate price
    at target antigüedad = current_year - target_year.
    Returns (estimated_usd, sample_count) or None.
    """
    target_age = _current_year() - target_year
    if target_age < 0:
        return None

    q = session.query(
        Listing.id, Listing.price_ars, Listing.price_usd_equiv,
        Listing.year, Listing.first_seen, Listing.model  # model included — no N+1
    ).filter(
        Listing.brand.ilike(f"%{brand}%"),
        (Listing.price_ars > 0) | (Listing.price_usd_equiv > 0),
        Listing.year.isnot(None),
        Listing.first_seen >= cutoff,
        Listing.hidden != True,
    ).all()

    ages, prices, weights = [], [], []
    for lid, price_ars, price_usd_equiv, car_year, first_seen, row_model in q:
        if exclude_id and lid == exclude_id:
            continue
        nm = _normalize_model(row_model or "")
        if base_model not in nm and nm not in base_model:
            continue
        age = _current_year() - (car_year or 0)
        if age < 0 or age > 25:
            continue
        usd = price_usd_equiv if price_usd_equiv and price_usd_equiv > 0 \
              else _ars_to_usd(price_ars, usd_rate)
        w = _decay_weight(first_seen, half_life) if first_seen else 1.0
        ages.append(float(age))
        prices.append(usd)
        weights.append(w)

    if len(ages) < 3:
        return None

    curve = _fit_depreciation_curve(ages, prices, weights)
    if curve is None:
        return None

    P0, alpha = curve
    estimated = P0 * math.exp(-alpha * target_age)
    # Sanity check: estimate must be within 5x of data range
    if not (min(prices) * 0.2 <= estimated <= max(prices) * 5):
        return None
    return estimated, len(ages)


def _brand_age_fallback(
    session, brand, target_year, usd_rate, half_life, cutoff, exclude_id
) -> Optional[tuple[float, int, list[float]]]:
    """
    Same brand, any model, antigüedad within ±2 years of target.
    Returns (weighted_median_usd, sample_count) or None.
    """
    target_age = _current_year() - target_year
    min_year = _current_year() - (target_age + 2)
    max_year = _current_year() - (target_age - 2)

    q = session.query(
        Listing.id, Listing.price_ars, Listing.price_usd_equiv, Listing.first_seen
    ).filter(
        Listing.brand.ilike(f"%{brand}%"),
        (Listing.price_ars > 0) | (Listing.price_usd_equiv > 0),
        Listing.year.between(min_year, max_year),
        Listing.first_seen >= cutoff,
        Listing.hidden != True,
    ).all()

    prices, weights = [], []
    for lid, price_ars, price_usd_equiv, first_seen in q:
        if exclude_id and lid == exclude_id:
            continue
        usd = price_usd_equiv if price_usd_equiv and price_usd_equiv > 0 \
              else _ars_to_usd(price_ars, usd_rate)
        w = _decay_weight(first_seen, half_life) if first_seen else 1.0
        prices.append(usd)
        weights.append(w)

    if len(prices) < MIN_SAMPLE_SIZE:
        return None
    return _weighted_median(prices, weights), len(prices), prices


def _save_market_ref(session, brand, model, year, median_ars, median_usd, usd_rate, count):
    # Normalize model to prevent fragmentation (e.g. "Hilux 2.4 CD SR" and "Hilux 4x4" → "hilux")
    norm_model = _normalize_model(model).title() or model
    try:
        ref = session.query(MarketReference).filter_by(brand=brand, model=norm_model, year=year).first()
        if ref:
            ref.median_price_ars = median_ars
            ref.avg_price_ars = median_ars  # avg_price_ars kept for schema compat
            ref.median_usd = median_usd
            ref.usd_rate_used = usd_rate
            ref.sample_count = count
            ref.updated_at = datetime.utcnow()
        else:
            session.add(MarketReference(
                brand=brand, model=norm_model, year=year,
                avg_price_ars=median_ars, median_price_ars=median_ars,
                median_usd=median_usd, usd_rate_used=usd_rate,
                sample_count=count, updated_at=datetime.utcnow(),
            ))
        session.commit()
    except Exception as e:
        logger.warning(f"_save_market_ref failed for {brand} {norm_model} {year}: {e}")
        session.rollback()


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_listing(session, listing_dict: dict) -> dict:
    from config import get_usd_mep_rate
    usd_rate = get_usd_mep_rate()

    price_ars = listing_dict.get("price_ars")
    price_usd = listing_dict.get("price_usd")

    # Resolve price in ARS and USD equiv
    if price_usd and price_usd > 0:
        # Listing already has a USD price (ML USD listings) — use directly
        price_usd_equiv = price_usd
        if not price_ars or price_ars <= 0:
            price_ars = price_usd * usd_rate
    elif price_ars and price_ars > 0:
        price_usd_equiv = _ars_to_usd(price_ars, usd_rate)
    else:
        return {"score": 0.0, "discount_pct": 0.0, "is_deal": False,
                "deal_reason": "Sin precio disponible", "price_usd_equiv": None}

    brand      = listing_dict.get("brand") or ""
    model      = listing_dict.get("model") or ""
    year       = listing_dict.get("year") or 0
    km         = listing_dict.get("km") or 0
    listing_id = listing_dict.get("id")

    market_usd, market_ars, sample_count, ref_type, percentile_rank, confidence_index = calculate_market_reference(
        session, brand, model, year, listing_km=km, exclude_id=listing_id,
        listing_price_usd=price_usd_equiv,
    )

    if market_usd is None:
        return {
            "score": 0.0,
            "discount_pct": 0.0,
            "is_deal": False,
            "deal_reason": f"Sin referencia de mercado para {brand} {_normalize_model(model)} {year} (pocos datos)",
            "price_usd_equiv": price_usd_equiv,
            "percentile_rank": None,
            "ref_type": "cold",
            "confidence_index": 0,
        }

    # All comparison in USD — inflation-neutral
    discount_pct = (market_usd - price_usd_equiv) / market_usd * 100
    base_score = max(0.0, min(100.0, discount_pct))

    if km > 150_000:
        km_mod = -15
    elif km > 100_000:
        km_mod = -5
    else:
        km_mod = 0

    if year and year < 2012:
        age_mod = -5
    else:
        age_mod = 0

    # Motivated seller signal: listing age + price drops lower the required discount threshold.
    # A seller who has dropped price 2+ times after 45 days is clearly motivated.
    days_on_market   = listing_dict.get("days_on_market", 0) or 0
    price_changes_ct = listing_dict.get("price_changes_count", 0) or 0
    if days_on_market > 45 and price_changes_ct >= 2:
        motivated_discount_reduction = 2.0
        motivated_note = f" Vendedor motivado ({days_on_market}d en mercado, {price_changes_ct} bajas de precio)."
    elif days_on_market > 21 and price_changes_ct >= 1:
        motivated_discount_reduction = 1.0
        motivated_note = f" Baja de precio reciente ({days_on_market}d en mercado)."
    else:
        motivated_discount_reduction = 0.0
        motivated_note = ""

    ref_labels = {
        "exact":         f"{sample_count} muestras, km similar",
        "exact_nokm":    f"{sample_count} muestras",
        "broad":         f"ref. amplia ±3 años, {sample_count} muestras",
        "curve":         f"curva depreciación, {sample_count} muestras",
        "brand_fallback":f"ref. marca similar, {sample_count} muestras",
        "ml_model":      "modelo ML hedónico",
    }
    ref_penalties = {
        "exact": 0, "exact_nokm": -2, "broad": -5,
        "curve": -8, "brand_fallback": -15, "ml_model": -5,
    }
    ref_note    = f" ({ref_labels.get(ref_type, str(sample_count))})"
    ref_penalty = ref_penalties.get(ref_type, 0)

    # Confidence multiplier: full confidence at 5+ samples (was 10 — too aggressive
    # for a young dataset where most models have 3-5 comparables)
    confidence  = min(1.0, math.sqrt(sample_count / 5.0)) if sample_count > 0 else 1.0

    raw_score   = max(0.0, min(100.0, base_score + km_mod + age_mod + ref_penalty))
    final_score = max(0.0, min(100.0, raw_score * confidence))

    # is_deal: based purely on discount_pct relative to reference quality.
    # score/confidence only affects ranking — not deal detection.
    # Tighter reference → lower discount required (we trust the benchmark more).
    # Motivated seller signal lowers threshold by up to 2 pp.
    _deal_min_discount = {
        "exact":          4.0,    # same model, ±1yr, similar km — high trust
        "exact_nokm":     6.0,    # same model, ±1yr
        "broad":          8.0,    # same model, ±3yr
        "curve":          10.0,   # depreciation curve interpolation
        "brand_fallback": 10.0,   # same brand, any model — low trust
        "ml_model":       8.0,
    }
    effective_min_discount = _deal_min_discount.get(ref_type, 100.0) - motivated_discount_reduction
    is_deal = (
        discount_pct >= effective_min_discount
        and ref_type != "cold"
    )

    market_label = f"USD {market_usd:,.0f} ({format_ars(market_ars)})"
    asking_label = f"USD {price_usd_equiv:,.0f} ({format_ars(price_ars)})"
    discount_str = f"{discount_pct:.0f}% por debajo" if discount_pct >= 0 else f"{abs(discount_pct):.0f}% por encima"
    km_str       = f"{km:,} km" if km else "N/A"

    deal_reason = (
        f"{year} {brand} {_normalize_model(model).title()} — "
        f"precio: {asking_label} vs mediana mercado: {market_label}{ref_note} "
        f"({discount_str}). {km_str}.{motivated_note} Score: {final_score:.0f}/100"
    )

    return {
        "score": final_score,
        "discount_pct": discount_pct,
        "is_deal": is_deal,
        "deal_reason": deal_reason,
        "market_price_ars": market_ars,
        "price_usd_equiv": price_usd_equiv,
        "percentile_rank": percentile_rank,
        "ref_type": ref_type,
        "confidence_index": confidence_index,
    }


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------

def update_market_references(session) -> None:
    combos = (
        session.query(Listing.brand, Listing.model, Listing.year)
        .filter(Listing.price_ars > 0)
        .distinct()
        .all()
    )
    updated = 0
    for brand, model, year in combos:
        if not brand or not model or not year:
            continue
        try:
            calculate_market_reference(session, brand, model, year)
            updated += 1
        except Exception as e:
            logger.warning(f"Market ref update failed for {brand} {model} {year}: {e}")
    logger.info(f"Updated {updated} market references")


def update_segment_velocity(session) -> int:
    """
    Compute avg/median days-to-sale per brand/model/year from confirmed sold listings.
    Requires at least 2 sold samples per segment.
    Returns count of segments updated.
    """
    from collections import defaultdict

    sold = session.query(
        Listing.brand, Listing.model, Listing.year,
        Listing.first_seen, Listing.sold_at
    ).filter(
        Listing.status == "sold",
        Listing.sold_at.isnot(None),
        Listing.first_seen.isnot(None),
        Listing.brand.isnot(None),
        Listing.model.isnot(None),
        Listing.year.isnot(None),
    ).all()

    groups: dict = defaultdict(list)
    for brand, model, year, first_seen, sold_at in sold:
        days = (sold_at - first_seen).days
        if 0 <= days <= 365:
            base_model = _normalize_model(model or "")
            groups[(brand, base_model, year)].append(days)

    updated = 0
    for (brand, base_model, year), days_list in groups.items():
        if len(days_list) < 2:
            continue
        avg = sum(days_list) / len(days_list)
        med = statistics.median(days_list)
        try:
            ref = session.query(SegmentVelocity).filter_by(
                brand=brand, model=base_model, year=year
            ).first()
            if ref:
                ref.avg_days_to_sale   = avg
                ref.median_days_to_sale = med
                ref.sample_count       = len(days_list)
                ref.updated_at         = datetime.utcnow()
            else:
                session.add(SegmentVelocity(
                    brand=brand, model=base_model, year=year,
                    avg_days_to_sale=avg, median_days_to_sale=med,
                    sample_count=len(days_list), updated_at=datetime.utcnow(),
                ))
            updated += 1
        except Exception as e:
            logger.warning(f"SegmentVelocity update failed {brand} {base_model} {year}: {e}")
            session.rollback()

    try:
        session.commit()
    except Exception as e:
        session.rollback()
        logger.error(f"update_segment_velocity commit failed: {e}")

    logger.info(f"Segment velocity: {updated} segments updated from {len(sold)} sold listings")
    return updated


def rescore_all_active_listings(session) -> tuple[int, int]:
    """
    Re-score every active listing using the current comparables DB.
    Fixes stale ref_types from sequential batch scoring (early listings
    scored before later-batch comparables were committed).
    Returns (rescored_count, deals_upgraded).
    """
    from config import get_usd_mep_rate
    usd_rate = get_usd_mep_rate()
    listings = session.query(Listing).filter(
        Listing.status == "active",
        Listing.hidden != True,
        Listing.is_agency != True,
        (Listing.price_ars > 0) | (Listing.price_usd_equiv > 0),
    ).all()
    rescored = 0
    upgraded = 0
    for obj in listings:
        try:
            listing_dict = {
                "id": obj.id,
                "brand": obj.brand,
                "model": obj.model,
                "year": obj.year,
                "km": obj.km,
                "price_ars": obj.price_ars,
                "price_usd": obj.price_usd,
                "days_on_market": (datetime.utcnow() - obj.first_seen).days if obj.first_seen else 0,
                "price_changes_count": obj.price_changes_count or 0,
            }
            result = score_listing(session, listing_dict)
            old_ref = obj.ref_type
            obj.score = result["score"]
            obj.discount_pct = result["discount_pct"]
            obj.deal_reason = result["deal_reason"]
            obj.market_price_ars = result.get("market_price_ars")
            obj.price_usd_equiv = result.get("price_usd_equiv")
            obj.percentile_rank = result.get("percentile_rank")
            obj.ref_type = result.get("ref_type")
            obj.confidence_index = result.get("confidence_index")
            if result["is_deal"] and not obj.is_deal:
                obj.is_deal = True
                obj.alerted = False
                upgraded += 1
            elif not result["is_deal"]:
                obj.is_deal = False
                obj.alerted = False
            rescored += 1
        except Exception as e:
            logger.debug(f"Rescore failed for {obj.id}: {e}")
    try:
        session.commit()
    except Exception as e:
        session.rollback()
        logger.error(f"Rescore batch commit failed: {e}")
    logger.info(f"rescore_all_active_listings: {rescored} rescored, {upgraded} new deals")
    return rescored, upgraded


def _record_price_event(session, listing_obj, event_type: str, now: datetime):
    """Append a price event to PriceHistory. Call after price is set on listing_obj."""
    from database import PriceHistory
    days = (now - listing_obj.first_seen).days if listing_obj.first_seen else 0
    session.add(PriceHistory(
        listing_id=listing_obj.id,
        price_ars=listing_obj.price_ars,
        price_usd_equiv=listing_obj.price_usd_equiv,
        recorded_at=now,
        days_on_market=days,
        event_type=event_type,
    ))


def process_listings(listings_dicts: list[dict]) -> tuple[int, int, int, list]:
    session = SessionLocal()
    new_count = 0
    deal_count = 0
    updated_count = 0
    new_ml_ids = []
    seen_ids: set[str] = set()   # track all IDs processed this scan

    skipped_distance = 0

    try:
        for listing_dict in listings_dicts:
            listing_id = listing_dict.get("id")
            if not listing_id:
                continue

            try:
                # Skip listings with no price — they can't be scored and pollute the DB
                if not listing_dict.get("price_ars") and not listing_dict.get("price_usd"):
                    continue

                lat, lon, city = coords_from_listing_dict(listing_dict)
                distance = distance_from_darregueira(lat, lon)
                if distance is not None and distance > config.MAX_DISTANCE_KM:
                    skipped_distance += 1
                    continue

                seen_ids.add(listing_id)
                existing = session.query(Listing).filter_by(id=listing_id).first()
                now = datetime.utcnow()

                if existing:
                    if existing.hidden or existing.is_agency:
                        existing.last_seen = now
                        session.commit()
                        continue

                    # Enrich listing_dict with market signals from existing DB row
                    listing_dict = dict(listing_dict)
                    listing_dict["days_on_market"] = (now - existing.first_seen).days if existing.first_seen else 0
                    listing_dict["price_changes_count"] = existing.price_changes_count or 0

                    # Detect price change — record in history
                    new_price_ars = listing_dict.get("price_ars")
                    old_price_ars = existing.price_ars
                    price_changed = (
                        new_price_ars and old_price_ars
                        and abs(new_price_ars - old_price_ars) / old_price_ars > 0.005  # >0.5% change
                    )

                    existing.last_seen = now
                    existing.price_ars = new_price_ars
                    existing.price_usd = listing_dict.get("price_usd")
                    existing.title = listing_dict.get("title", existing.title)
                    existing.km = listing_dict.get("km", existing.km)
                    existing.thumbnail = listing_dict.get("thumbnail", existing.thumbnail)
                    existing.seller_lat = lat
                    existing.seller_lon = lon
                    existing.seller_city = city
                    existing.distance_km = distance
                    existing.status = "active"  # re-activate if it was marked stale

                    if price_changed:
                        existing.price_changes_count = (existing.price_changes_count or 0) + 1
                        existing.last_price_change = now
                        _record_price_event(session, existing, "price_change", now)

                    listing_obj = existing
                    updated_count += 1
                else:
                    listing_obj = Listing(
                        id=listing_id,
                        source=listing_dict.get("source", ""),
                        title=listing_dict.get("title", ""),
                        brand=listing_dict.get("brand", ""),
                        model=listing_dict.get("model", ""),
                        year=listing_dict.get("year"),
                        km=listing_dict.get("km"),
                        price_ars=listing_dict.get("price_ars"),
                        price_usd=listing_dict.get("price_usd"),
                        fuel=listing_dict.get("fuel", ""),
                        transmission=listing_dict.get("transmission", ""),
                        condition=listing_dict.get("condition", "used"),
                        url=listing_dict.get("url", ""),
                        thumbnail=listing_dict.get("thumbnail", ""),
                        raw_data=listing_dict.get("raw_data"),
                        seller_lat=lat,
                        seller_lon=lon,
                        seller_city=city,
                        distance_km=distance,
                        first_seen=now,
                        last_seen=now,
                        status="active",
                        price_changes_count=0,
                        is_agency=bool(listing_dict.get("is_agency", False)),
                    )
                    session.add(listing_obj)
                    new_count += 1
                    if listing_dict.get("source") == "mercadolibre":
                        new_ml_ids.append(listing_id)

                session.flush()

                score_result = score_listing(session, listing_dict)
                listing_obj.score = score_result["score"]
                listing_obj.discount_pct = score_result["discount_pct"]
                listing_obj.deal_reason = score_result["deal_reason"]
                listing_obj.market_price_ars = score_result.get("market_price_ars")
                listing_obj.price_usd_equiv = score_result.get("price_usd_equiv")
                listing_obj.percentile_rank = score_result.get("percentile_rank")
                listing_obj.ref_type = score_result.get("ref_type")
                listing_obj.confidence_index = score_result.get("confidence_index")

                if score_result["is_deal"] and not listing_obj.is_deal:
                    listing_obj.is_deal = True
                    listing_obj.alerted = False
                    deal_count += 1
                elif not score_result["is_deal"]:
                    listing_obj.is_deal = False

                # Record initial price event for new listings only
                if listing_obj.first_seen == now:
                    _record_price_event(session, listing_obj, "initial", now)

                session.commit()

            except Exception as e:
                session.rollback()
                logger.warning(f"Error processing listing {listing_id}: {e}")
                continue

        # After processing all listings: mark unseen active listings as sold.
        # Only mark listings from sources that actually returned data this scan —
        # if a scraper failed (0 results), we don't want to purge its listings.
        scraped_sources = {d.get("source") for d in listings_dicts if d.get("source")}
        try:
            _mark_sold_listings(session, seen_ids, scraped_sources)
        except Exception as e:
            logger.warning(f"mark_sold_listings failed: {e}")

    finally:
        session.close()

    logger.info(
        f"process_listings: new={new_count}, updated={updated_count}, deals={deal_count}, "
        f"skipped_distance={skipped_distance} (>{config.MAX_DISTANCE_KM}km from {ORIGIN_NAME})"
    )
    return new_count, deal_count, updated_count, new_ml_ids


def _mark_sold_listings(session, seen_ids: set, scraped_sources: Optional[set] = None) -> int:
    """
    Listings not seen in this scan and not updated for 2+ scan intervals
    are likely sold. Mark status='sold' and record a 'sold' price event.
    Only marks listings from sources that actually returned data this scan —
    if a scraper failed (e.g. ML blocked), its listings are preserved.
    Returns count of newly sold listings.
    """
    from database import PriceHistory
    cutoff = datetime.utcnow() - timedelta(minutes=2 * config.SCAN_INTERVAL_MINUTES)
    q = (
        session.query(Listing)
        .filter(
            Listing.status == "active",
            Listing.last_seen < cutoff,
            Listing.hidden != True,
            Listing.is_agency != True,
            Listing.price_ars > 0,
        )
    )
    # Only consider listings from sources that actually ran this scan
    if scraped_sources:
        q = q.filter(Listing.source.in_(scraped_sources))
    candidates = q.all()
    sold_count = 0
    now = datetime.utcnow()
    for lst in candidates:
        if lst.id in seen_ids:
            continue  # was seen this scan, skip
        lst.status = "sold"
        lst.sold_at = lst.last_seen   # best estimate: last time we saw it
        days = (lst.last_seen - lst.first_seen).days if lst.first_seen and lst.last_seen else 0
        session.add(PriceHistory(
            listing_id=lst.id,
            price_ars=lst.price_ars,
            price_usd_equiv=lst.price_usd_equiv,
            recorded_at=now,
            days_on_market=days,
            event_type="sold",
        ))
        sold_count += 1

    if sold_count:
        session.commit()
        logger.info(f"Panel: marked {sold_count} listings as sold")
    return sold_count
