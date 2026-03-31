"""
Hedonic pricing model — LightGBM trained on the full listing universe.

Architecture:
    features: brand (target-encoded), model (target-encoded), antigüedad (years),
              km, province (target-encoded)
    target:   log(price_usd)
    output:   predicted price_usd for any brand/model/year/km combination

Activation threshold: MIN_TRAINING_SAMPLES listings in the database.
Below that threshold, the scorer falls back to weighted median / depreciation curve.

Usage:
    from pricing_model.pipeline import PricingPipeline
    pipeline = PricingPipeline()
    pipeline.train(session)          # or loads cached model automatically
    price = pipeline.predict(brand, model, year, km, province)  # USD
    # Returns None if model not trained yet
"""

import logging
import math
import os
import pickle
import threading
from datetime import datetime
from typing import Optional

MODEL_VERSION = 2  # increment when features or schema change

logger = logging.getLogger(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")
MIN_TRAINING_SAMPLES = 300   # minimum listings before activating ML model
RETRAIN_EVERY_HOURS  = 24    # retrain at most once per day

_pipeline_cache: dict = {"model": None, "trained_at": None}


class PricingPipeline:
    """
    Wraps LightGBM training + inference with target encoding for categoricals.
    Falls back gracefully (returns None) when not enough data or not trained yet.
    """

    def __init__(self):
        self._model       = None
        self._encoders    = {}   # target encoders per categorical column
        self._trained_at  = None
        self._load_cached()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def is_ready(self) -> bool:
        return self._model is not None

    def needs_retraining(self) -> bool:
        if self._trained_at is None:
            return True
        hours = (datetime.utcnow() - self._trained_at).total_seconds() / 3600
        return hours >= RETRAIN_EVERY_HOURS

    def train(self, session) -> bool:
        """
        Fetch all listings from DB, engineer features, train LightGBM.
        Returns True if training succeeded.
        """
        try:
            import lightgbm as lgb
            import numpy as np
        except ImportError:
            logger.warning("lightgbm/numpy not installed — pricing model unavailable")
            return False

        from database import Listing
        from config import get_usd_mep_rate
        usd_rate = get_usd_mep_rate()

        rows = session.query(
            Listing.brand, Listing.model, Listing.year,
            Listing.km, Listing.seller_city,
            Listing.fuel, Listing.transmission,
            Listing.price_ars, Listing.price_usd_equiv
        ).filter(
            Listing.price_ars > 0,
            Listing.year.isnot(None),
            Listing.km.isnot(None),
            Listing.is_agency != True,
            Listing.hidden != True,
        ).all()

        if len(rows) < MIN_TRAINING_SAMPLES:
            logger.info(f"Pricing model: only {len(rows)} samples, need {MIN_TRAINING_SAMPLES} — skipping training")
            return False

        current_year = datetime.utcnow().year
        records = []
        for brand, model, year, km, city, fuel, transmission, price_ars, price_usd_equiv in rows:
            usd = price_usd_equiv if price_usd_equiv and price_usd_equiv > 0 \
                  else (price_ars / usd_rate if price_ars else None)
            if not usd or usd <= 0:
                continue
            antigüedad = current_year - (year or current_year)
            if antigüedad < 0 or antigüedad > 30:
                continue
            records.append({
                "brand":        (brand or "").strip().lower(),
                "model":        self._base_model(model or ""),
                "antiguedad":   float(antigüedad),
                "km":           float(km or 0),
                "province":     (city or "").strip().lower(),
                "fuel":         (fuel or "").strip().lower(),
                "transmission": (transmission or "").strip().lower(),
                "log_price":    math.log(usd),
            })

        if len(records) < MIN_TRAINING_SAMPLES:
            return False

        import numpy as np

        # Target encoding for categoricals
        for col in ("brand", "model", "province", "fuel", "transmission"):
            encoder = {}
            global_mean = sum(r["log_price"] for r in records) / len(records)
            groups: dict = {}
            for r in records:
                groups.setdefault(r[col], []).append(r["log_price"])
            for key, vals in groups.items():
                # Bayesian smoothing: blend group mean with global mean
                n = len(vals)
                smooth = 10  # weight of global mean
                encoder[key] = (sum(vals) + smooth * global_mean) / (n + smooth)
            encoder["__default__"] = global_mean
            self._encoders[col] = encoder

        X = np.array([
            [
                self._encode(r["brand"], "brand"),
                self._encode(r["model"], "model"),
                r["antiguedad"],
                r["km"],
                self._encode(r["province"],     "province"),
                self._encode(r["fuel"],         "fuel"),
                self._encode(r["transmission"], "transmission"),
            ]
            for r in records
        ])
        y = np.array([r["log_price"] for r in records])

        dataset = lgb.Dataset(X, label=y, feature_name=[
            "brand_enc", "model_enc", "antiguedad", "km",
            "province_enc", "fuel_enc", "transmission_enc"
        ])
        params = {
            "objective":      "regression",
            "metric":         "rmse",
            "num_leaves":     31,
            "learning_rate":  0.05,
            "n_estimators":   200,
            "min_child_samples": 5,
            "verbose":        -1,
        }
        self._model = lgb.train(params, dataset, num_boost_round=200,
                                valid_sets=[dataset], callbacks=[lgb.log_evaluation(period=-1)])
        self._trained_at = datetime.utcnow()
        self._save_cached()
        logger.info(f"Pricing model trained on {len(records)} listings")
        return True

    def predict(self, brand: str, model: str, year: int,
                km: int, province: str = "") -> Optional[float]:
        """
        Returns estimated price in USD, or None if model not ready.
        """
        if not self._model:
            return None
        try:
            import numpy as np
            from scorer import _normalize_model
            current_year = datetime.utcnow().year
            antigüedad = float(current_year - (year or current_year))
            X = np.array([[
                self._encode((brand or "").lower(),        "brand"),
                self._encode(_normalize_model(model),      "model"),
                antigüedad,
                float(km or 0),
                self._encode((province or "").lower(),     "province"),
                self._encode("",                           "fuel"),
                self._encode("",                           "transmission"),
            ]])
            log_price = self._model.predict(X)[0]
            return math.exp(log_price)
        except Exception as e:
            logger.debug(f"Pricing model predict error: {e}")
            return None

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _base_model(self, model: str) -> str:
        from scorer import _normalize_model
        return _normalize_model(model)

    def _encode(self, value: str, col: str) -> float:
        enc = self._encoders.get(col, {})
        return enc.get(value, enc.get("__default__", 0.0))

    def _save_cached(self):
        try:
            with open(MODEL_PATH, "wb") as f:
                pickle.dump({
                    "version":    MODEL_VERSION,
                    "model":      self._model,
                    "encoders":   self._encoders,
                    "trained_at": self._trained_at,
                }, f)
        except Exception as e:
            logger.warning(f"Could not save pricing model: {e}")

    def _load_cached(self):
        if not os.path.exists(MODEL_PATH):
            return
        try:
            with open(MODEL_PATH, "rb") as f:
                data = pickle.load(f)
            if data.get("version") != MODEL_VERSION:
                logger.warning(f"Pricing model version mismatch (file={data.get('version')}, "
                               f"expected={MODEL_VERSION}) — discarding, will retrain")
                os.remove(MODEL_PATH)
                return
            self._model      = data.get("model")
            self._encoders   = data.get("encoders", {})
            self._trained_at = data.get("trained_at")
            if self._model:
                logger.info(f"Pricing model loaded from cache (trained {self._trained_at})")
        except Exception as e:
            logger.warning(f"Could not load cached pricing model: {e}")


# Module-level singleton — thread-safe
_pipeline: Optional[PricingPipeline] = None
_pipeline_lock = threading.Lock()


def get_pipeline() -> PricingPipeline:
    global _pipeline
    with _pipeline_lock:
        if _pipeline is None:
            _pipeline = PricingPipeline()
    return _pipeline


def maybe_retrain(session) -> None:
    """Call this periodically (e.g. daily). Trains only when enough data and stale."""
    p = get_pipeline()
    if p.needs_retraining():
        p.train(session)
