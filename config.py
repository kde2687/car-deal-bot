import os
import time
import logging
import threading
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dólar MEP — actualizado desde dolarapi.com, cacheado 24hs
# ---------------------------------------------------------------------------
_usd_mep_cache: dict = {"rate": None, "ts": 0.0}
_usd_mep_lock = threading.RLock()
_USD_MEP_FALLBACK = 1300.0  # Actualizado abril 2026 — revisar periódicamente
_USD_MEP_TTL = 14400  # 4 horas (era 24h — demasiado tiempo en mercado volátil)


def get_usd_mep_rate() -> float:
    """Fetch MEP dollar rate from dolarapi.com, cache for 24h. Thread-safe."""
    global _usd_mep_cache
    with _usd_mep_lock:
        now = time.time()
        if _usd_mep_cache["rate"] and now - _usd_mep_cache["ts"] < _USD_MEP_TTL:
            return _usd_mep_cache["rate"]
        try:
            import urllib.request, json
            req = urllib.request.Request(
                "https://dolarapi.com/v1/dolares/bolsa",
                headers={"User-Agent": "Mozilla/5.0 (compatible; CarDealBot/1.0)"},
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read())
            rate = float(data.get("venta") or data.get("compra") or _USD_MEP_FALLBACK)
            _usd_mep_cache = {"rate": rate, "ts": now}
            logger.info(f"Dólar MEP actualizado: {rate:.0f} ARS/USD")
            return rate
        except Exception as e:
            logger.warning(f"No se pudo obtener dólar MEP: {e} — usando fallback {_USD_MEP_FALLBACK}")
            rate = _usd_mep_cache["rate"] or _USD_MEP_FALLBACK
            # Cache the fallback so we don't retry on every listing
            _usd_mep_cache = {"rate": rate, "ts": now}
            return rate

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
DEAL_SCORE_THRESHOLD = int(os.getenv("DEAL_SCORE_THRESHOLD", "15"))
SCAN_INTERVAL_MINUTES = int(os.getenv("SCAN_INTERVAL_MINUTES", "30"))
MIN_YEAR = int(os.getenv("MIN_YEAR", "2010"))
MAX_KM = int(os.getenv("MAX_KM", "200000"))
MIN_KM  = int(os.getenv("MIN_KM", "500"))      # Mínimo km — descarta 0km/nuevos
BRANDS = os.getenv("BRANDS", "Toyota,Ford,Volkswagen,Chevrolet,Renault,Peugeot,Fiat,Citroen,Nissan,Honda,Hyundai,Kia,Jeep,Dodge,Mitsubishi,Suzuki,MG,Haval,Chery,JAC,BYD,DFSK,Changan,Geely,Jetour")
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///deals.db")
FLASK_SECRET_KEY = os.getenv("FLASK_SECRET_KEY", "changeme")
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

BRANDS_LIST = [b.strip() for b in BRANDS.split(",") if b.strip()]

def min_price_for_year(year: int) -> float:
    """Minimum realistic ARS price for a car of the given year (anti-anticipo filter)."""
    if not year:
        return MIN_PRICE_ARS
    extra = max(0, year - 2015) * MIN_PRICE_ARS_PER_YEAR_INCREMENT
    return MIN_PRICE_ARS + extra

MAX_DISTANCE_KM = float(os.getenv("MAX_DISTANCE_KM", "1000"))

# Price sanity filters
MIN_PRICE_ARS = float(os.getenv("MIN_PRICE_ARS", "8000000"))    # $8M ARS mínimo base
# Precio mínimo por año: $8M base + $1M por año desde 2015 (ej: 2022 → $15M mínimo).
# El incremento fue reducido de $1.5M a $1M para que autos de años recientes con
# descuentos genuinos del 15-25% no sean descartados antes de llegar al scoring.
# A $1M/año, un auto 2023 tiene piso de $16M ARS (~USD 10.7k al MEP de $1500),
# lo cual sigue capturando precios de anticipo/fraude pero no corta deals reales.
MIN_PRICE_ARS_PER_YEAR_INCREMENT = float(os.getenv("MIN_PRICE_ARS_PER_YEAR_INCREMENT", "1000000"))
MAX_PRICE_ARS = float(os.getenv("MAX_PRICE_ARS", "200000000"))  # $200M ARS máximo
MIN_PRICE_USD = float(os.getenv("MIN_PRICE_USD", "3000"))       # USD 3.000 mínimo
MAX_PRICE_USD = float(os.getenv("MAX_PRICE_USD", "150000"))     # USD 150.000 máximo

# Seller type filter
ONLY_PRIVATE_SELLERS = os.getenv("ONLY_PRIVATE_SELLERS", "true").lower() == "true"

# Tipo de cambio — se usa get_usd_mep_rate() en runtime; este es el fallback estático
USD_TO_ARS_RATE = float(os.getenv("USD_TO_ARS_RATE", "1350"))

# Market reference window and decay
MARKET_HISTORY_DAYS  = int(os.getenv("MARKET_HISTORY_DAYS", "365"))   # 1 año de historia
MARKET_HALF_LIFE_DAYS = int(os.getenv("MARKET_HALF_LIFE_DAYS", "14")) # peso 0.5 a los 14 días (era 30 — muy largo para inflación 140% anual)
MARKET_KM_TOLERANCE  = int(os.getenv("MARKET_KM_TOLERANCE", "40000")) # ±40k km para referencia exacta

# MercadoLibre API credentials
ML_APP_ID = os.getenv("ML_APP_ID", "")
ML_CLIENT_SECRET = os.getenv("ML_CLIENT_SECRET", "")

# Outbound proxy for ML requests to bypass datacenter IP blocks.
# Single proxy:  ML_PROXY_URL=http://user:pass@1.2.3.4:1234
# Multiple (rotated randomly): ML_PROXY_URLS=http://u:p@1.2.3.4:1234,http://u:p@5.6.7.8:5678
ML_PROXY_URL  = os.getenv("ML_PROXY_URL", "")
ML_PROXY_URLS = [u.strip() for u in os.getenv("ML_PROXY_URLS", "").split(",") if u.strip()]

# Email digest
SMTP_USER      = os.getenv("SMTP_USER", "")
SMTP_PASSWORD  = os.getenv("SMTP_PASSWORD", "")
DIGEST_RECIPIENT = os.getenv("DIGEST_RECIPIENT", "")

# Dashboard public URL (used in email digests and Telegram alerts)
DASHBOARD_URL = os.getenv("DASHBOARD_URL", "https://cardeal.ar")
