"""
MercadoLibre OAuth2 token manager.

Supports two flows:
1. authorization_code — user logs in once via /admin/ml_login, we store the
   refresh_token in the DB. This is the correct flow for the Search API.
2. client_credentials — fallback; works for some endpoints but NOT search.

Token lifecycle:
  access_token  valid for 6 hours (21600 s) — refreshed automatically
  refresh_token valid for 6 months — persisted in DB (model_artifacts table)
"""
import asyncio
import logging
import time
from typing import Optional

import httpx

import config

logger = logging.getLogger(__name__)

ML_TOKEN_URL = "https://api.mercadolibre.com/oauth/token"
ML_AUTH_URL  = "https://auth.mercadolibre.com.ar/authorization"
_REFRESH_BUFFER = 300          # refresh 5 min before expiry
_REFRESH_TOKEN_KEY = "ml_refresh_token"


def _load_refresh_token() -> Optional[str]:
    """Load persisted ML refresh_token from DB."""
    try:
        from database import SessionLocal, ModelArtifact
        session = SessionLocal()
        try:
            row = session.query(ModelArtifact).filter_by(model_name=_REFRESH_TOKEN_KEY).first()
            if row and row.artifact:
                return row.artifact.decode()
        finally:
            session.close()
    except Exception as e:
        logger.debug(f"Could not load ML refresh token: {e}")
    return None


def _save_refresh_token(token: str) -> None:
    """Persist ML refresh_token to DB."""
    try:
        from datetime import datetime
        from database import SessionLocal, ModelArtifact
        session = SessionLocal()
        try:
            row = session.query(ModelArtifact).filter_by(model_name=_REFRESH_TOKEN_KEY).first()
            if row:
                row.artifact = token.encode()
                row.updated_at = datetime.utcnow()
            else:
                session.add(ModelArtifact(
                    model_name=_REFRESH_TOKEN_KEY,
                    version=1,
                    artifact=token.encode(),
                    trained_at=datetime.utcnow(),
                ))
            session.commit()
            logger.info("ML refresh_token saved to DB")
        finally:
            session.close()
    except Exception as e:
        logger.warning(f"Could not save ML refresh token: {e}")


class MLTokenManager:
    def __init__(self):
        self._token: Optional[str] = None
        self._expires_at: float = 0.0
        self._refresh_token: Optional[str] = None
        self._lock = asyncio.Lock()
        self._loaded_from_db = False

    def _ensure_loaded(self):
        if not self._loaded_from_db:
            self._refresh_token = _load_refresh_token()
            self._loaded_from_db = True

    def set_tokens(self, access_token: str, refresh_token: str, expires_in: int = 21600):
        """Called after a successful authorization_code exchange."""
        self._token = access_token
        self._expires_at = time.time() + expires_in
        self._refresh_token = refresh_token
        self._loaded_from_db = True
        _save_refresh_token(refresh_token)
        logger.info("ML tokens set via authorization_code flow")

    async def get_token(self, client: httpx.AsyncClient) -> Optional[str]:
        """Return a valid access token, refreshing if needed."""
        if not config.ML_APP_ID or not config.ML_CLIENT_SECRET:
            return None

        async with self._lock:
            self._ensure_loaded()

            if self._token and time.time() < self._expires_at - _REFRESH_BUFFER:
                return self._token

            # Try refresh_token flow first (authorization_code derived)
            if self._refresh_token:
                success = await self._refresh_with_token(client)
                if success:
                    return self._token

            # Fallback: client_credentials (limited scope — search will 403)
            await self._refresh_client_credentials(client)
            return self._token

    async def _refresh_with_token(self, client: httpx.AsyncClient) -> bool:
        """Refresh access_token using the stored refresh_token."""
        try:
            resp = await client.post(
                ML_TOKEN_URL,
                data={
                    "grant_type": "refresh_token",
                    "client_id": config.ML_APP_ID,
                    "client_secret": config.ML_CLIENT_SECRET,
                    "refresh_token": self._refresh_token,
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                timeout=15.0,
            )
            resp.raise_for_status()
            data = resp.json()
            self._token = data["access_token"]
            self._expires_at = time.time() + data.get("expires_in", 21600)
            # ML rotates refresh_tokens — save the new one
            new_rt = data.get("refresh_token")
            if new_rt and new_rt != self._refresh_token:
                self._refresh_token = new_rt
                _save_refresh_token(new_rt)
            logger.info("ML access_token refreshed via refresh_token")
            return True
        except Exception as e:
            logger.warning(f"ML refresh_token flow failed: {e} — clearing stored token")
            self._refresh_token = None
            self._token = None
            return False

    async def _refresh_client_credentials(self, client: httpx.AsyncClient) -> None:
        """Fallback: client_credentials grant (does NOT work for search API)."""
        try:
            resp = await client.post(
                ML_TOKEN_URL,
                data={
                    "grant_type": "client_credentials",
                    "client_id": config.ML_APP_ID,
                    "client_secret": config.ML_CLIENT_SECRET,
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                timeout=15.0,
            )
            resp.raise_for_status()
            data = resp.json()
            self._token = data["access_token"]
            self._expires_at = time.time() + data.get("expires_in", 21600)
            logger.info("ML token via client_credentials (search API will be limited)")
        except Exception as e:
            logger.warning(f"ML client_credentials failed: {e}")
            self._token = None

    def get_authorization_url(self) -> str:
        """Build the URL the user must visit to authorize the app."""
        return (
            f"{ML_AUTH_URL}"
            f"?response_type=code"
            f"&client_id={config.ML_APP_ID}"
            f"&redirect_uri=https://cardeal.ar/admin/ml_callback"
        )


# Module-level singleton
_manager = MLTokenManager()


async def get_auth_headers(client: httpx.AsyncClient) -> dict:
    """Return Authorization headers if credentials are configured, else empty dict."""
    token = await _manager.get_token(client)
    if token:
        return {"Authorization": f"Bearer {token}"}
    return {}


def get_authorization_url() -> str:
    return _manager.get_authorization_url()
