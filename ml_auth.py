"""
MercadoLibre OAuth2 token manager.
Uses client_credentials flow to get and auto-refresh access tokens.

Token is valid for 6 hours (21600 seconds). We refresh 5 minutes before expiry.
"""
import asyncio
import logging
import time
from typing import Optional

import httpx

import config

logger = logging.getLogger(__name__)

ML_TOKEN_URL = "https://api.mercadolibre.com/oauth/token"
_REFRESH_BUFFER = 300  # refresh 5 min before expiry


class MLTokenManager:
    def __init__(self):
        self._token: Optional[str] = None
        self._expires_at: float = 0.0
        self._lock = asyncio.Lock()

    async def get_token(self, client: httpx.AsyncClient) -> Optional[str]:
        """Return a valid access token, refreshing if needed."""
        if not config.ML_APP_ID or not config.ML_CLIENT_SECRET:
            return None

        async with self._lock:
            if self._token and time.time() < self._expires_at - _REFRESH_BUFFER:
                return self._token
            await self._refresh(client)
            return self._token

    async def _refresh(self, client: httpx.AsyncClient) -> None:
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
            logger.info("ML OAuth token refreshed (expires in %ds)", data.get("expires_in", 21600))
        except Exception as e:
            logger.warning("ML OAuth token refresh failed: %s", e)
            self._token = None


# Module-level singleton
_manager = MLTokenManager()


async def get_auth_headers(client: httpx.AsyncClient) -> dict:
    """Return Authorization headers if credentials are configured, else empty dict."""
    token = await _manager.get_token(client)
    if token:
        return {"Authorization": f"Bearer {token}"}
    return {}
