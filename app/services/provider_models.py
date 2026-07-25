"""Fetch provider model catalogs without any Qt/UI dependency."""
from __future__ import annotations

from typing import Dict, List, Optional

import requests

from app.core.providers import PROVIDER_BASE_URLS, filter_models


def fetch_provider_catalog(
    provider: str,
    api_key: str,
    proxies: Optional[Dict[str, str]] = None,
    timeout: float = 10.0,
) -> Dict[str, List[str]]:
    """Fetch one provider catalog and filter it for both supported features."""
    if provider not in PROVIDER_BASE_URLS:
        raise ValueError(f"Model discovery is not available for provider '{provider}'.")
    if not api_key.strip():
        raise ValueError(f"An API key is required to load {provider} models.")
    response = requests.get(
        f"{PROVIDER_BASE_URLS[provider]}/models",
        headers={"Authorization": f"Bearer {api_key.strip()}"},
        proxies=proxies,
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    catalog = {
        feature: filter_models(payload, feature)
        for feature in ("transcription", "rephrasing")
    }
    if not any(catalog.values()):
        raise ValueError(f"The {provider} model catalog contained no supported models.")
    return catalog
