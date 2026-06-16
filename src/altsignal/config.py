"""Runtime configuration and secrets, loaded from environment (+ optional .env)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

# Load .env if python-dotenv is installed (it's a core dep, but stay defensive).
try:  # pragma: no cover - trivial
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover
    pass

# src/altsignal/config.py -> parents[2] == project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
CONFIGS_DIR = PROJECT_ROOT / "configs"
REPORTS_DIR = PROJECT_ROOT / "reports_out"


def _env(name: str, default: str = "") -> str:
    return (os.getenv(name) or "").strip() or default


@dataclass(frozen=True)
class Settings:
    """All tunables + secrets in one place. Construct via :func:`get_settings`."""

    contact_email: str
    sec_user_agent: str
    browser_user_agent: str
    fred_api_key: str | None
    reddit_client_id: str | None
    reddit_client_secret: str | None
    reddit_user_agent: str
    cache_ttl: int
    db_path: Path
    industries_path: Path

    @classmethod
    def load(cls) -> "Settings":
        contact = _env("ALTSIGNAL_CONTACT_EMAIL", "anonymous@example.com")
        sec_ua = _env("ALTSIGNAL_SEC_USER_AGENT", f"altsignal/0.1 ({contact})")
        db = _env("ALTSIGNAL_DB_PATH")
        try:
            ttl = int(_env("ALTSIGNAL_CACHE_TTL", "86400"))
        except ValueError:
            ttl = 86400
        return cls(
            contact_email=contact,
            sec_user_agent=sec_ua,
            # A realistic desktop UA helps with endpoints that reject obvious bots
            # (e.g. Google Trends). Used only for free/public endpoints.
            browser_user_agent=_env(
                "ALTSIGNAL_BROWSER_USER_AGENT",
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
            ),
            fred_api_key=_env("FRED_API_KEY") or None,
            reddit_client_id=_env("REDDIT_CLIENT_ID") or None,
            reddit_client_secret=_env("REDDIT_CLIENT_SECRET") or None,
            reddit_user_agent=_env("REDDIT_USER_AGENT", "altsignal/0.1"),
            cache_ttl=ttl,
            db_path=Path(db) if db else (DATA_DIR / "altsignal.sqlite"),
            industries_path=CONFIGS_DIR / "industries.toml",
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings.load()
