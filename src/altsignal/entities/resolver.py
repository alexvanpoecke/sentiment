"""Resolve a ticker/name into a rich Entity and route it to relevant signals.

This is the "think downstream" step: SEC tells us the SIC code; the industry
routing table (configs/industries.toml) tells us which connectors and seed
search terms make sense for that kind of business.
"""

from __future__ import annotations

import tomllib
from functools import lru_cache

from ..config import Settings, get_settings
from ..models import Entity
from ..registry import get_connector


@lru_cache(maxsize=4)
def _load_industries_cached(path_str: str) -> dict:
    with open(path_str, "rb") as f:
        return tomllib.load(f)


def load_industries(settings: Settings | None = None) -> dict:
    settings = settings or get_settings()
    return _load_industries_cached(str(settings.industries_path))


def match_industry(sic: str | None, cfg: dict) -> dict | None:
    if not sic:
        return None
    try:
        sic_i = int(sic)
    except ValueError:
        return None
    industries = cfg.get("industry", [])
    for ind in industries:  # exact SIC match wins
        if sic_i in ind.get("sic", []):
            return ind
    s = str(sic_i)  # else fall back to 2-digit major group
    for ind in industries:
        if any(str(code)[:2] == s[:2] for code in ind.get("sic", [])):
            return ind
    return None


def substitute(terms: list[str], brand: str, ticker: str | None) -> list[str]:
    out, seen = [], set()
    for t in terms:
        v = t.replace("{brand}", brand).replace("{ticker}", ticker or "").strip()
        if v and v not in seen:
            seen.add(v)
            out.append(v)
    return out


def resolve(query: str, settings: Settings | None = None, store=None, edgar=None) -> Entity:
    settings = settings or get_settings()
    cfg = load_industries(settings)
    edgar = edgar or get_connector("edgar", store=store, settings=settings)

    ent = Entity(query=query)
    r = edgar.resolve(query)
    if not r:
        # Couldn't map to a SEC filer — still return a usable entity with defaults.
        ent.name = query
        ent.brands = [ent.short_name]
        ent.connectors = list(cfg.get("default_connectors", []))
        ent.seed_terms = substitute(
            list(cfg.get("default_seed_terms", [])), ent.short_name, None
        )
        return ent

    ent.cik, ent.ticker, ent.name = r["cik"], r["ticker"], r["title"]
    try:
        subs = edgar.submissions(r["cik"])
    except Exception:
        subs = {}
    ent.sic = str(subs["sic"]) if subs.get("sic") else None
    ent.sic_description = subs.get("sicDescription")
    ent.fiscal_year_end = subs.get("fiscalYearEnd")
    ent.aliases = list(subs.get("tickers", []) or [])
    business = (subs.get("addresses") or {}).get("business") or {}
    ent.country = business.get("stateOrCountry")
    ent.brands = [ent.short_name]

    ind = match_industry(ent.sic, cfg)
    if ind:
        ent.industry_key = ind.get("key")
        ent.sector = ind.get("label")
        ent.connectors = list(ind.get("connectors", cfg.get("default_connectors", [])))
        ent.macro_series = list(ind.get("macro_series", []))
        ent.subreddits = list(ind.get("subreddits", []))
        seed = ind.get("seed_terms") or cfg.get("default_seed_terms", [])
    else:
        ent.sector = ent.sic_description
        ent.connectors = list(cfg.get("default_connectors", []))
        seed = cfg.get("default_seed_terms", [])
    ent.seed_terms = substitute(list(seed), ent.short_name, ent.ticker)
    return ent
