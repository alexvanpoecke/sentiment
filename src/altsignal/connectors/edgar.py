"""SEC EDGAR connector: entity resolution + quarterly revenue from XBRL facts.

Uses the official data.sec.gov JSON APIs (no scraping). SEC requires a
descriptive User-Agent with a contact — set ALTSIGNAL_CONTACT_EMAIL.
"""

from __future__ import annotations

from datetime import date

from ..models import Observation, Signal
from ..registry import register
from .base import Connector

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"

# Ordered preference of us-gaap revenue concepts.
REVENUE_CONCEPTS = [
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "Revenues",
    "RevenueFromContractWithCustomerIncludingAssessedTax",
    "SalesRevenueNet",
    "SalesRevenueGoodsNet",
]

_FORMS = {"10-Q", "10-K", "10-Q/A", "10-K/A"}


def _d(s: str) -> date:
    return date.fromisoformat(s)


@register
class EdgarConnector(Connector):
    source = "edgar"
    title = "SEC EDGAR (filings & XBRL fundamentals)"
    free = True
    min_interval = 0.25  # SEC fair-access: stay well under 10 req/s

    # ----------------------------------------------------------- raw fetches
    def tickers(self) -> dict[str, dict]:
        """Return {TICKER: {cik, title}} from the official ticker map."""
        raw = self.get_json("edgar:company_tickers", TICKERS_URL, ttl=7 * 86400)
        out: dict[str, dict] = {}
        for row in raw.values():
            out[str(row["ticker"]).upper()] = {
                "cik": f"{int(row['cik_str']):010d}",
                "title": row["title"],
            }
        return out

    def resolve(self, query: str) -> dict | None:
        """Resolve a ticker or company-name fragment to {cik, ticker, title}."""
        q = query.strip()
        table = self.tickers()
        if q.upper() in table:
            t = table[q.upper()]
            return {"cik": t["cik"], "ticker": q.upper(), "title": t["title"]}
        # name substring (first match)
        ql = q.lower()
        for ticker, t in table.items():
            if ql in t["title"].lower():
                return {"cik": t["cik"], "ticker": ticker, "title": t["title"]}
        return None

    def submissions(self, cik: str) -> dict:
        return self.get_json(f"edgar:submissions:{cik}", SUBMISSIONS_URL.format(cik=cik))

    def companyfacts(self, cik: str) -> dict:
        return self.get_json(f"edgar:facts:{cik}", COMPANYFACTS_URL.format(cik=cik))

    # ------------------------------------------------- quarterly revenue logic
    @staticmethod
    def _discrete_quarters(entries: list[dict]) -> dict[date, tuple[float, date]]:
        """Turn raw XBRL duration facts into discrete quarterly values.

        Handles two filer styles uniformly by grouping on the period *start* and
        differencing cumulative YTD figures:
          * cumulative filers: one start per fiscal year, increasing ends ->
            Q_i = cum(end_i) - cum(end_{i-1})
          * discrete filers: each quarter has its own ~90d start -> taken as-is
        Annual-only (~365d) spans fall out naturally (their differenced span
        is ~90d only as the trailing Q4 piece).
        Returns {end_date: (value, filed_date)}, keeping the latest filing.
        """
        # companyfacts re-reports the same (start,end) period across many filings.
        # Collapse each period to its latest filing first, so the cumulative
        # ladder we difference below is clean (and restatements win).
        dedup: dict[tuple[date, date], dict] = {}
        for e in entries:
            if e.get("form") not in _FORMS or "start" not in e or "end" not in e:
                continue
            try:
                parsed = {**e, "_start": _d(e["start"]), "_end": _d(e["end"]), "_filed": _d(e["filed"])}
            except (KeyError, ValueError):
                continue
            k = (parsed["_start"], parsed["_end"])
            cur = dedup.get(k)
            if cur is None or parsed["_filed"] > cur["_filed"]:
                dedup[k] = parsed

        groups: dict[date, list[dict]] = {}
        for e in dedup.values():
            groups.setdefault(e["_start"], []).append(e)

        discrete: dict[date, tuple[float, date]] = {}
        for start, lst in groups.items():
            lst.sort(key=lambda e: e["_end"])
            prev_end: date | None = None
            prev_val = 0.0
            for e in lst:
                period_start = prev_end or start
                span = (e["_end"] - period_start).days
                val = float(e["val"]) - prev_val
                # Skip negative differences: revenue is non-negative for these
                # concepts, so a negative almost always means adjacent cumulative
                # rungs came from inconsistent restatement vintages.
                if 78 <= span <= 100 and val >= 0:
                    cur = discrete.get(e["_end"])
                    if cur is None or e["_filed"] > cur[1]:
                        discrete[e["_end"]] = (val, e["_filed"])
                prev_end, prev_val = e["_end"], float(e["val"])
        return discrete

    def quarterly_revenue(self, cik: str) -> Signal:
        facts = self.companyfacts(cik)
        usgaap = facts.get("facts", {}).get("us-gaap", {})
        best: tuple[str, dict[date, tuple[float, date]]] | None = None
        for concept in REVENUE_CONCEPTS:
            node = usgaap.get(concept)
            if not node:
                continue
            usd = node.get("units", {}).get("USD")
            if not usd:
                continue
            disc = self._discrete_quarters(usd)
            # Honor preference order: the first concept with enough history wins
            # outright; only fall back to a less-preferred concept if none has enough.
            if len(disc) >= 8:
                best = (concept, disc)
                break
            if best is None or len(disc) > len(best[1]):
                best = (concept, disc)
        if best is None or not best[1]:
            raise RuntimeError(f"no quarterly revenue concept found for CIK {cik}")

        concept, disc = best
        obs = [
            Observation(ts=end, value=val, as_of=filed)
            for end, (val, filed) in sorted(disc.items())
        ]
        return Signal(
            entity_key=cik,
            source=self.source,
            metric="revenue",
            freq="Q",
            unit="USD",
            observations=obs,
            meta={"concept": concept, "cik": cik, "entity_name": facts.get("entityName")},
        )

    # --------------------------------------------------------------- fetch()
    def fetch(self, *, query: str | None = None, cik: str | None = None, metric: str = "revenue", **_):
        if cik is None:
            if not query:
                raise ValueError("edgar.fetch needs `query` (ticker/name) or `cik`")
            r = self.resolve(query)
            if not r:
                raise RuntimeError(f"could not resolve {query!r} to a SEC filer")
            cik = r["cik"]
        if metric != "revenue":
            raise ValueError(f"edgar connector currently exposes metric 'revenue', not {metric!r}")
        return [self.quarterly_revenue(cik)]
