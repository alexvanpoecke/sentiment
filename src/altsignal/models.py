"""Core data models. Plain dataclasses — no third-party dependency."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any


@dataclass(frozen=True)
class Observation:
    """A single point in a time series."""

    ts: date  # period end / observation date
    value: float
    as_of: date | None = None  # point-in-time vintage (filing/publish date), if known


@dataclass
class Signal:
    """A normalized time series from one source for one entity + metric."""

    entity_key: str  # e.g. "WGO" or "macro:UMCSENT"
    source: str  # connector source id, e.g. "edgar", "google_trends"
    metric: str  # e.g. "revenue", "search_interest", "pageviews"
    freq: str = "Q"  # D / W / M / Q / A
    geo: str | None = None
    unit: str | None = None
    observations: list[Observation] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.observations)

    @property
    def label(self) -> str:
        base = f"{self.source}:{self.metric}"
        return f"{base} [{self.geo}]" if self.geo else base

    def sorted(self) -> list[Observation]:
        return sorted(self.observations, key=lambda o: o.ts)

    def series(self) -> list[tuple[date, float]]:
        return [(o.ts, o.value) for o in self.sorted()]

    def as_dict(self) -> dict[date, float]:
        return {o.ts: o.value for o in self.sorted()}

    def latest(self) -> Observation | None:
        s = self.sorted()
        return s[-1] if s else None


@dataclass
class Entity:
    """A resolved company and everything we infer about which signals matter."""

    query: str
    ticker: str | None = None
    cik: str | None = None  # 10-digit zero-padded
    name: str | None = None
    aliases: list[str] = field(default_factory=list)
    sic: str | None = None
    sic_description: str | None = None
    sector: str | None = None  # our industry-bucket label
    industry_key: str | None = None
    domains: list[str] = field(default_factory=list)
    brands: list[str] = field(default_factory=list)
    seed_terms: list[str] = field(default_factory=list)
    peers: list[str] = field(default_factory=list)
    macro_series: list[str] = field(default_factory=list)
    subreddits: list[str] = field(default_factory=list)
    connectors: list[str] = field(default_factory=list)
    fiscal_year_end: str | None = None  # "MMDD"
    country: str | None = None

    @property
    def key(self) -> str:
        return self.ticker or self.cik or self.query

    @property
    def short_name(self) -> str:
        """Best-effort brand name: company name minus common corporate suffixes.

        SEC names are ALL-CAPS (e.g. "WINNEBAGO INDUSTRIES, INC."), so matching is
        case- and punctuation-insensitive (the old title-case suffix list never
        matched, leaving "Inc"/"Corporation" on every resolved company).
        """
        forms = {
            "incorporated", "corporation", "company", "inc", "corp", "co",
            "ltd", "llc", "lp", "plc", "holdings", "group", "sa", "ag", "nv",
        }
        words = (self.name or self.query).replace(",", " ").split()
        # Drop trailing state markers like /MN/ or /DE/.
        while words and words[-1].startswith("/") and words[-1].endswith("/"):
            words.pop()
        # Drop trailing corporate-form tokens (case/punctuation-insensitive).
        while words and words[-1].strip(".").lower() in forms:
            words.pop()
        result = " ".join(words)
        return result.title() if result.isupper() else result


@dataclass
class LagStat:
    """Correlation + out-of-sample skill of driver-at-lag vs target, per lag."""

    lag: int
    r: float
    p_value: float
    n: int
    folds: int = 0
    model_mae: float | None = None
    naive_mae: float | None = None
    skill: float | None = None  # (naive_mae - model_mae) / naive_mae; >0 beats naive


@dataclass
class ForecastResult:
    """Output of the signal -> KPI forecast workflow."""

    entity_key: str
    kpi_metric: str
    kpi_source: str
    driver_metric: str
    driver_source: str
    driver_label: str
    freq: str = "Q"

    n_obs: int = 0
    best_lag: int = 0
    corr: float = 0.0
    corr_p: float = 1.0
    slope: float = 0.0
    intercept: float = 0.0
    r2: float = 0.0
    resid_std: float = 0.0

    target_period: date | None = None
    current_driver_yoy: float | None = None
    predicted_yoy: float | None = None
    base_level: float | None = None  # year-ago KPI level used to convert YoY -> level
    predicted_level: float | None = None
    alpha: float = 0.20
    pi_low_yoy: float | None = None
    pi_high_yoy: float | None = None
    pi_low_level: float | None = None
    pi_high_level: float | None = None

    backtest_n: int = 0
    backtest_mae_yoy: float | None = None
    backtest_naive_mae_yoy: float | None = None

    lag_table: list[LagStat] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class DriverContribution:
    """One driver's forecast within a triangulation ensemble."""

    label: str
    source: str
    n: int = 0
    lag: int = 0
    corr: float = 0.0
    skill: float | None = None
    predicted_yoy: float | None = None
    target_period: date | None = None
    weight: float = 0.0  # normalized ensemble weight


@dataclass
class TriangulationResult:
    """Ensemble nowcast blending several independent driver forecasts."""

    entity_key: str
    target_period: date | None = None
    drivers: list[DriverContribution] = field(default_factory=list)
    ensemble_yoy: float | None = None
    base_level: float | None = None
    ensemble_level: float | None = None
    agreement_stdev: float | None = None  # dispersion of driver predictions
    notes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class ScreenRow:
    """One (company, driver) result in a cross-sectional skill screen."""

    ticker: str
    name: str | None = None
    driver: str | None = None
    n: int = 0
    lag: int = 0
    corr: float = 0.0
    skill: float | None = None  # out-of-sample skill vs naive persistence
    predicted_yoy: float | None = None
    target_period: date | None = None
    error: str | None = None


@dataclass
class FactorCoef:
    """One coefficient in a multifactor regression."""

    name: str
    coef: float
    t: float
    p: float


@dataclass
class MultiFactorResult:
    """Forecast from a single regression on several signals (+ optional seasonality)."""

    entity_key: str
    kpi_metric: str = "revenue"
    kpi_source: str = "edgar"
    seasonal: bool = False
    driver_labels: list[str] = field(default_factory=list)
    n_obs: int = 0
    r2: float = 0.0
    features: list[FactorCoef] = field(default_factory=list)

    target_period: date | None = None
    predicted_yoy: float | None = None
    pi_low_yoy: float | None = None
    pi_high_yoy: float | None = None
    base_level: float | None = None
    predicted_level: float | None = None
    pi_low_level: float | None = None
    pi_high_level: float | None = None
    alpha: float = 0.20

    backtest_n: int = 0
    backtest_mae: float | None = None
    backtest_naive_mae: float | None = None
    skill: float | None = None

    notes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
