"""MCP server tests. Skipped entirely unless the optional `mcp` extra is installed."""

import asyncio
import json
from datetime import date

import pytest

pytest.importorskip("mcp", reason="install the 'mcp' extra to test the MCP server")

from altsignal import mcp_server as m  # noqa: E402
from altsignal.models import DriverContribution, ForecastResult, LagStat  # noqa: E402

EXPECTED_TOOLS = {
    "list_sources", "resolve_company", "get_signal",
    "forecast", "triangulate", "screen", "multifactor",
}


def test_all_tools_registered_with_schemas():
    tools = asyncio.run(m.mcp.list_tools())
    names = {t.name for t in tools}
    assert EXPECTED_TOOLS <= names
    forecast = next(t for t in tools if t.name == "forecast")
    assert "query" in forecast.inputSchema["properties"]


def test_jsonable_converts_dates_and_nested_dataclasses():
    res = ForecastResult(
        entity_key="WGO", kpi_metric="revenue", kpi_source="edgar",
        driver_metric="pageviews", driver_source="wikipedia", driver_label="Wikipedia",
        target_period=date(2026, 6, 30),
        lag_table=[LagStat(lag=4, r=0.5, p_value=0.1, n=8, skill=0.2)],
    )
    out = m._jsonable(res)
    # dates become ISO strings, nested dataclasses become dicts, and it's JSON-safe
    assert out["target_period"] == "2026-06-30"
    assert out["lag_table"][0]["lag"] == 4
    assert json.loads(json.dumps(out))["driver_source"] == "wikipedia"


def test_jsonable_handles_lists_of_dataclasses():
    rows = [DriverContribution(label="GDELT", source="gdelt", target_period=date(2026, 3, 31))]
    out = m._jsonable(rows)
    assert out[0]["target_period"] == "2026-03-31"


def test_list_sources_is_serializable():
    srcs = m.list_sources()
    assert {"source", "title", "free", "available"} <= set(srcs[0])
    json.dumps(srcs)  # must not raise
