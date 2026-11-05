"""Focused tests for the remaining data-gate closures (round-3 step 2).

SYNTHETIC fixture data only — not real gold, not market evidence.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import datetime as dt
import json

import pytest

from gold_data import (
    ConsumerRequirements,
    Direction,
    IntegrityError,
    MarketSnapshot,
    PlanStatus,
    PriceLevel,
    TradePlan,
    Zone,
    load_snapshot,
    read_mt4_csv,
)
from gold_data.quality import run_quality_checks


def _mk(p, rows):
    p.write_bytes(("\n".join(",".join(str(x) for x in r) for r in rows) + "\n").encode())
    return p


def _hourly(n=10, start="2026.01.05 00:00", step=60, b=2000.0):
    t0 = dt.datetime.strptime(start, "%Y.%m.%d %H:%M")
    return [((t0 + dt.timedelta(minutes=step * i)).strftime("%Y.%m.%d %H:%M"),
             b + i, b + i + 2, b + i - 1, b + i + 0.5, 100 + i) for i in range(n)]


def _imp(tmp_path, **kw):
    return read_mt4_csv(_mk(tmp_path / "XAUUSD.smlH1.csv", _hourly(10, start="2026.01.05 00:00")),
                        symbol="XAUUSD", **kw)


def test_zero_staleness_tolerance_boundary(tmp_path):
    # last bar opens 09:00, closes 10:00. With tolerance 0 and ref == close,
    # age == 0 -> NOT stale (fresh at the boundary); ref one minute later -> stale.
    imp = _imp(tmp_path, timezone="Etc/UTC")
    at_close = run_quality_checks(imp, reference_time=dt.datetime(2026, 1, 5, 10, 0),
                                  stale_tolerance=dt.timedelta(0))
    assert at_close.staleness["tolerance_minutes"] == 0
    assert at_close.staleness["is_stale"] is False
    later = run_quality_checks(imp, reference_time=dt.datetime(2026, 1, 5, 10, 1),
                               stale_tolerance=dt.timedelta(0))
    assert later.staleness["is_stale"] is True


def test_closed_status_recomputed_on_load_not_trusted(tmp_path):
    imp = _imp(tmp_path, timezone="Etc/UTC")
    snap = MarketSnapshot.build([imp], analysis_reference_time=dt.datetime(2026, 1, 5, 13, 0))
    assert snap.timeframes["H1"].closed_last_bar["is_closed"] is True
    d = snap.save(tmp_path / "s")
    # tamper the stored closed flag (NOT part of the id) to a lie
    meta = json.loads((d / "meta.json").read_text())
    meta["timeframes"]["H1"]["closed_last_bar"]["is_closed"] = False
    (d / "meta.json").write_text(json.dumps(meta))
    loaded = load_snapshot(d)
    # load recomputes from content+ref, so the lie is ignored
    assert loaded.timeframes["H1"].closed_last_bar["is_closed"] is True


def test_staleness_recomputed_on_load(tmp_path):
    imp = _imp(tmp_path, timezone="Etc/UTC")
    snap = MarketSnapshot.build([imp], analysis_reference_time=dt.datetime(2026, 1, 5, 11, 0),
                                stale_tolerance=dt.timedelta(minutes=120))
    d = snap.save(tmp_path / "s")
    meta = json.loads((d / "meta.json").read_text())
    meta["timeframes"]["H1"]["quality"]["staleness"]["age_minutes"] = -99999
    (d / "meta.json").write_text(json.dumps(meta))
    loaded = load_snapshot(d)
    # recomputed: last close 10:00, ref 11:00 -> age 60 min
    assert loaded.timeframes["H1"].quality["staleness"]["age_minutes"] == 60.0


def test_source_id_inconsistency_rejected(tmp_path):
    imp = _imp(tmp_path, timezone="Etc/UTC", source_id="A")
    snap = MarketSnapshot.build([imp], analysis_reference_time=dt.datetime(2026, 1, 5, 13, 0))
    d = snap.save(tmp_path / "s")
    meta = json.loads((d / "meta.json").read_text())
    meta["timeframes"]["H1"]["source"]["source_id"] = "B"  # not in snapshot source_ids
    (d / "meta.json").write_text(json.dumps(meta))
    with pytest.raises(IntegrityError, match="source_id"):
        load_snapshot(d)


def test_old_schema_rejected_with_message(tmp_path):
    imp = _imp(tmp_path, timezone="Etc/UTC")
    snap = MarketSnapshot.build([imp], analysis_reference_time=dt.datetime(2026, 1, 5, 13, 0))
    d = snap.save(tmp_path / "s")
    meta = json.loads((d / "meta.json").read_text())
    meta["schema_version"] = "gold_snapshot/2"
    (d / "meta.json").write_text(json.dumps(meta))
    with pytest.raises(IntegrityError, match="schema_version"):
        load_snapshot(d)


def test_plan_usage_separates_structure_from_expiry():
    plan = TradePlan(owner="e", owner_kind="expert", cycle_id="c", snapshot_id="s",
                     direction=Direction.LONG, entry=Zone("demand", 2000, 2005, "e"),
                     stop_loss=PriceLevel(1995, "e", "stop"),
                     targets=[PriceLevel(2020, "e", "t")],
                     activation_condition="h1 close>2005", status=PlanStatus.PENDING,
                     expiry="2026-01-06T00:00:00")
    assert plan.is_structurally_valid is True
    assert plan.is_usable is True                       # structure + actionable status
    assert plan.usable_at("2026-01-05T12:00:00") is True
    assert plan.usable_at("2026-01-07T00:00:00") is False   # expired
