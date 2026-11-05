"""Tests for the gold expert desk (three independent experts).

All candle data here is SYNTHETIC fixture data — not real gold, not market
evidence, not a claim about any method's effectiveness.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import datetime as dt

import pandas as pd
import pytest

from gold_data import read_mt4_csv
from gold_data.contracts import Direction, Recommendation
from gold_data.snapshot import MarketSnapshot
from experts import (
    ExpertDesk,
    MomentumTrendExpert,
    PriceStructureExpert,
    SupplyDemandExpert,
)
from experts import indicators as ind
from experts.base import plan_feasible
from experts.narrative import narrate


def _write_tf(tmp_path, name, closes, step):
    t0 = dt.datetime(2026, 1, 1)
    rows = []
    for i, c in enumerate(closes):
        o = round(c - 0.5, 2)
        hi = round(max(o, c) + (3.0 if i == 41 else 1.0), 2)
        lo = round(min(o, c) - 1.0, 2)
        t = t0 + dt.timedelta(minutes=step * i)
        rows.append(f"{t.strftime('%Y.%m.%d %H:%M')},{o},{hi},{lo},{round(c,2)},100")
    p = tmp_path / name
    p.write_bytes(("\n".join(rows) + "\n").encode())
    return read_mt4_csv(p, symbol="XAUUSD", timezone="Etc/UTC")


def _snapshot(tmp_path, closes_h1, closes_m15=None, ref=dt.datetime(2026, 2, 1)):
    imps = [_write_tf(tmp_path, "XAUUSD.smlH1.csv", closes_h1, 60)]
    imps.append(_write_tf(tmp_path, "XAUUSD.smlM15.csv", closes_m15 or closes_h1, 15))
    return MarketSnapshot.build(imps, analysis_reference_time=ref)


def _gentle_uptrend(n=260):
    import random
    random.seed(3)
    return [2000 + i * 0.5 + random.uniform(-0.6, 0.6) for i in range(n)]


def _flat(n=120):
    return [2000.0 for _ in range(n)]


def _demand_scenario():
    closes = [2000.0] * 41 + [2030.0]
    closes += [2030.0 + (i - 41) * 2 for i in range(42, 55)]
    closes += [closes[54] - (i - 54) * 2.2 for i in range(55, 80)]
    return [max(c, 2000.2) for c in closes]


# --- independence & different tools ---------------------------------------


def test_three_experts_independent_and_use_different_tools(tmp_path):
    snap = _snapshot(tmp_path, _gentle_uptrend())
    desk = ExpertDesk(tmp_path / "desk.db")
    try:
        results, inserted = desk.run_cycle(snap, "c1", run_mode="deterministic")
        assert inserted and len(results) == 3
        ids = {r.expert_id for r in results}
        assert ids == {"expert_price_structure", "expert_momentum_trend", "expert_supply_demand"}
        by = {r.expert_id: r for r in results}
        # each expert's SCREEN exposes its own tools, not the others'
        assert "structure" in by["expert_price_structure"].screen
        assert "rsi" in by["expert_momentum_trend"].screen
        assert "zones" in by["expert_supply_demand"].screen
        assert "rsi" not in by["expert_price_structure"].screen
        # every report is a valid contract and names its own method version
        vers = {r.method_version for r in results}
        assert len(vers) == 3
        for r in results:
            assert r.report.snapshot_id == snap.snapshot_id
            assert r.report.timeframes_used
            if r.plan is not None:
                assert r.plan.is_usable
    finally:
        desk.close()


def test_first_reports_are_independent_objects(tmp_path):
    snap = _snapshot(tmp_path, _gentle_uptrend())
    desk = ExpertDesk(tmp_path / "desk.db")
    try:
        results, _ = desk.run_cycle(snap, "c1")
        a, b = results[0].report, results[1].report
        assert a.report_id != b.report_id
        a.evidence.append("MUTATION")
        assert "MUTATION" not in b.evidence   # separate objects, no shared state
    finally:
        desk.close()


# --- look-ahead / confirmation timing -------------------------------------


def test_swing_confirmation_excludes_unconfirmed_tail():
    # a strict top at the very last bar cannot be confirmed (no right bars)
    highs = [1, 2, 3, 2, 1, 2, 3, 4, 5]  # max at last index
    df = pd.DataFrame({"open": highs, "high": highs, "low": highs, "close": highs},
                      index=pd.date_range("2026-01-01", periods=len(highs), freq="h"))
    swings = ind.swing_points(df, left=2, right=2)
    last_t = str(df.index[-1])
    assert all(s.time != last_t for s in swings)             # last bar never a confirmed pivot
    for s in swings:
        assert s.confirmed_at <= str(df.index[-1])           # confirmed only by bars in-frame
        assert s.confirmed_at > s.time                        # confirmation is after the pivot


def test_confirmed_swing_appears_only_after_right_bars():
    vals = [1, 2, 5, 2, 1, 0, 1, 2]   # peak (5) at index 2
    df = pd.DataFrame({"open": vals, "high": vals, "low": vals, "close": vals},
                      index=pd.date_range("2026-01-01", periods=len(vals), freq="h"))
    swings = ind.swing_points(df, left=2, right=2)
    peak = [s for s in swings if s.kind == "high" and s.price == 5]
    assert peak and peak[0].confirmed_at == str(df.index[4])  # index 2 + right(2)


# --- insufficient data: wait, never crash ---------------------------------


def test_insufficient_data_waits_without_crash(tmp_path):
    snap = _snapshot(tmp_path, [2000.0, 2001.0, 2002.0, 2001.5, 2002.5], ref=dt.datetime(2026, 1, 2))
    desk = ExpertDesk(tmp_path / "desk.db")
    try:
        results, _ = desk.run_cycle(snap, "c1")
        assert len(results) == 3
        mom = next(r for r in results if r.expert_id == "expert_momentum_trend")
        assert mom.report.recommendation == Recommendation.WAIT
        assert mom.missing  # indicators reported missing, not fabricated
        assert all(r.success for r in results)  # ran cleanly
    finally:
        desk.close()


# --- plans: geometry correct, never invalid -------------------------------


def test_emitted_plan_is_valid_long(tmp_path):
    snap = _snapshot(tmp_path, _demand_scenario(), ref=dt.datetime(2026, 1, 10))
    sd = SupplyDemandExpert()
    r = sd.analyze(snap, "c1", run_mode="deterministic")
    assert r.plan is not None and r.report.recommendation == Recommendation.BUY
    p = r.plan
    assert p.direction == Direction.LONG
    assert p.stop_loss.value < p.entry.low <= p.entry.high < p.targets[0].value
    assert p.is_usable and p.is_structurally_valid


def test_plan_feasible_rules():
    assert plan_feasible(Direction.LONG, 100, 101, 99, [102, 103])
    assert not plan_feasible(Direction.LONG, 100, 101, 100.5, [102])   # stop not below entry
    assert not plan_feasible(Direction.LONG, 100, 101, 99, [100.5])    # target not above entry
    assert not plan_feasible(Direction.LONG, 100, 101, 99, [103, 102]) # targets unsorted
    assert plan_feasible(Direction.SHORT, 100, 101, 102, [99, 98])
    assert not plan_feasible(Direction.SHORT, 100, 101, 99, [98])      # stop not above entry


def test_no_expert_emits_an_invalid_plan(tmp_path):
    # across several scenarios, any emitted plan must validate
    for closes in (_gentle_uptrend(), _flat(), _demand_scenario()):
        snap = _snapshot(tmp_path, closes, ref=dt.datetime(2026, 2, 1))
        for ex in (PriceStructureExpert(), MomentumTrendExpert(), SupplyDemandExpert()):
            r = ex.analyze(snap, "c1", run_mode="deterministic")
            if r.plan is not None:
                assert r.plan.is_usable


# --- justified wait --------------------------------------------------------


def test_flat_market_waits_with_reasons(tmp_path):
    snap = _snapshot(tmp_path, _flat(), ref=dt.datetime(2026, 1, 15))
    desk = ExpertDesk(tmp_path / "desk.db")
    try:
        results, _ = desk.run_cycle(snap, "c1")
        for r in results:
            assert r.report.recommendation == Recommendation.WAIT
            assert r.report.wait_reason and r.report.wait_reason.strip()
            assert r.plan is None
    finally:
        desk.close()


# --- save / restore / idempotency -----------------------------------------


def test_save_restore_after_restart(tmp_path):
    snap = _snapshot(tmp_path, _demand_scenario(), ref=dt.datetime(2026, 1, 10))
    store = tmp_path / "desk.db"
    d1 = ExpertDesk(store, memory_path=tmp_path / "mem.db")
    results, _ = d1.run_cycle(snap, "cycleX", run_mode="deterministic")
    orig = {r.expert_id: (r.report.recommendation, r.plan is not None) for r in results}
    d1.close()

    d2 = ExpertDesk(store, memory_path=tmp_path / "mem.db")   # restart
    recovered = d2.load_cycle_results("cycleX")
    assert {r.expert_id for r in recovered} == set(orig)
    for r in recovered:
        assert (r.report.recommendation, r.plan is not None) == orig[r.expert_id]
    # memory recalled after restart
    mrows = d2.memory.recall("expert_supply_demand", category="analysis:XAUUSD")
    assert len(mrows) >= 1
    d2.close()


def test_cycle_idempotent_no_duplicate(tmp_path):
    snap = _snapshot(tmp_path, _demand_scenario(), ref=dt.datetime(2026, 1, 10))
    store = tmp_path / "desk.db"
    desk = ExpertDesk(store)
    try:
        desk.run_cycle(snap, "cy", run_mode="deterministic")
        results2, inserted2 = desk.run_cycle(snap, "cy", run_mode="deterministic")
        assert inserted2 is False                 # re-running does not duplicate
        rows = desk.store._conn.execute(
            "SELECT COUNT(*) FROM reports WHERE cycle_id=?", ("cy",)).fetchone()[0]
        assert rows == 3
        assert len(results2) == 3                  # still returns the stored three
    finally:
        desk.close()


# --- LLM narrative control -------------------------------------------------


class _FakeLLM:
    def __init__(self, text, is_mock=False, model="fake-real"):
        self._text, self.is_mock, self.model = text, is_mock, model

    def complete(self, system, user, *, max_tokens=1024):
        from types import SimpleNamespace
        return SimpleNamespace(text=self._text, is_mock=self.is_mock, model=self.model)


def test_llm_output_rejected_when_it_invents_numbers(tmp_path):
    snap = _snapshot(tmp_path, _demand_scenario(), ref=dt.datetime(2026, 1, 10))
    sd = SupplyDemandExpert()
    # a backend that injects a fabricated price level not in the evidence
    bad = _FakeLLM("أرى هدفًا جديدًا عند 9999.99 لم يحسبه النظام.")
    r = sd.analyze(snap, "c1", run_mode="llm", llm_backend=bad)
    assert r.run_mode.startswith("deterministic (llm output rejected")
    assert "9999.99" not in r.analysis_ar          # the invented number never reaches the report

    good = _FakeLLM("السوق في اتجاه صاعد مع انحياز شرائي عند منطقة الطلب.")
    r2 = sd.analyze(snap, "c1", run_mode="llm", llm_backend=good)
    assert r2.run_mode.startswith("llm:")


def test_deterministic_mode_is_labelled(tmp_path):
    snap = _snapshot(tmp_path, _gentle_uptrend())
    r = MomentumTrendExpert().analyze(snap, "c1", run_mode="deterministic")
    assert r.run_mode == "deterministic"
    # llm requested but no backend -> clearly labelled fallback, not silent
    r2 = MomentumTrendExpert().analyze(snap, "c1", run_mode="llm", llm_backend=None)
    assert r2.run_mode == "deterministic (llm requested, no key)"
