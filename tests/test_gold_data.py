"""Tests for the gold data foundation (gold_data/).

All CSV data built here is SYNTHETIC and exists only as a test fixture. It is
NOT real gold data and must never be read as evidence about the gold market or
about analysis quality. Real-file import is exercised separately against the
user's uploaded files and reported in PROGRESS.md.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import datetime as dt

import pytest

from gold_data import (
    Direction,
    ExpertReport,
    InstrumentKind,
    Outcome,
    PlanStatus,
    PriceLevel,
    TradePlan,
    Zone,
    UnsupportedFormatError,
    normalize_symbol,
    read_mt4_csv,
)
from gold_data.contracts import DiscussionMessage, MarketState, MessageType, Recommendation
from gold_data.snapshot import ConsumerRequirements, MarketSnapshot, load_snapshot
from gold_data.timeframe import Timeframe, resolve_timeframe


# --- fixture helpers (SYNTHETIC data only) ---------------------------------


def _hourly_rows(start="2026.01.05 00:00", n=60, base=2000.0, step_min=60):
    t0 = dt.datetime.strptime(start, "%Y.%m.%d %H:%M")
    rows = []
    for i in range(n):
        t = t0 + dt.timedelta(minutes=step_min * i)
        o = base + i
        h = o + 2.0
        low = o - 1.5
        c = o + 0.5
        rows.append((t.strftime("%Y.%m.%d %H:%M"), o, h, low, c, 100 + i, 0))
    return rows


def _write(path: Path, rows, *, header=None, delim=",", encoding="utf-8",
           combined_dt=True, with_extra=True):
    """Write SYNTHETIC MT4/MT5-style rows to a CSV fixture file."""
    lines = []
    if header:
        lines.append(delim.join(header))
    for r in rows:
        tstr, o, h, l, c, v, extra = r
        if combined_dt:
            fields = [tstr, o, h, l, c, v]
        else:  # split Date / Time
            d, tm = tstr.split(" ")
            fields = [d, tm, o, h, l, c, v]
        if with_extra:
            fields = fields + [extra]
        lines.append(delim.join(str(x) for x in fields))
    text = "\n".join(lines) + "\n"
    path.write_bytes(text.encode(encoding))
    return path


# --- symbol normalisation --------------------------------------------------


def test_symbol_normalization_rule():
    for raw in ("XAUUSDm", "XAUUSD.sml", "XAUUSD.r", "XAUUSD-ECN", "xauusd"):
        ns = normalize_symbol(raw)
        assert ns.canonical == "XAUUSD"
        assert ns.kind == InstrumentKind.SPOT
        assert ns.original == raw            # original always preserved
        assert ns.known is True
    assert normalize_symbol("XAUUSD.sml").broker_suffix == ".sml"


def test_spot_and_futures_are_distinct_instruments():
    spot = normalize_symbol("XAUUSD")
    fut = normalize_symbol("GC=F")
    assert spot.kind == InstrumentKind.SPOT and fut.kind == InstrumentKind.FUTURES
    assert spot.canonical != fut.canonical
    from gold_data import same_instrument
    assert not same_instrument(spot, fut)


def test_unknown_symbol_flagged_not_mislabelled():
    ns = normalize_symbol("WTFUSD")
    assert ns.known is False
    assert ns.kind == InstrumentKind.UNKNOWN


# --- import: supported formats ---------------------------------------------


def test_read_mt5_combined_datetime_comma_no_header(tmp_path):
    f = _write(tmp_path / "XAUUSD.smlH1.csv", _hourly_rows(), combined_dt=True)
    s = read_mt4_csv(f)
    assert s.symbol.canonical == "XAUUSD"
    assert s.symbol.original == "XAUUSD.sml"       # broker suffix kept
    assert s.timeframe.timeframe == Timeframe.H1
    assert s.raw_row_count == 60
    assert list(s.frame.columns) == ["open", "high", "low", "close", "volume"]
    assert "extra_1" in s.extras.columns          # 7th column preserved, not dropped
    assert s.tz_known is False                     # timezone not assumed


def test_read_mt4_standard_with_header_and_split_datetime(tmp_path):
    f = _write(tmp_path / "gold60.csv", _hourly_rows(),
               header=["Date", "Time", "Open", "High", "Low", "Close", "Volume"],
               combined_dt=False, with_extra=False)
    s = read_mt4_csv(f, symbol="XAUUSD")
    assert s.had_header is True
    assert s.timeframe.timeframe == Timeframe.H1
    assert s.raw_row_count == 60


def test_read_tab_separated_and_utf16(tmp_path):
    f = _write(tmp_path / "XAUUSDmM15.csv", _hourly_rows(step_min=15),
               delim="\t", encoding="utf-16", combined_dt=True)
    s = read_mt4_csv(f)
    assert s.encoding == "utf-16"
    assert s.delimiter == "\t"
    assert s.timeframe.timeframe == Timeframe.M15


def test_unknown_format_is_rejected_with_message(tmp_path):
    bad = tmp_path / "junk.csv"
    bad.write_text("name,qty,price\nwidget,3,9.99\ngadget,1,19.99\n")  # has delimiter, no dates
    with pytest.raises(UnsupportedFormatError) as e:
        read_mt4_csv(bad)
    assert "date" in str(e.value).lower() or "column" in str(e.value).lower()

    nodelim = tmp_path / "junk2.csv"
    nodelim.write_text("hello world\nno delimiter here\n")
    with pytest.raises(UnsupportedFormatError):
        read_mt4_csv(nodelim)


# --- quality checks --------------------------------------------------------


def _import(tmp_path, rows, name="XAUUSD.smlH1.csv", **kw):
    return read_mt4_csv(_write(tmp_path / name, rows, **kw))


def test_invalid_ohlc_is_rejected_not_fixed(tmp_path):
    from gold_data.quality import run_quality_checks
    rows = _hourly_rows(n=20)
    # row 5: High below Open -> inconsistent
    t, o, h, l, c, v, x = rows[5]
    rows[5] = (t, o, o - 5.0, l, c, v, x)   # high < open
    # row 7: negative price
    t, o, h, l, c, v, x = rows[7]
    rows[7] = (t, -1.0, h, l, c, v, x)
    rep = run_quality_checks(_import(tmp_path, rows))
    assert rep.rejected_counts.get("ohlc_inconsistent", 0) >= 1
    assert rep.rejected_counts.get("nonpositive_price", 0) >= 1
    assert rep.accepted_rows == 20 - rep.rejected_total


def test_duplicate_timestamps_identical_vs_conflicting(tmp_path):
    from gold_data.quality import run_quality_checks
    rows = _hourly_rows(n=10)
    identical = rows[3]
    conflicting = (rows[6][0], 1.0, 9.0, 0.5, 2.0, 1, 0)  # same ts, different OHLC
    rows2 = rows + [identical, conflicting]
    rep = run_quality_checks(_import(tmp_path, rows2))
    assert rep.identical_duplicates_collapsed == 1
    assert rep.conflicting_duplicate_groups == 1
    assert rep.rejected_counts.get("conflicting_duplicate", 0) == 2


def test_out_of_order_is_sorted_and_reported(tmp_path):
    from gold_data.quality import run_quality_checks
    rows = _hourly_rows(n=10)
    rows = rows[::-1]  # reversed
    rep = run_quality_checks(_import(tmp_path, rows))
    assert rep.was_out_of_order is True
    assert rep.accepted_frame.index.is_monotonic_increasing


def test_unknown_timezone_blocks_nothing_but_is_flagged(tmp_path):
    s = _import(tmp_path, _hourly_rows(n=10))
    assert s.tz_known is False
    assert any("timezone UNKNOWN" in n for n in s.notes)
    s2 = read_mt4_csv(_write(tmp_path / "XAUUSD.smlH1b.csv", _hourly_rows(n=10)),
                      timezone="Etc/UTC")
    assert s2.tz_known is True


def test_staleness_vs_reference_time(tmp_path):
    from gold_data.quality import run_quality_checks
    s = _import(tmp_path, _hourly_rows(n=10, start="2026.01.05 00:00"))
    # last bar is 2026-01-05 09:00; reference a week later -> stale
    ref = dt.datetime(2026, 1, 12, 0, 0)
    rep = run_quality_checks(s, reference_time=ref, stale_tolerance=dt.timedelta(hours=2))
    # tz is UNKNOWN here, so staleness is provisional (not a confirmed evaluation)
    assert rep.staleness["status"] == "provisional"
    assert rep.staleness["is_stale"] is True
    # with the timezone known it becomes a confirmed evaluation
    s_tz = read_mt4_csv(_write(tmp_path / "XAUUSD.smlH1tz.csv", _hourly_rows(n=10)),
                        timezone="Etc/UTC")
    rep_tz = run_quality_checks(s_tz, reference_time=ref, stale_tolerance=dt.timedelta(hours=2))
    assert rep_tz.staleness["status"] == "evaluated"
    assert rep_tz.staleness["is_stale"] is True
    # a reference right after the last closed bar -> fresh
    rep2 = run_quality_checks(s, reference_time=dt.datetime(2026, 1, 5, 9, 30),
                              stale_tolerance=dt.timedelta(hours=2))
    assert rep2.staleness["is_stale"] is False


def test_lookahead_cutoff_rejects_future_bars(tmp_path):
    from gold_data.quality import run_quality_checks
    s = _import(tmp_path, _hourly_rows(n=20, start="2026.01.05 00:00"))
    cutoff = dt.datetime(2026, 1, 5, 9, 0)   # keep only bars up to 09:00
    rep = run_quality_checks(s, cutoff_time=cutoff)
    assert rep.rejected_counts.get("after_reference_time", 0) == 10
    assert rep.accepted_frame.index.max() <= cutoff


# --- timeframe resolution --------------------------------------------------


def test_timeframe_inferred_and_verified(tmp_path):
    s = _import(tmp_path, _hourly_rows(n=30, step_min=5), name="x.csv")
    assert s.timeframe.timeframe == Timeframe.M5
    assert s.timeframe.source in ("inferred", "filename+verified")


def test_explicit_timeframe_disagreement_is_flagged(tmp_path):
    import pandas as pd
    rows = _hourly_rows(n=20, step_min=60)
    ts = pd.Series([dt.datetime.strptime(r[0], "%Y.%m.%d %H:%M") for r in rows])
    res = resolve_timeframe(ts, explicit="M15")   # data is hourly, claim M15
    assert res.timeframe == Timeframe.M15
    assert res.consistent is False and res.note


# --- snapshot --------------------------------------------------------------


def test_snapshot_available_and_missing_timeframes(tmp_path):
    h1 = _import(tmp_path, _hourly_rows(n=50, step_min=60), name="XAUUSD.smlH1.csv")
    m15 = _import(tmp_path, _hourly_rows(n=50, step_min=15), name="XAUUSD.smlM15.csv")
    snap = MarketSnapshot.build([h1, m15])
    assert set(snap.available_timeframes) == {"H1", "M15"}
    assert "D1" in snap.missing_timeframes and "H4" in snap.missing_timeframes


def test_snapshot_refuses_to_mix_instruments(tmp_path):
    xau = _import(tmp_path, _hourly_rows(n=30), name="XAUUSD.smlH1.csv")
    eur = _import(tmp_path, _hourly_rows(n=30, step_min=15), name="EURUSDM15.csv")
    with pytest.raises(ValueError, match="mix instruments"):
        MarketSnapshot.build([xau, eur])


def test_snapshot_unqualified_validity_is_never_silently_valid(tmp_path):
    h1 = _import(tmp_path, _hourly_rows(n=50), name="XAUUSD.smlH1.csv")
    snap = MarketSnapshot.build([h1])
    assert snap.validity().valid is False          # no requirements => not valid
    req = ConsumerRequirements(required_timeframes=("H1",), min_bars_per_tf=10)
    assert snap.validity(req).valid is True
    req_missing = ConsumerRequirements(required_timeframes=("D1",))
    v = snap.validity(req_missing)
    assert v.valid is False and any("D1" in r for r in v.reasons)


def test_snapshot_save_load_preserves_identity(tmp_path):
    h1 = _import(tmp_path, _hourly_rows(n=50), name="XAUUSD.smlH1.csv")
    m15 = _import(tmp_path, _hourly_rows(n=50, step_min=15), name="XAUUSD.smlM15.csv")
    snap = MarketSnapshot.build([h1, m15], analysis_reference_time=dt.datetime(2026, 2, 1))
    out = snap.save(tmp_path / "snap")
    loaded = load_snapshot(out)
    assert loaded.snapshot_id == snap.snapshot_id
    assert loaded.compute_id() == snap.snapshot_id           # reproducible
    assert loaded.symbol_originals == snap.symbol_originals
    assert loaded.timeframes["H1"].bar_count == snap.timeframes["H1"].bar_count
    assert loaded.timeframes["H1"].source["file_sha256"] == snap.timeframes["H1"].source["file_sha256"]


def test_snapshot_excludes_forming_bar_no_future_leak(tmp_path):
    # bars 00:00..09:00 (open times). At ref 09:30 the 09:00 H1 bar is still
    # forming (closes 10:00) and MUST NOT appear in the analysis-safe set, so
    # its final OHLC can't leak. The last closed bar is then 08:00.
    h1 = _import(tmp_path, _hourly_rows(n=10, start="2026.01.05 00:00"), name="XAUUSD.smlH1.csv")
    snap_open = MarketSnapshot.build([h1], analysis_reference_time=dt.datetime(2026, 1, 5, 9, 30))
    d = snap_open.timeframes["H1"]
    assert str(d.frame.index[-1]) == "2026-01-05 08:00:00"      # 09:00 excluded
    assert d.forming_bar is not None and d.forming_bar["open_time"] == "2026-01-05 09:00:00"
    # once 09:00 has closed (ref 10:00) it becomes available
    snap_closed = MarketSnapshot.build([h1], analysis_reference_time=dt.datetime(2026, 1, 5, 10, 0))
    assert str(snap_closed.timeframes["H1"].frame.index[-1]) == "2026-01-05 09:00:00"


# --- contracts -------------------------------------------------------------


def test_expert_report_wait_requires_reason():
    with pytest.raises(ValueError, match="wait_reason"):
        ExpertReport(expert_id="e1", cycle_id="c1", snapshot_id="s1", specialty="trend",
                     method_version="1.0", direction=Direction.NEUTRAL,
                     market_state=MarketState.RANGE, recommendation=Recommendation.WAIT)
    ok = ExpertReport(expert_id="e1", cycle_id="c1", snapshot_id="s1", specialty="trend",
                      method_version="1.0", direction=Direction.NEUTRAL,
                      market_state=MarketState.RANGE, recommendation=Recommendation.WAIT,
                      wait_reason="no confluence across timeframes")
    assert ok.recommendation == Recommendation.WAIT


def test_expert_report_revision_does_not_overwrite_original():
    orig = ExpertReport(expert_id="e1", cycle_id="c1", snapshot_id="s1", specialty="trend",
                        method_version="1.0", direction=Direction.LONG,
                        market_state=MarketState.TREND_UP, recommendation=Recommendation.BUY)
    revised = orig.revise(direction=Direction.NEUTRAL, recommendation=Recommendation.WAIT,
                          wait_reason="reconsidered after discussion")
    assert revised.report_id != orig.report_id
    assert revised.revises_report_id == orig.report_id
    assert revised.revision == 2
    assert orig.direction == Direction.LONG      # original untouched


def test_expert_report_no_default_uncertainty():
    r = ExpertReport(expert_id="e1", cycle_id="c1", snapshot_id="s1", specialty="trend",
                     method_version="1.0", direction=Direction.LONG,
                     market_state=MarketState.TREND_UP, recommendation=Recommendation.BUY)
    assert r.uncertainty is None                  # never defaulted to a fixed confidence


def test_trade_plan_rejects_bad_ordering_when_actionable():
    entry = Zone(kind="demand", low=2000, high=2005, source="e1:H1")
    # long plan but stop ABOVE entry -> invalid
    with pytest.raises(ValueError, match="stop"):
        TradePlan(owner="e1", owner_kind="expert", cycle_id="c1", snapshot_id="s1",
                  direction=Direction.LONG, entry=entry,
                  stop_loss=PriceLevel(2010, "e1", "stop"),
                  targets=[PriceLevel(2020, "e1", "target")],
                  activation_condition="H1 close above 2005", status=PlanStatus.PENDING)


def test_trade_plan_actionable_requires_core_fields():
    with pytest.raises(ValueError, match="core execution fields"):
        TradePlan(owner="office", owner_kind="office", cycle_id="c1", snapshot_id="s1",
                  direction=Direction.LONG, status=PlanStatus.ACTIVE)  # no entry/stop/targets


def test_wait_without_plan_vs_pending_plan():
    # pending plan MUST have an activation condition (distinguishes it from "wait, no plan")
    with pytest.raises(ValueError, match="activation_condition"):
        TradePlan(owner="e1", owner_kind="expert", cycle_id="c1", snapshot_id="s1",
                  direction=Direction.LONG, entry=Zone("demand", 2000, 2005, "e1"),
                  stop_loss=PriceLevel(1995, "e1", "stop"),
                  targets=[PriceLevel(2020, "e1", "target")], status=PlanStatus.PENDING)
    # a valid pending plan
    pend = TradePlan(owner="e1", owner_kind="expert", cycle_id="c1", snapshot_id="s1",
                     direction=Direction.LONG, entry=Zone("demand", 2000, 2005, "e1"),
                     stop_loss=PriceLevel(1995, "e1", "stop"),
                     targets=[PriceLevel(2020, "e1", "target"), PriceLevel(2030, "e1", "target")],
                     activation_condition="H1 close above 2005", status=PlanStatus.PENDING)
    assert pend.is_actionable
    # "wait, no plan" is simply the absence of a plan + an expert WAIT report
    wait_report = ExpertReport(expert_id="e1", cycle_id="c1", snapshot_id="s1", specialty="x",
                               method_version="1", direction=Direction.NEUTRAL,
                               market_state=MarketState.UNCLEAR, recommendation=Recommendation.WAIT,
                               wait_reason="await breakout", trade_plan_id=None)
    assert wait_report.trade_plan_id is None


def test_report_plan_outcome_linked_by_ids():
    report = ExpertReport(expert_id="e1", cycle_id="c1", snapshot_id="snap1", specialty="trend",
                          method_version="1.0", direction=Direction.SHORT,
                          market_state=MarketState.TREND_DOWN, recommendation=Recommendation.SELL)
    plan = TradePlan(owner="e1", owner_kind="expert", cycle_id="c1", snapshot_id="snap1",
                     direction=Direction.SHORT, supporting_report_ids=[report.report_id],
                     entry=Zone("supply", 2010, 2015, "e1:H1"),
                     stop_loss=PriceLevel(2025, "e1", "stop"),
                     targets=[PriceLevel(2000, "e1", "t1"), PriceLevel(1990, "e1", "t2")],
                     activation_condition="rejection wick on H1", status=PlanStatus.PENDING)
    outcome = Outcome(plan_id=plan.plan_id, plan_version=plan.version, owner=plan.owner,
                      events=[{"time": "2026-01-10T00:00", "type": "activated", "price": 2012}],
                      ambiguity_status="unknown")
    # the three link up by id, and results/owner stay separable (expert vs office)
    assert report.report_id in plan.supporting_report_ids
    assert outcome.plan_id == plan.plan_id and outcome.plan_version == plan.version
    assert outcome.owner == "e1"
    assert outcome.result is None                 # no fabricated result / win-rate


def test_discussion_message_is_contract_only():
    m = DiscussionMessage(cycle_id="c1", sender="e1", message_type=MessageType.OBJECTION,
                          claim="trend is exhausted", related_report_id="report_x",
                          evidence_refs=["snap1:H1:rsi"])
    assert m.message_type == MessageType.OBJECTION
    assert m.message_id.startswith("msg_")


# --- CLI exit codes --------------------------------------------------------


def test_cli_import_success_and_save(tmp_path):
    from gold_data.cli import main
    f = _write(tmp_path / "XAUUSD.smlH1.csv", _hourly_rows(n=60))
    out = tmp_path / "snap_out"
    rc = main(["import", str(f), "--ref-time", "2026-01-10T00:00",
               "--out", str(out), "--require-timeframes", "H1", "--min-bars", "10"])
    assert rc == 0
    assert (out / "meta.json").exists() and (out / "H1.csv").exists()


def test_cli_unsupported_format_exit_3(tmp_path):
    from gold_data.cli import main
    bad = tmp_path / "junk.csv"
    bad.write_text("a,b,c\n1,2,3\n")
    assert main(["check", str(bad)]) == 3


def test_cli_validity_failure_exit_4(tmp_path):
    from gold_data.cli import main
    f = _write(tmp_path / "XAUUSD.smlH1.csv", _hourly_rows(n=60))
    # require a timeframe that isn't present -> validity fails -> exit 4
    rc = main(["import", str(f), "--require-timeframes", "D1"])
    assert rc == 4


def test_cli_show_roundtrip(tmp_path, capsys):
    from gold_data.cli import main
    f = _write(tmp_path / "XAUUSD.smlH1.csv", _hourly_rows(n=60))
    out = tmp_path / "snap_out"
    main(["import", str(f), "--out", str(out)])
    capsys.readouterr()
    assert main(["show", str(out)]) == 0
    printed = capsys.readouterr().out
    assert "XAUUSD" in printed and "NOT 'the current price'" in printed
