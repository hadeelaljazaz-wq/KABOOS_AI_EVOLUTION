"""Regression tests for the 10 reviewed defects in gold_data/.

Each test reproduces a reported wrong behaviour and asserts the corrected one.
All CSV data here is SYNTHETIC fixture data — never real gold, never evidence
about the market. Reducing warning counts is NOT the goal; correct behaviour is.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import datetime as dt

import pytest

from gold_data import (
    ConsumerRequirements,
    Direction,
    ExpertReport,
    IntegrityError,
    MarketSnapshot,
    PriceLevel,
    PlanStatus,
    TradePlan,
    UnsupportedFormatError,
    Zone,
    load_snapshot,
    read_mt4_csv,
)
from gold_data.cli import main as cli_main
from gold_data.contracts import MarketState, Recommendation
from gold_data.quality import run_quality_checks
from gold_data.timeutil import TimezoneError, coerce_reference


def _write(path, rows, *, header=None, delim=",", enc="utf-8", combined=True, extra=False):
    lines = []
    if header:
        lines.append(delim.join(header))
    for r in rows:
        tstr, o, h, l, c, v = r[:6]
        fields = [tstr, o, h, l, c, v] if combined else [*tstr.split(" "), o, h, l, c, v]
        if extra:
            fields = fields + [r[6]]
        lines.append(delim.join(str(x) for x in fields))
    path.write_bytes(("\n".join(lines) + "\n").encode(enc))
    return path


def _hourly(n=10, start="2026.01.05 00:00", step=60, base=2000.0):
    t0 = dt.datetime.strptime(start, "%Y.%m.%d %H:%M")
    return [((t0 + dt.timedelta(minutes=step * i)).strftime("%Y.%m.%d %H:%M"),
             base + i, base + i + 2, base + i - 1, base + i + 0.5, 100 + i) for i in range(n)]


def _imp(tmp_path, rows, name="XAUUSD.smlH1.csv", **kw):
    return read_mt4_csv(_write(tmp_path / name, rows, **{k: v for k, v in kw.items()
                                                         if k in ("header", "delim", "enc", "combined", "extra")}),
                        **{k: v for k, v in kw.items() if k in ("symbol", "timeframe", "timezone", "source_id")})


# ---- Item 1: timezone model ----------------------------------------------


def test_item1_same_instant_in_different_zones_is_equal():
    # 09:30 Moscow (UTC+3) == 06:30 UTC — must coerce to the same instant
    a = coerce_reference("2026-01-05T09:30:00", "Europe/Moscow", True)
    b = coerce_reference("2026-01-05T06:30:00+00:00", "Europe/Moscow", True)
    assert a == b


def test_item1_cutoff_correct_when_source_and_ref_tz_differ(tmp_path):
    # data stamped in Moscow local time; reference given in UTC
    imp = _imp(tmp_path, _hourly(n=13, start="2026.01.05 00:00"), timezone="Europe/Moscow")
    # ref 06:30Z == 09:30 Moscow -> 09:00 Moscow bar (closes 10:00 Moscow=07:00Z) is forming;
    # last closed bar is 08:00 Moscow.
    rep = run_quality_checks(imp, cutoff_time="2026-01-05T06:30:00+00:00")
    assert str(rep.accepted_frame.index[-1]) == "2026-01-05 08:00:00"


def test_item1_bogus_timezone_rejected(tmp_path):
    with pytest.raises(TimezoneError):
        _imp(tmp_path, _hourly(), timezone="not/a_timezone")


def test_item1_tzaware_ref_against_unknown_tz_refused(tmp_path):
    imp = _imp(tmp_path, _hourly())  # tz unknown
    with pytest.raises(TimezoneError):
        run_quality_checks(imp, reference_time="2026-01-05T06:30:00+00:00")


# ---- Item 2: closed bar / future leak ------------------------------------


def test_item2_require_closed_rejects_provisional(tmp_path):
    imp = _imp(tmp_path, _hourly(n=10, start="2026.01.05 00:00"))  # tz UNKNOWN -> provisional
    snap = MarketSnapshot.build([imp], analysis_reference_time=dt.datetime(2026, 1, 5, 13, 0))
    d = snap.timeframes["H1"]
    assert d.closed_last_bar["status"] == "provisional" and d.closed_last_bar["is_closed"] is True
    v = snap.validity(ConsumerRequirements(required_timeframes=("H1",), min_bars_per_tf=1,
                                           require_closed_last_bar=True))
    assert v.valid is False and any("confirmed close" in r or "status" in r for r in v.reasons)
    # with a known tz the confirmed close passes
    imp_tz = _imp(tmp_path, _hourly(n=10), name="XAUUSD.smlH1b.csv", timezone="Etc/UTC")
    snap_tz = MarketSnapshot.build([imp_tz], analysis_reference_time=dt.datetime(2026, 1, 5, 13, 0))
    assert snap_tz.validity(ConsumerRequirements(required_timeframes=("H1",), min_bars_per_tf=1,
                                                 require_closed_last_bar=True)).valid is True


# ---- Item 3: validity ----------------------------------------------------


def test_item3_timeframe_conflict_blocks_validity(tmp_path):
    imp = _imp(tmp_path, _hourly(n=20), timeframe="M15")  # hourly data claimed M15
    snap = MarketSnapshot.build([imp])
    v = snap.validity(ConsumerRequirements(required_timeframes=("M15",), min_bars_per_tf=5))
    assert v.valid is False and any("conflict" in r for r in v.reasons)


def test_item3_zero_bar_timeframe_blocks_even_without_min_bars(tmp_path):
    # cutoff in the far past -> all bars are "after reference" -> zero accepted
    imp = _imp(tmp_path, _hourly(n=10, start="2026.01.05 00:00"), timezone="Etc/UTC")
    snap = MarketSnapshot.build([imp], analysis_reference_time=dt.datetime(2020, 1, 1))
    assert snap.timeframes["H1"].bar_count == 0
    v = snap.validity(ConsumerRequirements(required_timeframes=("H1",)))  # no min_bars
    assert v.valid is False and any("no accepted rows" in r for r in v.reasons)


def test_item3_bad_requirements_values_rejected():
    with pytest.raises(ValueError):
        ConsumerRequirements(min_bars_per_tf=-5)
    with pytest.raises(ValueError):
        ConsumerRequirements(required_timeframes=("H9",))
    with pytest.raises(ValueError):
        ConsumerRequirements(max_staleness=dt.timedelta(seconds=-1))


# ---- Item 4: CSV header mapping ------------------------------------------


def test_item4_header_maps_by_name_no_positional_swap(tmp_path):
    # header puts Close before Open; values must map by NAME, not position
    rows = [("2026.01.05 00:00", 11, 12, 9, 10, 100)]  # fields: dt, A, B, C, D, V
    # declare: col1=Close col2=High col3=Low col4=Open  -> open=10, close=11
    f = _write(tmp_path / "XAUUSD.smlH1.csv",
               [("2026.01.05 00:00", 11, 12, 9, 10, 100),
                ("2026.01.05 01:00", 12, 13, 10, 11, 100),
                ("2026.01.05 02:00", 13, 14, 11, 12, 100)],
               header=["Date Time", "Close", "High", "Low", "Open", "Volume"])
    s = read_mt4_csv(f, symbol="XAUUSD")
    assert float(s.frame["open"].iloc[0]) == 10.0
    assert float(s.frame["close"].iloc[0]) == 11.0


def test_item4_duplicate_and_missing_columns_rejected(tmp_path):
    dup = _write(tmp_path / "dup.csv", [("2026.01.05 00:00", 1, 2, 0.5, 1.5, 9)],
                 header=["Date Time", "Open", "Open", "Low", "Close", "Volume"])
    with pytest.raises(UnsupportedFormatError, match="duplicate"):
        read_mt4_csv(dup, symbol="XAUUSD")
    miss = tmp_path / "miss.csv"  # header has no Low column
    miss.write_text("Date Time,Open,High,Close,Volume\n2026.01.05 00:00,1,2,1.5,9\n")
    with pytest.raises(UnsupportedFormatError, match="missing required"):
        read_mt4_csv(miss, symbol="XAUUSD")


def test_item4_corrupt_first_row_not_swallowed_as_header(tmp_path):
    # first row is corrupt data (not recognised header names, not a date) -> reject,
    # do NOT silently treat it as a header and drop it
    bad = tmp_path / "corrupt.csv"
    bad.write_text("garbage,row,here,boom,x,y\n2026.01.05 00:00,1,2,0.5,1.5,9\n")
    with pytest.raises(UnsupportedFormatError, match="date"):
        read_mt4_csv(bad, symbol="XAUUSD")


def test_item4_supported_formats_still_work(tmp_path):
    a = read_mt4_csv(_write(tmp_path / "XAUUSD.smlH1.csv", _hourly(12)), symbol="XAUUSD")
    b = read_mt4_csv(_write(tmp_path / "g60.csv", _hourly(12), combined=False,
                            header=["Date", "Time", "Open", "High", "Low", "Close", "Volume"]),
                     symbol="XAUUSD")
    assert a.raw_row_count == 12 and b.raw_row_count == 12 and b.had_header


# ---- Item 5: gap classification ------------------------------------------


def test_item5_weekday_gap_not_called_session_break(tmp_path):
    # Monday M1 gap 09:02 -> 11:02 (118 missing bars) must be UNRESOLVED, not a
    # session break, and the missing-bar count must stay visible.
    rows = [("2026.01.05 09:00", 1, 2, 0.5, 1.5, 9),
            ("2026.01.05 09:01", 1, 2, 0.5, 1.5, 9),
            ("2026.01.05 09:02", 1, 2, 0.5, 1.5, 9),
            ("2026.01.05 11:02", 1, 2, 0.5, 1.5, 9),
            ("2026.01.05 11:03", 1, 2, 0.5, 1.5, 9)]
    imp = read_mt4_csv(_write(tmp_path / "XAUUSD.smlM1.csv", rows), symbol="XAUUSD", timeframe="M1")
    rep = run_quality_checks(imp)
    gaps = [g for g in rep.largest_gaps if g["minutes"] == 120.0]
    assert gaps and gaps[0]["category"] == "unresolved"
    assert gaps[0]["missing_bars"] == 119
    assert rep.unresolved_gap_count >= 1


def test_item5_long_gap_with_weekend_not_hidden(tmp_path):
    # a 6-day M1 gap that merely contains a weekend must not be classed as a
    # plain weekend closure (extra trading days are missing).
    rows = [("2026.01.02 09:00", 1, 2, 0.5, 1.5, 9),   # Fri
            ("2026.01.02 09:01", 1, 2, 0.5, 1.5, 9),
            ("2026.01.08 09:00", 1, 2, 0.5, 1.5, 9),   # following Thu
            ("2026.01.08 09:01", 1, 2, 0.5, 1.5, 9)]
    imp = read_mt4_csv(_write(tmp_path / "XAUUSD.smlM1.csv", rows), symbol="XAUUSD", timeframe="M1")
    rep = run_quality_checks(imp)
    big = max(rep.largest_gaps, key=lambda g: g["minutes"])
    assert big["category"] == "unresolved"       # not hidden as weekend_or_closed
    assert big["missing_bars"] > 2000


# ---- Item 6: revise deep copy --------------------------------------------


def test_item6_revise_is_deep_independent():
    orig = ExpertReport(expert_id="e", cycle_id="c", snapshot_id="s", specialty="x",
                        method_version="1", direction=Direction.LONG,
                        market_state=MarketState.TREND_UP, recommendation=Recommendation.BUY,
                        zones=[Zone("demand", 2000, 2005, "e")],
                        levels=[PriceLevel(2010, "e", "target")])
    rev = orig.revise()
    rev.zones[0].low = 1.0
    rev.levels[0].value = 1.0
    rev.zones.append(Zone("supply", 3000, 3001, "e"))
    assert orig.zones[0].low == 2000              # original zone untouched
    assert orig.levels[0].value == 2010           # original level untouched
    assert len(orig.zones) == 1                    # original list untouched
    # and the reverse direction
    orig.zones[0].high = 9999
    assert rev.zones[0].high == 2005
    assert rev.report_id != orig.report_id and rev.revises_report_id == orig.report_id


# ---- Item 7: price / plan contracts --------------------------------------


def test_item7_nonfinite_prices_rejected():
    for bad in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError):
            PriceLevel(bad, "s")
        with pytest.raises(ValueError):
            Zone("support", bad, 10, "s")
    with pytest.raises(ValueError):
        PriceLevel(True, "s")                      # bool rejected


def test_item7_plan_updated_revalidates():
    plan = TradePlan(owner="e", owner_kind="expert", cycle_id="c", snapshot_id="s",
                     direction=Direction.LONG, entry=Zone("demand", 2000, 2005, "e"),
                     stop_loss=PriceLevel(1995, "e", "stop"),
                     targets=[PriceLevel(2020, "e", "t")],
                     activation_condition="h1 close>2005", status=PlanStatus.PENDING)
    # moving the stop above entry via the validated path must fail
    with pytest.raises(ValueError, match="stop"):
        plan.updated(stop_loss=PriceLevel(2010, "e", "stop"))
    good = plan.updated(targets=[PriceLevel(2020, "e", "t"), PriceLevel(2030, "e", "t")])
    assert good.version == plan.version + 1


def test_item7_expiry_validated():
    with pytest.raises(ValueError, match="expiry"):
        TradePlan(owner="e", owner_kind="expert", cycle_id="c", snapshot_id="s",
                  direction=Direction.LONG, status=PlanStatus.CANDIDATE, expiry="not-a-date")


def test_item7_nested_dict_coerced_through_validation():
    # a dict entry with a NaN price must be rejected via the validated path
    with pytest.raises(ValueError):
        TradePlan(owner="e", owner_kind="expert", cycle_id="c", snapshot_id="s",
                  direction=Direction.LONG,
                  entry={"kind": "demand", "low": float("nan"), "high": 2005, "source": "e"},
                  status=PlanStatus.CANDIDATE)


# ---- Item 8: content fingerprint + safe save/load ------------------------


def _build_snap(tmp_path, ref=dt.datetime(2026, 1, 5, 13, 0)):
    imp = _imp(tmp_path, _hourly(n=10), timezone="Etc/UTC")
    return MarketSnapshot.build([imp], analysis_reference_time=ref)


def test_item8_tampered_csv_rejected_on_load(tmp_path):
    snap = _build_snap(tmp_path)
    d = snap.save(tmp_path / "snap")
    lines = (d / "H1.csv").read_text().splitlines()
    lines[1] = lines[1].replace("2000.5", "9000.5")  # change the first bar's close
    assert "9000.5" in lines[1]                        # sanity: the tamper landed
    (d / "H1.csv").write_text("\n".join(lines) + "\n")
    with pytest.raises(IntegrityError, match="content hash"):
        load_snapshot(d)


def test_item8_clean_roundtrip_identity_stable(tmp_path):
    snap = _build_snap(tmp_path)
    d = snap.save(tmp_path / "snap")
    loaded = load_snapshot(d)
    assert loaded.snapshot_id == snap.snapshot_id
    loaded.verify_integrity()  # no raise


def test_item8_in_memory_mutation_detected(tmp_path):
    snap = _build_snap(tmp_path)
    snap.timeframes["H1"].frame.iat[0, 3] = 99999.0  # mutate a close in place
    with pytest.raises(IntegrityError):
        snap.verify_integrity()
    with pytest.raises(IntegrityError):
        snap.save(tmp_path / "snap2")  # save refuses to persist drifted content


def test_item8_path_traversal_and_missing_rejected(tmp_path):
    snap = _build_snap(tmp_path)
    d = snap.save(tmp_path / "snap")
    import json
    meta = json.loads((d / "meta.json").read_text())
    meta["timeframes"]["H1"]["bars_file"] = "../evil.csv"
    (d / "meta.json").write_text(json.dumps(meta))
    with pytest.raises(IntegrityError, match="unsafe|unexpected"):
        load_snapshot(d)
    # partial save (no meta.json) is not accepted as complete
    bare = tmp_path / "bare"
    bare.mkdir()
    (bare / "H1.csv").write_text("time,open,high,low,close,volume\n")
    with pytest.raises(IntegrityError, match="meta.json"):
        load_snapshot(bare)


# ---- Item 9: source identity + extras ------------------------------------


def test_item9_source_id_declared_not_inferred(tmp_path):
    imp = _imp(tmp_path, _hourly())
    assert imp.source_id == "unknown"              # not guessed from '.sml'
    imp2 = _imp(tmp_path, _hourly(), name="XAUUSD.smlH1b.csv", source_id="brokerX:serverEU")
    assert imp2.source_id == "brokerX:serverEU"


def test_item9_mixed_sources_refused_unless_explicit(tmp_path):
    a = _imp(tmp_path, _hourly(step=60), name="XAUUSD.smlH1.csv", source_id="A")
    b = _imp(tmp_path, _hourly(step=15), name="XAUUSD.smlM15.csv", source_id="B")
    with pytest.raises(ValueError, match="mix data sources"):
        MarketSnapshot.build([a, b])
    snap = MarketSnapshot.build([a, b], allow_mixed_sources=True)
    assert snap.source_ids == ["A", "B"]


def test_item9_extras_preserved_through_save_load(tmp_path):
    rows = [("2026.01.05 00:00", 1, 2, 0.5, 1.5, 9, 7),
            ("2026.01.05 01:00", 1, 2, 0.5, 1.5, 9, 8),
            ("2026.01.05 02:00", 1, 2, 0.5, 1.5, 9, 9)]
    imp = read_mt4_csv(_write(tmp_path / "XAUUSD.smlH1.csv", rows, extra=True),
                       symbol="XAUUSD", timeframe="H1", timezone="Etc/UTC")
    assert "extra_1" in imp.extras.columns
    snap = MarketSnapshot.build([imp])
    assert "extra_1" in snap.timeframes["H1"].extras.columns
    d = snap.save(tmp_path / "snap")
    loaded = load_snapshot(d)
    ex = loaded.timeframes["H1"].extras
    # extras are uninterpreted and kept as RAW TEXT (not coerced to numbers),
    # so leading zeros / 'NA' survive a save/load round-trip (review round-2 item 5)
    assert "extra_1" in ex.columns and list(ex["extra_1"]) == ["7", "8", "9"]


def test_item9_extras_difference_makes_duplicate_conflicting(tmp_path):
    rows = [("2026.01.05 00:00", 1, 2, 0.5, 1.5, 9, 7),
            ("2026.01.05 01:00", 1, 2, 0.5, 1.5, 9, 7),
            ("2026.01.05 01:00", 1, 2, 0.5, 1.5, 9, 8)]  # same ts+OHLCV, diff extra
    imp = read_mt4_csv(_write(tmp_path / "XAUUSD.smlH1.csv", rows, extra=True),
                       symbol="XAUUSD", timeframe="H1")
    rep = run_quality_checks(imp)
    assert rep.conflicting_duplicate_groups == 1
    assert rep.identical_duplicates_collapsed == 0


# ---- Item 10: CLI strict vs descriptive + warnings -----------------------


def test_item10_check_strict_staleness_returns_failure(tmp_path):
    f = _write(tmp_path / "XAUUSD.smlH1.csv", _hourly(n=20))
    rc = cli_main(["check", str(f), "--ref-time", "2026-02-01T00:00", "--stale-tolerance", "120"])
    assert rc == 4


def test_item10_check_descriptive_no_tolerance_is_ok(tmp_path):
    f = _write(tmp_path / "XAUUSD.smlH1.csv", _hourly(n=20))
    assert cli_main(["check", str(f), "--ref-time", "2026-02-01T00:00"]) == 0


def test_item10_import_invalid_timezone_is_usage_error(tmp_path):
    f = _write(tmp_path / "XAUUSD.smlH1.csv", _hourly(n=20))
    assert cli_main(["import", str(f), "--timezone", "not/a_zone"]) == 2


def test_item10_import_mixed_instruments_handled_cleanly(tmp_path):
    a = _write(tmp_path / "XAUUSD.smlH1.csv", _hourly(step=60))
    b = _write(tmp_path / "EURUSDM15.csv", _hourly(step=15))
    rc = cli_main(["import", str(a), str(b)])
    assert rc == 4   # ValueError -> clean failure code, no traceback


def test_item10_show_tampered_returns_error(tmp_path):
    imp = _imp(tmp_path, _hourly(n=10), timezone="Etc/UTC")
    snap = MarketSnapshot.build([imp], analysis_reference_time=dt.datetime(2026, 1, 5, 13, 0))
    d = snap.save(tmp_path / "snap")
    (d / "H1.csv").write_text((d / "H1.csv").read_text().replace("2001.5", "9001.5"))
    assert cli_main(["show", str(d)]) == 3
