"""Regression tests for the round-2 reviewed defects in gold_data/.

Each test exercises a PUBLIC consumer path (validity gate, plan validation,
save/load, CLI), not just a helper. All CSV data is SYNTHETIC fixture data,
clearly not real gold and not market evidence.
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
    UnsupportedFormatError,
    Zone,
    load_snapshot,
    read_mt4_csv,
)
from gold_data.cli import main as cli_main
from gold_data.quality import run_quality_checks
from gold_data.timeutil import TimezoneError


def _mk(path, rows, enc="utf-8"):
    path.write_bytes(("\n".join(",".join(str(x) for x in r) for r in rows) + "\n").encode(enc))
    return path


def _hourly(n=10, start="2026.01.05 00:00", step=60, b=2000.0, extra=None):
    t0 = dt.datetime.strptime(start, "%Y.%m.%d %H:%M")
    out = []
    for i in range(n):
        r = [(t0 + dt.timedelta(minutes=step * i)).strftime("%Y.%m.%d %H:%M"),
             b + i, b + i + 2, b + i - 1, b + i + 0.5, 100 + i]
        if extra is not None:
            r.append(extra if not isinstance(extra, list) else extra[i])
        out.append(tuple(r))
    return out


def _imp(tmp_path, rows, name="XAUUSD.smlH1.csv", **kw):
    return read_mt4_csv(_mk(tmp_path / name, rows), **kw)


def _snap(tmp_path, n=10, ref=dt.datetime(2026, 1, 5, 13, 0), **kw):
    imp = _imp(tmp_path, _hourly(n), symbol="XAUUSD", timezone="Etc/UTC", **kw)
    return MarketSnapshot.build([imp], analysis_reference_time=ref)


# ---- R2-1: integrity gates validity --------------------------------------


def _req():
    return ConsumerRequirements(required_timeframes=("H1",), min_bars_per_tf=1,
                                require_closed_last_bar=True)


def test_r1_tampered_ohlc_makes_snapshot_invalid(tmp_path):
    snap = _snap(tmp_path)
    assert snap.validity(_req()).valid is True          # clean -> valid
    snap.timeframes["H1"].frame.iat[0, 3] = 99999.0      # mutate a close in memory
    v = snap.validity(_req())
    assert v.valid is False and any("integrity" in r for r in v.reasons)


def test_r1_tampered_extras_make_snapshot_invalid(tmp_path):
    imp = _imp(tmp_path, _hourly(6, extra="7"), symbol="XAUUSD", timeframe="H1", timezone="Etc/UTC")
    snap = MarketSnapshot.build([imp], analysis_reference_time=dt.datetime(2026, 1, 6))
    req = ConsumerRequirements(required_timeframes=("H1",), min_bars_per_tf=1)
    assert snap.validity(req).valid is True
    snap.timeframes["H1"].extras.iat[0, 0] = "CHANGED"
    assert snap.validity(req).valid is False


# ---- R2-2: plan re-validation on current nested values -------------------


def _pending_plan():
    return TradePlan(owner="e", owner_kind="expert", cycle_id="c", snapshot_id="s",
                     direction=Direction.LONG, entry=Zone("demand", 2000, 2005, "e"),
                     stop_loss=PriceLevel(1995, "e", "stop"),
                     targets=[PriceLevel(2020, "e", "t")],
                     activation_condition="h1 close>2005", status=PlanStatus.PENDING)


def test_r2_mutated_stop_rejected_on_revalidate():
    p = _pending_plan()
    p.stop_loss.value = -1                      # mutate a pre-built nested object
    with pytest.raises(ValueError):
        p.validate()
    assert p.is_usable is False                 # usage gate refuses it
    with pytest.raises(ValueError):
        p.updated()                             # updated() re-validates too


def test_r2_mutated_entry_and_target_rejected():
    p = _pending_plan()
    p.entry.low, p.entry.high = 2100, 2000      # now low > high
    with pytest.raises(ValueError):
        p.validate()
    p2 = _pending_plan()
    p2.targets[0].value = float("nan")
    assert p2.is_usable is False


def test_r2_is_actionable_is_status_only_not_validity():
    p = _pending_plan()
    p.stop_loss.value = -5
    assert p.is_actionable is True              # status is still pending
    assert p.is_usable is False                 # but it is NOT usable


def test_r2_external_reference_cannot_mutate_plan():
    z = Zone("demand", 2000, 2005, "e")
    p = TradePlan(owner="e", owner_kind="expert", cycle_id="c", snapshot_id="s",
                  direction=Direction.LONG, entry=z, stop_loss=PriceLevel(1995, "e", "stop"),
                  targets=[PriceLevel(2020, "e", "t")], status=PlanStatus.CANDIDATE)
    z.low = 1.0                                 # mutate the caller's original object
    assert p.entry.low == 2000                  # the plan kept its own copy


# ---- R2-3: DST localization ----------------------------------------------


def test_r3_spring_forward_nonexistent_hour_rejected(tmp_path):
    # Europe/Berlin 2026-03-29: 02:00->03:00 does not exist
    rows = [("2026.03.29 01:30", 1, 2, 0.5, 1.5, 9),
            ("2026.03.29 02:00", 1, 2, 0.5, 1.5, 9),   # nonexistent
            ("2026.03.29 02:30", 1, 2, 0.5, 1.5, 9),   # nonexistent
            ("2026.03.29 03:30", 1, 2, 0.5, 1.5, 9)]
    imp = read_mt4_csv(_mk(tmp_path / "XAUUSD.smlM30.csv", rows), symbol="XAUUSD",
                       timeframe="M30", timezone="Europe/Berlin")
    rep = run_quality_checks(imp)
    assert rep.rejected_counts.get("dst_unlocalizable_local_time") == 2
    assert not rep.accepted_frame.index.isna().any()
    assert rep.accepted_rows == 2


def test_r3_fall_back_ambiguous_hour_rejected(tmp_path):
    # Europe/Berlin 2026-10-25: 02:00->03:00 occurs twice (ambiguous)
    rows = [("2026.10.25 01:30", 1, 2, 0.5, 1.5, 9),
            ("2026.10.25 02:30", 1, 2, 0.5, 1.5, 9),   # ambiguous
            ("2026.10.25 03:30", 1, 2, 0.5, 1.5, 9)]
    imp = read_mt4_csv(_mk(tmp_path / "XAUUSD.smlM30.csv", rows), symbol="XAUUSD",
                       timeframe="M30", timezone="Europe/Berlin")
    rep = run_quality_checks(imp)
    assert rep.rejected_counts.get("dst_unlocalizable_local_time") == 1
    assert rep.accepted_rows == 2


def test_r3_normal_day_all_accepted(tmp_path):
    imp = read_mt4_csv(_mk(tmp_path / "XAUUSD.smlM30.csv",
                           _hourly(6, start="2026.05.01 10:00", step=30)),
                       symbol="XAUUSD", timeframe="M30", timezone="Europe/Berlin")
    rep = run_quality_checks(imp)
    assert rep.rejected_counts == {} and rep.accepted_rows == 6


def test_r3_utc_ref_equals_source_tz_instant(tmp_path):
    # Moscow local data; a UTC cutoff and the same instant as Moscow-naive cutoff
    # must keep the same bars.
    imp = _imp(tmp_path, _hourly(n=13, start="2026.01.05 00:00"), symbol="XAUUSD",
               timezone="Europe/Moscow")
    a = run_quality_checks(imp, cutoff_time="2026-01-05T06:30:00+00:00")  # 09:30 Moscow
    b = run_quality_checks(imp, cutoff_time="2026-01-05T09:30:00")        # naive -> Moscow
    assert a.accepted_rows == b.accepted_rows
    assert str(a.accepted_frame.index[-1]) == str(b.accepted_frame.index[-1]) == "2026-01-05 08:00:00"


def test_r3_nonexistent_reference_time_clean_error(tmp_path):
    imp = _imp(tmp_path, _hourly(6, start="2026.05.01 10:00"), symbol="XAUUSD",
               timezone="Europe/Berlin")
    with pytest.raises(TimezoneError):
        run_quality_checks(imp, reference_time="2026-03-29T02:30:00")  # nonexistent local
    # and the CLI surfaces it as a usage error, no traceback
    f = _mk(tmp_path / "XAUUSD.smlM30.csv", _hourly(6, start="2026.05.01 10:00", step=30))
    assert cli_main(["check", str(f), "--timezone", "Europe/Berlin",
                     "--ref-time", "2026-03-29T02:30:00"]) == 2


# ---- R2-4: empty snapshots -----------------------------------------------


def test_r4_fully_empty_snapshot_saves_loads_and_is_invalid(tmp_path):
    snap = _snap(tmp_path, ref=dt.datetime(2020, 1, 1))   # everything after ref -> empty
    assert snap.timeframes["H1"].bar_count == 0
    # not valid even when the only requirement is a known timezone
    assert snap.validity(ConsumerRequirements(require_tz_known=True)).valid is False
    d = snap.save(tmp_path / "empty")
    loaded = load_snapshot(d)                               # no IndexError
    assert loaded.timeframes["H1"].bar_count == 0
    assert loaded.validity(ConsumerRequirements(required_timeframes=("H1",))).valid is False


def test_r4_empty_tf_with_valid_required_tf_is_ok(tmp_path):
    h1 = read_mt4_csv(_mk(tmp_path / "XAUUSD.smlH1.csv", _hourly(10, start="2026.01.05 00:00")),
                      symbol="XAUUSD", timezone="Etc/UTC")
    m15 = read_mt4_csv(_mk(tmp_path / "XAUUSD.smlM15.csv", _hourly(10, start="2026.01.05 20:00", step=15)),
                       symbol="XAUUSD", timezone="Etc/UTC")
    snap = MarketSnapshot.build([h1, m15], analysis_reference_time=dt.datetime(2026, 1, 5, 10, 0))
    assert snap.timeframes["M15"].bar_count == 0           # all after ref
    assert snap.timeframes["H1"].bar_count > 0
    # requiring only H1 is valid; the empty, non-required M15 does not block
    assert snap.validity(ConsumerRequirements(required_timeframes=("H1",), min_bars_per_tf=5)).valid is True


def test_r4_cli_empty_import_returns_accept_failure(tmp_path):
    f = _mk(tmp_path / "XAUUSD.smlH1.csv", _hourly(10, start="2026.01.05 00:00"))
    assert cli_main(["import", str(f), "--timezone", "Etc/UTC", "--ref-time", "2020-01-01T00:00"]) == 4


# ---- R2-5: stable content/type round-trip --------------------------------


def test_r5_extras_text_numeric_missing_preserved(tmp_path):
    rows = [("2026.01.05 00:00", 1.23456, 2, 0.5, 1.5, 9, "007"),
            ("2026.01.05 01:00", 1, 2, 0.5, 1.5, 9, "NA"),
            ("2026.01.05 02:00", 1, 2, 0.5, 1.5, 9, "")]
    imp = read_mt4_csv(_mk(tmp_path / "XAUUSD.smlH1.csv", rows), symbol="XAUUSD",
                       timeframe="H1", timezone="Etc/UTC")
    snap = MarketSnapshot.build([imp], analysis_reference_time=dt.datetime(2026, 1, 6))
    d = snap.save(tmp_path / "s1")
    s2 = load_snapshot(d)
    ex = s2.timeframes["H1"].extras["extra_1"].tolist()
    assert ex == ["007", "NA", ""]                         # leading zeros, 'NA', empty all kept as text
    assert float(s2.timeframes["H1"].frame["open"].iloc[0]) == 1.23456   # price precision kept
    # repeated build->save->load->verify->save->load is stable
    s2.verify_integrity()
    d2 = s2.save(tmp_path / "s2")
    s3 = load_snapshot(d2)
    s3.verify_integrity()
    assert s3.snapshot_id == snap.snapshot_id


# ---- R2-6: identity-bearing metadata bound to id -------------------------


def _tamper_meta(d, **changes):
    meta = json.loads((d / "meta.json").read_text())
    meta.update(changes)
    (d / "meta.json").write_text(json.dumps(meta))


def test_r6_editing_source_ids_rejected(tmp_path):
    snap = _snap(tmp_path, source_id="A")
    d = snap.save(tmp_path / "s")
    _tamper_meta(d, source_ids=["EVIL"])
    with pytest.raises(IntegrityError):
        load_snapshot(d)


def test_r6_editing_tz_or_timeframe_facts_rejected(tmp_path):
    snap = _snap(tmp_path)
    d = snap.save(tmp_path / "s")
    # tz_known vs timezone made inconsistent
    _tamper_meta(d, tz_known=False)
    with pytest.raises(IntegrityError):
        load_snapshot(d)


def test_r6_editing_bar_count_or_bounds_rejected(tmp_path):
    snap = _snap(tmp_path)
    d = snap.save(tmp_path / "s")
    meta = json.loads((d / "meta.json").read_text())
    meta["timeframes"]["H1"]["bar_count"] = 999
    (d / "meta.json").write_text(json.dumps(meta))
    with pytest.raises(IntegrityError):
        load_snapshot(d)


# ---- R2-7: CLI strict check ----------------------------------------------


def test_r7_check_tolerance_without_ref_fails(tmp_path):
    f = _mk(tmp_path / "XAUUSD.smlH1.csv", _hourly(20))
    assert cli_main(["check", str(f), "--stale-tolerance", "120"]) == 4


def test_r7_check_provisional_fails(tmp_path):
    f = _mk(tmp_path / "XAUUSD.smlH1.csv", _hourly(20, start="2026.01.05 00:00"))
    # tz unknown -> staleness provisional -> cannot confirm -> fail
    assert cli_main(["check", str(f), "--ref-time", "2026-01-05T20:00", "--stale-tolerance", "120"]) == 4


def test_r7_check_confirmed_fresh_passes(tmp_path):
    f = _mk(tmp_path / "XAUUSD.smlH1.csv", _hourly(20, start="2026.01.05 00:00"))
    # tz known + fresh reference -> evaluated + not stale -> pass
    assert cli_main(["check", str(f), "--timezone", "Etc/UTC",
                     "--ref-time", "2026-01-05T19:30", "--stale-tolerance", "120"]) == 0


def test_r7_check_descriptive_without_tolerance_ok(tmp_path):
    f = _mk(tmp_path / "XAUUSD.smlH1.csv", _hourly(20))
    assert cli_main(["check", str(f)]) == 0


def test_r7_invalid_tolerance_is_usage_error(tmp_path):
    f = _mk(tmp_path / "XAUUSD.smlH1.csv", _hourly(20))
    assert cli_main(["check", str(f), "--stale-tolerance=-5"]) == 2
    assert cli_main(["check", str(f), "--stale-tolerance", "nan"]) == 2


def test_r7_empty_file_and_bad_meta_clean_errors(tmp_path):
    empty = tmp_path / "empty.csv"
    empty.write_text("")
    assert cli_main(["check", str(empty)]) == 3          # import error, no traceback
    # incompatible snapshot metadata on show
    snap = _snap(tmp_path)
    d = snap.save(tmp_path / "s")
    _tamper_meta(d, schema_version="bogus/999")
    assert cli_main(["show", str(d)]) == 3
