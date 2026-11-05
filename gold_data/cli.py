"""Command-line entry point for the gold data foundation.

    python3 -m gold_data import FILE [FILE ...] [options]   # build+save a snapshot
    python3 -m gold_data check  FILE [options]              # quality report for one file
    python3 -m gold_data show   SNAPSHOT_DIR                # print a saved snapshot

Exit codes (documented, review item 10):
    0  success (and, when acceptance conditions were asked for, they were met)
    2  usage / bad argument (invalid timezone, timeframe, or reference time)
    3  import / unsupported-format / load / integrity error
    4  acceptance failure (all rows rejected, strict requirements not met, or,
       in `check`, data is stale against a tolerance the user supplied)

Descriptive vs strict: with no acceptance flags, a command DESCRIBES the data
(exit 0 even if imperfect). The moment the user supplies an acceptance
condition (`--require-*`, `--min-bars`, `--max-staleness`, or a
`--stale-tolerance` in `check`), failing it returns a non-zero code. Saving a
snapshot is clearly labelled and never presented as acceptance success. Known
errors are reported as one-line messages, never an unhandled traceback.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import sys
from pathlib import Path
from typing import Optional

from .mt4_csv import UnsupportedFormatError, read_mt4_csv
from .snapshot import (
    ConsumerRequirements,
    IntegrityError,
    MarketSnapshot,
    load_snapshot,
)
from .timeutil import TimezoneError

EXIT_OK, EXIT_USAGE, EXIT_IMPORT, EXIT_QUALITY = 0, 2, 3, 4


class _Usage(Exception):
    pass


def _fmt_tf(d) -> str:
    q = d.quality
    rej = q.get("rejected_counts") or {}
    rej_s = ", ".join(f"{k}={v}" for k, v in rej.items()) or "none"
    cl = d.closed_last_bar
    closed = f"{cl.get('is_closed')} ({cl.get('status')})"
    stale = q.get("staleness", {})
    consistent = "OK" if d.timeframe_consistent else "CONFLICTS DATA"
    forming = f"  forming-bar-excluded={q.get('forming_excluded_count', 0)}"
    return (
        f"  {d.timeframe:<4} bars={d.bar_count:<7} tf={consistent}  "
        f"span={d.first_bar} .. {d.last_bar}\n"
        f"       last-closed={closed}  rejected=[{rej_s}]  "
        f"gaps={q.get('gap_count')} (unresolved={q.get('unresolved_gap_count')})  "
        f"stale={stale.get('is_stale')} ({stale.get('status')}){forming}"
    )


def format_snapshot_summary(snap: MarketSnapshot, validity=None) -> str:
    lines = [f"Snapshot {snap.snapshot_id}  [{snap.schema_version}]"]
    lines.append(f"Instrument : {snap.symbol_canonical} ({snap.instrument_kind.value})  "
                 f"broker symbols: {', '.join(snap.symbol_originals)}")
    spot = snap.instrument_kind.value == "spot"
    lines.append(f"             {'OTC spot/CFD — NOT exchange futures (GC=F)' if spot else snap.instrument_kind.value}")
    lines.append(f"Source id  : {', '.join(snap.source_ids)}")
    lines.append(f"Timezone   : {snap.timezone or 'UNKNOWN'} (known={snap.tz_known})")
    lines.append(f"Created    : {snap.created_at}")
    lines.append(f"Analysis ref time: {snap.analysis_reference_time or 'not set'}")
    lines.append(f"Available timeframes: {snap.available_timeframes}")
    lines.append(f"Missing timeframes  : {snap.missing_timeframes} (not fabricated)")
    lines.append("Per-timeframe (analysis-safe = closed bars only):")
    for tf in snap.available_timeframes:
        lines.append(_fmt_tf(snap.timeframes[tf]))
    lines.append("NOTE: 'last-closed' is the last CLOSED candle at the reference time; a "
                 "still-forming bar is excluded from analysis data and is NOT 'the current price'.")
    if snap.warnings:
        lines.append("Warnings:")
        for w in snap.warnings[:40]:
            lines.append(f"  - {w}")
    if validity is not None:
        lines.append(f"Validity (vs requirements): {'VALID' if validity.valid else 'NOT VALID'}")
        for r in validity.reasons:
            lines.append(f"  - {r}")
    return "\n".join(lines)


def _parse_ref_time(s: Optional[str]) -> Optional[_dt.datetime]:
    if not s:
        return None
    try:
        return _dt.datetime.fromisoformat(s)
    except ValueError:
        raise _Usage(f"--ref-time must be ISO-8601 (e.g. 2026-11-05T05:00 or "
                     f"2026-11-05T05:00+00:00), got {s!r}")


def _tolerance_minutes(value: Optional[float], flag: str) -> Optional[float]:
    """Validate a staleness tolerance in minutes. None = not supplied.

    0 is accepted (zero slack); negative, NaN and Infinity are rejected. The
    check is `is not None`, so a supplied 0 is never dropped by truthiness.
    """
    if value is None:
        return None
    import math
    if not math.isfinite(value) or value < 0:
        raise _Usage(f"{flag} must be a finite, non-negative number of minutes, got {value!r}")
    return value


def _requirements_from_args(args) -> Optional[ConsumerRequirements]:
    max_stale = _tolerance_minutes(args.max_staleness, "--max-staleness")
    # `is not None` so an explicit 0 (min-bars or staleness) is honoured, not dropped
    asked = (args.require_timeframes is not None or (args.min_bars is not None)
             or args.require_tz_known or args.require_closed or max_stale is not None)
    if not asked:
        return None
    try:
        return ConsumerRequirements(
            required_timeframes=tuple(args.require_timeframes or ()),
            min_bars_per_tf=args.min_bars if args.min_bars is not None else 0,
            require_tz_known=bool(args.require_tz_known),
            require_closed_last_bar=bool(args.require_closed),
            max_staleness=_dt.timedelta(minutes=max_stale) if max_stale is not None else None,
        )
    except ValueError as exc:
        raise _Usage(f"invalid requirements: {exc}")


def _cmd_import(args) -> int:
    files = [Path(f) for f in args.files]
    if args.timeframe and len(files) > 1:
        raise _Usage("--timeframe can only be used with a single file; with several "
                     "files the timeframe is resolved per file")
    ref = _parse_ref_time(args.ref_time)
    imports = [read_mt4_csv(f, symbol=args.symbol, timeframe=args.timeframe,
                            timezone=args.timezone, source_id=args.source_id) for f in files]
    snap = MarketSnapshot.build(
        imports, analysis_reference_time=ref,
        stale_tolerance=_dt.timedelta(minutes=args.stale_tolerance) if args.stale_tolerance else None,
        allow_mixed_sources=args.allow_mixed_sources)

    req = _requirements_from_args(args)
    validity = snap.validity(req) if req else None
    print(format_snapshot_summary(snap, validity))

    empty = all(d.bar_count == 0 for d in snap.timeframes.values())
    accepted = validity is None or validity.valid

    if args.out:
        out = snap.save(args.out)
        label = "VALID & accepted" if (validity and validity.valid) else \
                ("DESCRIPTIVE (no acceptance conditions asked)" if validity is None
                 else "DESCRIPTIVE / NOT VALID for the asked requirements")
        print(f"\nSaved snapshot to: {out}  [{label}]")

    if empty:
        print("[accept] every timeframe has zero accepted rows", file=sys.stderr)
        return EXIT_QUALITY
    if not accepted:
        print("\n[accept] snapshot does not meet the asserted requirements", file=sys.stderr)
        return EXIT_QUALITY
    return EXIT_OK


def _cmd_check(args) -> int:
    ref = _parse_ref_time(args.ref_time)
    imp = read_mt4_csv(Path(args.file), symbol=args.symbol, timeframe=args.timeframe,
                       timezone=args.timezone, source_id=args.source_id)
    from .quality import run_quality_checks
    tol_min = _tolerance_minutes(args.stale_tolerance, "--stale-tolerance")
    tol = _dt.timedelta(minutes=tol_min) if tol_min is not None else None
    rep = run_quality_checks(imp, reference_time=ref, stale_tolerance=tol, cutoff_time=ref)
    print(f"File     : {imp.source_name}")
    print(f"Symbol   : {imp.symbol.canonical} ({imp.symbol.kind.value})  original={imp.symbol.original}")
    print(f"Source id: {imp.source_id}")
    print(f"Timeframe: {imp.timeframe.timeframe.value} (source={imp.timeframe.source}, "
          f"consistent={imp.timeframe.consistent})")
    print(f"Rows     : total={rep.total_rows} accepted={rep.accepted_rows} rejected={rep.rejected_total} "
          f"forming-excluded={rep.forming_excluded_count}")
    print(f"Rejected by reason: {rep.rejected_counts or 'none'}")
    print(f"Duplicates: identical_collapsed={rep.identical_duplicates_collapsed} "
          f"conflicting_groups={rep.conflicting_duplicate_groups}")
    print(f"Out-of-order input: {rep.was_out_of_order}")
    print(f"Gaps     : total={rep.gap_count} unresolved={rep.unresolved_gap_count}")
    print(f"Span     : {rep.first_bar} .. {rep.last_bar}")
    print(f"Staleness: {rep.staleness}")
    if rep.warnings:
        print("Warnings:")
        for w in rep.warnings[:40]:
            print(f"  - {w}")
    # strict acceptance: a supplied tolerance makes freshness a REQUIREMENT, and
    # a requirement that cannot be proven is NOT met (review item 7).
    if rep.accepted_rows == 0:
        print("[accept] no accepted rows", file=sys.stderr)
        return EXIT_QUALITY
    if tol is not None:
        st = rep.staleness
        if st.get("status") != "evaluated":
            print(f"[accept] freshness cannot be CONFIRMED (staleness status="
                  f"{st.get('status')}: need a reference time and a known timezone)",
                  file=sys.stderr)
            return EXIT_QUALITY
        if st.get("is_stale") is not False:
            print("[accept] data is stale against the supplied --stale-tolerance", file=sys.stderr)
            return EXIT_QUALITY
    return EXIT_OK


def _cmd_show(args) -> int:
    snap = load_snapshot(args.directory)
    print(format_snapshot_summary(snap))
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="gold_data", description="KABOOS gold data foundation CLI")
    sub = p.add_subparsers(dest="command", required=True)

    def add_common(sp):
        sp.add_argument("--symbol", help="override symbol (else taken from filename)")
        sp.add_argument("--timeframe", help="explicit timeframe (M1/M5/M15/M30/H1/H4/D1)")
        sp.add_argument("--timezone", help="IANA tz of the data; omit to mark UNKNOWN")
        sp.add_argument("--source-id", help="declared broker/feed/server id (not inferred)")
        sp.add_argument("--ref-time", help="analysis reference time, ISO (e.g. 2026-11-05T05:00)")
        sp.add_argument("--stale-tolerance", type=float, metavar="MIN",
                        help="staleness tolerance in minutes (in `check`, an acceptance condition)")

    pi = sub.add_parser("import", help="import file(s) -> MarketSnapshot")
    pi.add_argument("files", nargs="+")
    add_common(pi)
    pi.add_argument("--out", help="directory to save the snapshot into")
    pi.add_argument("--allow-mixed-sources", action="store_true",
                    help="explicitly allow combining different source ids")
    pi.add_argument("--require-timeframes", nargs="*", metavar="TF")
    pi.add_argument("--min-bars", type=int)
    pi.add_argument("--require-tz-known", action="store_true")
    pi.add_argument("--require-closed", action="store_true")
    pi.add_argument("--max-staleness", type=float, metavar="MIN")
    pi.set_defaults(func=_cmd_import)

    pc = sub.add_parser("check", help="quality report for a single file")
    pc.add_argument("file")
    add_common(pc)
    pc.set_defaults(func=_cmd_check)

    ps = sub.add_parser("show", help="print a saved snapshot")
    ps.add_argument("directory")
    ps.set_defaults(func=_cmd_show)
    return p


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except _Usage as exc:
        print(f"[usage] {exc}", file=sys.stderr)
        return EXIT_USAGE
    except TimezoneError as exc:
        print(f"[usage] timezone/reference error: {exc}", file=sys.stderr)
        return EXIT_USAGE
    except UnsupportedFormatError as exc:
        print(f"[import error] {exc}", file=sys.stderr)
        return EXIT_IMPORT
    except IntegrityError as exc:
        print(f"[integrity error] {exc}", file=sys.stderr)
        return EXIT_IMPORT
    except (FileNotFoundError, OSError) as exc:
        print(f"[io error] {exc}", file=sys.stderr)
        return EXIT_IMPORT
    except ValueError as exc:
        # build/validity/requirement problems (e.g. mixed instruments)
        print(f"[error] {exc}", file=sys.stderr)
        return EXIT_QUALITY


if __name__ == "__main__":
    sys.exit(main())
