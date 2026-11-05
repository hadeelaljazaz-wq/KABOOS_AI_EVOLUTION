"""Command-line entry point for the gold data foundation.

    python3 -m gold_data import FILE [FILE ...] [options]   # build+save a snapshot
    python3 -m gold_data check  FILE [options]              # quality report for one file
    python3 -m gold_data show   SNAPSHOT_DIR                # print a saved snapshot

Exit codes:
    0  success
    2  usage error (bad arguments)
    3  import / unsupported-format error
    4  quality or validity failure (e.g. all rows rejected, or asserted
       requirements not met)

The CLI never calls the last close "the current price", never assumes a
timezone, and never fetches GC=F (or anything else) to stand in for missing
data.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import sys
from pathlib import Path
from typing import Optional

from .mt4_csv import UnsupportedFormatError, read_mt4_csv
from .snapshot import ConsumerRequirements, MarketSnapshot, load_snapshot

EXIT_OK, EXIT_USAGE, EXIT_IMPORT, EXIT_QUALITY = 0, 2, 3, 4


def _fmt_tf(d) -> str:
    q = d.quality
    rej = q.get("rejected_counts") or {}
    rej_s = ", ".join(f"{k}={v}" for k, v in rej.items()) or "none"
    cl = d.closed_last_bar
    closed = f"{cl.get('is_closed')} ({cl.get('status')})"
    stale = q.get("staleness", {})
    stale_s = stale.get("is_stale")
    return (
        f"  {d.timeframe:<4} bars={d.bar_count:<7} "
        f"span={d.first_bar} .. {d.last_bar}\n"
        f"       last-bar closed={closed}  rejected=[{rej_s}]  "
        f"gaps={q.get('gap_count')} (non-weekend={q.get('unexpected_gap_count')})  "
        f"stale={stale_s}"
    )


def format_snapshot_summary(snap: MarketSnapshot,
                            validity: Optional[object] = None) -> str:
    lines = []
    lines.append(f"Snapshot {snap.snapshot_id}  [{snap.schema_version}]")
    lines.append(f"Instrument : {snap.symbol_canonical} ({snap.instrument_kind.value})  "
                 f"broker symbols: {', '.join(snap.symbol_originals)}")
    spot = snap.instrument_kind.value == "spot"
    lines.append(f"             {'OTC spot/CFD — NOT exchange futures (GC=F)' if spot else snap.instrument_kind.value}")
    lines.append(f"Timezone   : {snap.timezone or 'UNKNOWN'} (known={snap.tz_known})")
    lines.append(f"Created    : {snap.created_at}")
    lines.append(f"Analysis ref time: {snap.analysis_reference_time or 'not set'}")
    lines.append(f"Available timeframes: {snap.available_timeframes}")
    lines.append(f"Missing timeframes  : {snap.missing_timeframes} (not fabricated)")
    lines.append("Per-timeframe:")
    for tf in snap.available_timeframes:
        lines.append(_fmt_tf(snap.timeframes[tf]))
    lines.append("NOTE: 'last bar' is the last RECORDED candle's open time; it is NOT "
                 "'the current price'. Closed status depends on the reference time, "
                 "timeframe and (unknown) timezone.")
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
        raise SystemExit(f"[usage] --ref-time must be ISO format (e.g. 2026-11-05T05:00), got {s!r}")


def _requirements_from_args(args) -> Optional[ConsumerRequirements]:
    if not any([args.require_timeframes, args.min_bars, args.require_tz_known,
                args.require_closed, args.max_staleness]):
        return None
    return ConsumerRequirements(
        required_timeframes=tuple(args.require_timeframes or ()),
        min_bars_per_tf=args.min_bars or 0,
        require_tz_known=bool(args.require_tz_known),
        require_closed_last_bar=bool(args.require_closed),
        max_staleness=_dt.timedelta(minutes=args.max_staleness) if args.max_staleness else None,
    )


def _cmd_import(args) -> int:
    files = [Path(f) for f in args.files]
    if args.timeframe and len(files) > 1:
        print("[usage] --timeframe can only be used with a single file; "
              "with several files the timeframe is resolved per file.", file=sys.stderr)
        return EXIT_USAGE
    ref = _parse_ref_time(args.ref_time)
    try:
        imports = [read_mt4_csv(f, symbol=args.symbol, timeframe=args.timeframe,
                                timezone=args.timezone) for f in files]
    except (UnsupportedFormatError, FileNotFoundError, OSError) as exc:
        print(f"[import error] {exc}", file=sys.stderr)
        return EXIT_IMPORT

    try:
        snap = MarketSnapshot.build(
            imports, analysis_reference_time=ref,
            stale_tolerance=_dt.timedelta(minutes=args.stale_tolerance) if args.stale_tolerance else None,
        )
    except ValueError as exc:
        print(f"[build error] {exc}", file=sys.stderr)
        return EXIT_QUALITY

    req = _requirements_from_args(args)
    validity = snap.validity(req) if req else None
    print(format_snapshot_summary(snap, validity))

    if all(d.bar_count == 0 for d in snap.timeframes.values()):
        print("[quality] every timeframe has zero accepted rows", file=sys.stderr)
        return EXIT_QUALITY

    if args.out:
        out = snap.save(args.out)
        print(f"\nSaved snapshot to: {out}")

    if validity is not None and not validity.valid:
        print("\n[validity] snapshot does not meet the asserted requirements", file=sys.stderr)
        return EXIT_QUALITY
    return EXIT_OK


def _cmd_check(args) -> int:
    ref = _parse_ref_time(args.ref_time)
    try:
        imp = read_mt4_csv(Path(args.file), symbol=args.symbol,
                           timeframe=args.timeframe, timezone=args.timezone)
    except (UnsupportedFormatError, FileNotFoundError, OSError) as exc:
        print(f"[import error] {exc}", file=sys.stderr)
        return EXIT_IMPORT
    from .quality import run_quality_checks
    rep = run_quality_checks(
        imp, reference_time=ref,
        stale_tolerance=_dt.timedelta(minutes=args.stale_tolerance) if args.stale_tolerance else None,
        cutoff_time=ref,
    )
    print(f"File     : {imp.source_name}")
    print(f"Symbol   : {imp.symbol.canonical} ({imp.symbol.kind.value})  original={imp.symbol.original}")
    print(f"Timeframe: {imp.timeframe.timeframe.value} (source={imp.timeframe.source}, "
          f"consistent={imp.timeframe.consistent})")
    print(f"Rows     : total={rep.total_rows} accepted={rep.accepted_rows} rejected={rep.rejected_total}")
    print(f"Rejected by reason: {rep.rejected_counts or 'none'}")
    print(f"Duplicates: identical_collapsed={rep.identical_duplicates_collapsed} "
          f"conflicting_groups={rep.conflicting_duplicate_groups}")
    print(f"Out-of-order input: {rep.was_out_of_order}")
    print(f"Gaps     : total={rep.gap_count} non-weekend={rep.unexpected_gap_count}")
    print(f"Span     : {rep.first_bar} .. {rep.last_bar}")
    print(f"Staleness: {rep.staleness}")
    if rep.warnings:
        print("Warnings:")
        for w in rep.warnings[:40]:
            print(f"  - {w}")
    return EXIT_OK if rep.accepted_rows > 0 else EXIT_QUALITY


def _cmd_show(args) -> int:
    try:
        snap = load_snapshot(args.directory)
    except (FileNotFoundError, OSError, KeyError, ValueError) as exc:
        print(f"[load error] {exc}", file=sys.stderr)
        return EXIT_IMPORT
    print(format_snapshot_summary(snap))
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="gold_data", description="KABOOS gold data foundation CLI")
    sub = p.add_subparsers(dest="command", required=True)

    def add_common(sp):
        sp.add_argument("--symbol", help="override symbol (else taken from filename)")
        sp.add_argument("--timeframe", help="explicit timeframe (M1/M5/M15/M30/H1/H4/D1)")
        sp.add_argument("--timezone", help="IANA tz of the data; omit to mark UNKNOWN")
        sp.add_argument("--ref-time", help="analysis reference time, ISO (e.g. 2026-11-05T05:00)")
        sp.add_argument("--stale-tolerance", type=float, metavar="MIN",
                        help="staleness tolerance in minutes")

    pi = sub.add_parser("import", help="import file(s) -> MarketSnapshot")
    pi.add_argument("files", nargs="+")
    add_common(pi)
    pi.add_argument("--out", help="directory to save the snapshot into")
    pi.add_argument("--require-timeframes", nargs="*", metavar="TF",
                    help="assert these timeframes are present")
    pi.add_argument("--min-bars", type=int, help="assert at least this many bars per TF")
    pi.add_argument("--require-tz-known", action="store_true")
    pi.add_argument("--require-closed", action="store_true")
    pi.add_argument("--max-staleness", type=float, metavar="MIN",
                    help="assert data is no older than this many minutes")
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
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
