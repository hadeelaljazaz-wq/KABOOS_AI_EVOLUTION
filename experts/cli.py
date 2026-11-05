"""CLI for the gold expert desk.

    python3 -m experts run SNAPSHOT_DIR --cycle-id C [--ref-time T]
            [--run-mode deterministic|llm] [--store PATH] [--memory PATH] [--force]
    python3 -m experts show --cycle-id C --store PATH
    python3 -m experts list --store PATH

Exit codes: 0 ok · 2 usage · 3 load/import error · 4 nothing usable produced.
Historical replay: pass --ref-time explicitly; an unknown timezone or a missing
timeframe blocks only the claims that need them, never the whole run.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

from gold_data.snapshot import IntegrityError, load_snapshot

from .desk import ExpertDesk

EXIT_OK, EXIT_USAGE, EXIT_IMPORT, EXIT_EMPTY = 0, 2, 3, 4


def _print_results(results) -> None:
    for r in results:
        print("\n" + "─" * 70)
        print(r.summary_ar())


def _cmd_run(args) -> int:
    try:
        snap = load_snapshot(args.snapshot_dir)
    except (IntegrityError, FileNotFoundError, OSError) as exc:
        print(f"[load error] {exc}", file=sys.stderr)
        return EXIT_IMPORT
    if args.run_mode not in ("deterministic", "llm"):
        print("[usage] --run-mode must be 'deterministic' or 'llm'", file=sys.stderr)
        return EXIT_USAGE

    desk = ExpertDesk(args.store, memory_path=args.memory)
    try:
        results, inserted = desk.run_cycle(
            snap, cycle_id=args.cycle_id, reference_time=args.ref_time,
            run_mode=args.run_mode, force=args.force)
        print(f"الدورة: {args.cycle_id} | اللقطة: {snap.snapshot_id} "
              f"({snap.symbol_canonical}/{snap.instrument_kind.value}) | "
              f"{'جديدة' if inserted else 'موجودة مسبقًا — عُرضت دون تكرار'}")
        print(f"المصدر: {', '.join(snap.source_ids)} | المنطقة الزمنية: "
              f"{snap.timezone or 'UNKNOWN'} (known={snap.tz_known}) | المرجع: "
              f"{args.ref_time or snap.analysis_reference_time}")
        print(f"الفريمات المتاحة: {snap.available_timeframes} | الناقصة: {snap.missing_timeframes}")
        _print_results(results)
        ok = any(r.success for r in results)
        return EXIT_OK if ok else EXIT_EMPTY
    finally:
        desk.close()


def _cmd_show(args) -> int:
    desk = ExpertDesk(args.store)
    try:
        data = desk.load_cycle(args.cycle_id)
        if data is None:
            print(f"[not found] no cycle {args.cycle_id!r} in {args.store}", file=sys.stderr)
            return EXIT_IMPORT
        print(f"الدورة: {data['cycle_id']} | اللقطة: {data['snapshot_id']} | "
              f"المرجع: {data['reference_time']} | وضع: {data['run_mode']} | حُفظت: {data['created_at']}")
        for r in desk.load_cycle_results(args.cycle_id):
            print("\n" + "─" * 70)
            print(r.summary_ar())
        return EXIT_OK
    finally:
        desk.close()


def _cmd_list(args) -> int:
    desk = ExpertDesk(args.store)
    try:
        rows = desk.list_cycles()
        if not rows:
            print("لا توجد دورات محفوظة.")
        for c in rows:
            print(f"{c['cycle_id']} | {c['symbol']} | snap={c['snapshot_id']} | "
                  f"ref={c['reference_time']} | mode={c['run_mode']} | {c['created_at']}")
        return EXIT_OK
    finally:
        desk.close()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="experts", description="KABOOS gold expert desk CLI")
    sub = p.add_subparsers(dest="command", required=True)

    pr = sub.add_parser("run", help="run the three experts on a saved snapshot")
    pr.add_argument("snapshot_dir")
    pr.add_argument("--cycle-id", required=True)
    pr.add_argument("--ref-time", default=None)
    pr.add_argument("--run-mode", default="deterministic")
    pr.add_argument("--store", required=True)
    pr.add_argument("--memory", default=None)
    pr.add_argument("--force", action="store_true", help="re-run even if the cycle exists")
    pr.set_defaults(func=_cmd_run)

    ps = sub.add_parser("show", help="print a saved cycle")
    ps.add_argument("--cycle-id", required=True)
    ps.add_argument("--store", required=True)
    ps.set_defaults(func=_cmd_show)

    pl = sub.add_parser("list", help="list saved cycles")
    pl.add_argument("--store", required=True)
    pl.set_defaults(func=_cmd_list)
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
