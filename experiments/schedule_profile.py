#!/usr/bin/env python3
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from optimizer import SLOT_LIMITS, schedule
from perf_takehome import KernelBuilder


def main():
    forest_height = 10
    rounds = 16
    batch_size = 256

    kb = KernelBuilder()
    kb.build_kernel(forest_height, (2 ** (forest_height + 1)) - 1, batch_size, rounds)
    instrs = kb.instrs

    cycles = len(instrs)
    totals = {engine: 0 for engine in SLOT_LIMITS if engine != "debug"}
    idle = {engine: 0 for engine in totals}
    full = {engine: 0 for engine in totals}
    peak = {engine: 0 for engine in totals}
    overlap = 0
    load_only = 0
    valu_only = 0

    for instr in instrs:
        load_used = len(instr.get("load", []))
        valu_used = len(instr.get("valu", []))
        if load_used > 0 and valu_used > 0:
            overlap += 1
        elif load_used > 0:
            load_only += 1
        elif valu_used > 0:
            valu_only += 1
        for engine, limit in SLOT_LIMITS.items():
            if engine == "debug":
                continue
            used = len(instr.get(engine, []))
            totals[engine] += used
            peak[engine] = max(peak[engine], used)
            if used == 0:
                idle[engine] += 1
            if used == limit:
                full[engine] += 1

    print(f"cycles: {cycles}")
    for engine, limit in SLOT_LIMITS.items():
        if engine == "debug":
            continue
        util = totals[engine] / (cycles * limit) if cycles else 0.0
        idle_pct = (idle[engine] / cycles * 100.0) if cycles else 0.0
        full_pct = (full[engine] / cycles * 100.0) if cycles else 0.0
        print(
            f"{engine}: util={util:.3f} peak={peak[engine]}/{limit} "
            f"idle={idle_pct:.1f}% full={full_pct:.1f}%"
        )
    lbs = {}
    for engine, limit in SLOT_LIMITS.items():
        if engine == "debug":
            continue
        total = totals[engine]
        lbs[engine] = (total + limit - 1) // limit
    if lbs:
        ordered = ", ".join(f"{engine}:{lbs[engine]}" for engine in SLOT_LIMITS if engine in lbs)
        print(f"lower_bounds: {ordered} max: {max(lbs.values())}")
    if cycles:
        print(
            "overlap(load+valu): "
            f"{overlap / cycles:.3f} "
            f"load_only={load_only / cycles:.3f} "
            f"valu_only={valu_only / cycles:.3f}"
        )
    stats = getattr(schedule, "last_stats", {})
    if stats and cycles:
        wait_cycles = stats.get("valu_idle_with_gather", 0)
        underfilled = stats.get("valu_underfilled_cycles", 0)
        ready_total = stats.get("ready_valu_total", 0)
        ready_hist = stats.get("ready_valu_hist", {})
        print(
            "valu_idle_with_gather: "
            f"{wait_cycles / cycles:.3f} "
            f"({wait_cycles}/{cycles})"
        )
        print(
            "valu_underfilled: "
            f"{underfilled / cycles:.3f} "
            f"({underfilled}/{cycles})"
        )
        print(
            "ready_valu_avg: "
            f"{ready_total / cycles:.2f}"
        )
        if ready_hist:
            summary = ", ".join(
                f"{k}:{ready_hist[k]}"
                for k in sorted(ready_hist)[:10]
            )
            print(f"ready_valu_hist: {summary}")


if __name__ == "__main__":
    main()
