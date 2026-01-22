#!/usr/bin/env python3
import argparse
import itertools
import math
import os
import sys
import time
from dataclasses import dataclass
from multiprocessing import Pool

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

import optimizer
from perf_takehome import KernelBuilder

OPS = None


@dataclass(frozen=True)
class Result:
    cycles: int
    max_lb: int
    flow_weight: float
    load_weight: float
    max_inflight: int
    seeds: tuple


def build_ops(forest_height: int, batch_size: int, rounds: int):
    kb = KernelBuilder()
    kb.build = lambda slots: slots
    kb.build_kernel(forest_height, (2 ** (forest_height + 1)) - 1, batch_size, rounds)
    return kb.instrs


def init_worker(ops):
    global OPS
    OPS = ops


def measure(config):
    flow_weight, load_weight, max_inflight, seeds = config
    optimizer.WEIGHT_FLOW_MULT = flow_weight
    optimizer.WEIGHT_LOAD_MULT = load_weight
    optimizer.MAX_INFLIGHT_GROUPS = max_inflight
    optimizer.SCHEDULE_SEEDS = seeds
    bundles = optimizer.schedule(OPS)
    cycles = len(bundles)
    totals = {engine: 0 for engine in optimizer.SLOT_LIMITS if engine != "debug"}
    for instr in bundles:
        for engine in totals:
            totals[engine] += len(instr.get(engine, []))
    max_lb = 0
    for engine, limit in optimizer.SLOT_LIMITS.items():
        if engine == "debug":
            continue
        total = totals[engine]
        max_lb = max(max_lb, (total + limit - 1) // limit)
    return Result(
        cycles=cycles,
        max_lb=max_lb,
        flow_weight=flow_weight,
        load_weight=load_weight,
        max_inflight=max_inflight,
        seeds=seeds,
    )


def parse_float_list(text):
    return [float(x) for x in text.split(",") if x.strip()]


def parse_int_list(text):
    return [int(x) for x in text.split(",") if x.strip()]


def make_seeds(count):
    if count <= 0:
        return (None,)
    return (None,) + tuple(range(count))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--flow-weights", default="0.0,0.01,0.02,0.03")
    parser.add_argument("--load-weights", default="0.02,0.05,0.08,0.1")
    parser.add_argument("--max-inflight", default="0,1,2,3")
    parser.add_argument("--seed-count", type=int, default=4)
    parser.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 1) // 2))
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--forest-height", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    flow_weights = parse_float_list(args.flow_weights)
    load_weights = parse_float_list(args.load_weights)
    max_inflight = parse_int_list(args.max_inflight)
    seeds = make_seeds(args.seed_count)

    ops = build_ops(args.forest_height, args.batch_size, args.rounds)
    configs = list(itertools.product(flow_weights, load_weights, max_inflight, [seeds]))

    print(
        f"configs={len(configs)} seeds={len(seeds)} "
        f"jobs={args.jobs} flow={flow_weights} load={load_weights} inflight={max_inflight}"
    )

    best = None
    results = []
    start = time.time()
    try:
        with Pool(processes=args.jobs, initializer=init_worker, initargs=(ops,)) as pool:
            for i, result in enumerate(pool.imap_unordered(measure, configs), 1):
                results.append(result)
                if best is None or result.cycles < best.cycles:
                    best = result
                    print(
                        f"best@{i}/{len(configs)} cycles={best.cycles} "
                        f"max_lb={best.max_lb} flow={best.flow_weight} "
                        f"load={best.load_weight} inflight={best.max_inflight}"
                    )
                if i % max(1, math.ceil(len(configs) / 10)) == 0:
                    elapsed = time.time() - start
                    print(f"progress {i}/{len(configs)} elapsed={elapsed:.1f}s")
    except PermissionError as exc:
        print(f"multiprocessing unavailable ({exc}); falling back to serial")
        init_worker(ops)
        for i, config in enumerate(configs, 1):
            result = measure(config)
            results.append(result)
            if best is None or result.cycles < best.cycles:
                best = result
                print(
                    f"best@{i}/{len(configs)} cycles={best.cycles} "
                    f"max_lb={best.max_lb} flow={best.flow_weight} "
                    f"load={best.load_weight} inflight={best.max_inflight}"
                )
            if i % max(1, math.ceil(len(configs) / 10)) == 0:
                elapsed = time.time() - start
                print(f"progress {i}/{len(configs)} elapsed={elapsed:.1f}s")

    results.sort(key=lambda r: (r.cycles, r.max_lb, r.flow_weight, r.load_weight, r.max_inflight))
    print("\nTop results:")
    for result in results[: args.top]:
        print(
            f"cycles={result.cycles} max_lb={result.max_lb} "
            f"flow={result.flow_weight} load={result.load_weight} inflight={result.max_inflight}"
        )


if __name__ == "__main__":
    main()
