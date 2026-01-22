#!/usr/bin/env python3
import os
import random
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from problem import Machine, build_mem_image, reference_kernel2, Tree, Input, N_CORES
from perf_takehome import KernelBuilder


def main():
    forest_height = 10
    rounds = 16
    batch_size = 256
    seed = 0

    random.seed(seed)
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)

    trace = {}
    mem_ref = list(mem)
    for _ in reference_kernel2(mem_ref, trace):
        pass

    block_size = 128
    stages = ["node_val", "hashed_val", "wrapped_idx"]
    debug_checks = []
    for round_id in range(rounds):
        for block_start in range(0, batch_size, block_size):
            debug_checks.append(
                {
                    "round": round_id,
                    "block_start": block_start,
                    "stages": stages,
                }
            )
    for block_start in range(0, batch_size, block_size):
        debug_checks.append(
            {
                "round": rounds - 1,
                "block_start": block_start,
                "stages": ["post_store"],
            }
        )

    kb = KernelBuilder()
    kb.build_kernel(
        forest_height,
        len(forest.values),
        len(inp.indices),
        rounds,
        debug_checks=debug_checks,
    )

    machine = Machine(list(mem), kb.instrs, kb.debug_info(), n_cores=N_CORES, value_trace=trace)
    machine.enable_pause = False
    machine.enable_debug = True
    machine.run()
    print("Debug checks passed.")


if __name__ == "__main__":
    main()
