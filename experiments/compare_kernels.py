#!/usr/bin/env python3
import os
import random
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from problem import Machine, build_mem_image, Tree, Input, N_CORES
from perf_takehome import KernelBuilder


def build_debug_checks(rounds, batch_size, block_size=128):
    debug_checks = []
    for round_id in range(rounds):
        for block_start in range(0, batch_size, block_size):
            debug_checks.append(
                {
                    "round": round_id,
                    "block_start": block_start,
                    "stages": ["node_val", "hashed_val", "wrapped_idx"],
                }
            )
    return debug_checks


def run_kernel(mem, forest_height, rounds, batch_size, debug_checks=None):
    kb = KernelBuilder()
    kb.build_kernel(
        forest_height,
        mem[1],
        batch_size,
        rounds,
        debug_checks=debug_checks,
    )
    machine = Machine(list(mem), kb.instrs, kb.debug_info(), n_cores=N_CORES)
    machine.enable_pause = False
    machine.enable_debug = False
    machine.run()
    return machine.mem, machine.cycle


def main():
    forest_height = 10
    rounds = 16
    batch_size = 256
    seed = 0

    random.seed(seed)
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)
    inp_indices_p = mem[5]
    inp_values_p = mem[6]

    mem_base, cycles_base = run_kernel(mem, forest_height, rounds, batch_size)
    debug_checks = build_debug_checks(rounds, batch_size)
    mem_debug, cycles_debug = run_kernel(
        mem, forest_height, rounds, batch_size, debug_checks=debug_checks
    )

    values_base = mem_base[inp_values_p : inp_values_p + batch_size]
    values_debug = mem_debug[inp_values_p : inp_values_p + batch_size]
    indices_base = mem_base[inp_indices_p : inp_indices_p + batch_size]
    indices_debug = mem_debug[inp_indices_p : inp_indices_p + batch_size]

    if values_base == values_debug and indices_base == indices_debug:
        print("Outputs match for debug vs non-debug kernels.")
        print(f"cycles: base={cycles_base} debug={cycles_debug}")
        return

    for i, (a, b) in enumerate(zip(values_base, values_debug)):
        if a != b:
            print(f"First values mismatch at i={i}: base={a} debug={b}")
            break
    for i, (a, b) in enumerate(zip(indices_base, indices_debug)):
        if a != b:
            print(f"First indices mismatch at i={i}: base={a} debug={b}")
            break
    print(f"cycles: base={cycles_base} debug={cycles_debug}")


if __name__ == "__main__":
    main()
