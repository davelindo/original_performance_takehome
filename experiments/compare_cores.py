#!/usr/bin/env python3
import os
import random
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from problem import Machine, build_mem_image, Tree, Input
from perf_takehome import KernelBuilder


def run_kernel(mem, forest_height, rounds, batch_size, n_cores):
    kb = KernelBuilder()
    kb.build_kernel(forest_height, mem[1], batch_size, rounds)
    machine = Machine(list(mem), kb.instrs, kb.debug_info(), n_cores=n_cores)
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

    mem_1, cycles_1 = run_kernel(mem, forest_height, rounds, batch_size, n_cores=1)
    mem_2, cycles_2 = run_kernel(mem, forest_height, rounds, batch_size, n_cores=2)

    inp_indices_p = mem[5]
    inp_values_p = mem[6]
    values_1 = mem_1[inp_values_p : inp_values_p + batch_size]
    values_2 = mem_2[inp_values_p : inp_values_p + batch_size]
    indices_1 = mem_1[inp_indices_p : inp_indices_p + batch_size]
    indices_2 = mem_2[inp_indices_p : inp_indices_p + batch_size]

    if values_1 == values_2 and indices_1 == indices_2:
        print("Outputs match for 1 core vs 2 cores.")
    else:
        for i, (a, b) in enumerate(zip(values_1, values_2)):
            if a != b:
                print(f"First values mismatch at i={i}: 1core={a} 2cores={b}")
                break
        for i, (a, b) in enumerate(zip(indices_1, indices_2)):
            if a != b:
                print(f"First indices mismatch at i={i}: 1core={a} 2cores={b}")
                break

    print(f"cycles: 1core={cycles_1} 2cores={cycles_2}")


if __name__ == "__main__":
    main()
