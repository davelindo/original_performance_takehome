#!/usr/bin/env python3
import os
import random
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from problem import Machine, build_mem_image, reference_kernel2, Tree, Input, N_CORES
from perf_takehome import KernelBuilder


def classify_addr(addr, forest_values_p, inp_indices_p, inp_values_p, batch_size):
    if addr < 7:
        return "header"
    if addr < inp_indices_p:
        return "forest"
    if addr < inp_values_p:
        return "indices"
    if addr < inp_values_p + batch_size:
        return "values"
    return "extra"


def main():
    forest_height = 10
    rounds = 16
    batch_size = 256
    seed = 0

    random.seed(seed)
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)

    mem_ref = list(mem)
    for _ in reference_kernel2(mem_ref):
        pass

    kb = KernelBuilder()
    kb.build_kernel(forest_height, len(forest.values), batch_size, rounds)
    machine = Machine(list(mem), kb.instrs, kb.debug_info(), n_cores=N_CORES)
    machine.enable_pause = False
    machine.enable_debug = False
    machine.run()
    mem_out = machine.mem

    if len(mem_out) != len(mem_ref):
        print(f"Memory size mismatch: got={len(mem_out)} expected={len(mem_ref)}")
        raise SystemExit(1)

    forest_values_p = mem[4]
    inp_indices_p = mem[5]
    inp_values_p = mem[6]

    mismatches = 0
    per_region = {"header": 0, "forest": 0, "indices": 0, "values": 0, "extra": 0}
    first = []
    for i, (got, exp) in enumerate(zip(mem_out, mem_ref)):
        if got != exp:
            region = classify_addr(i, forest_values_p, inp_indices_p, inp_values_p, batch_size)
            per_region[region] += 1
            mismatches += 1
            if len(first) < 10:
                first.append((i, region, got, exp))

    if mismatches == 0:
        print("Memory matches reference.")
        return

    print(f"Mismatches: {mismatches}")
    for region in ("header", "forest", "indices", "values", "extra"):
        if per_region[region]:
            print(f"{region}: {per_region[region]}")

    for i, region, got, exp in first:
        print(f"addr={i} region={region} got={got} expected={exp}")


if __name__ == "__main__":
    main()
