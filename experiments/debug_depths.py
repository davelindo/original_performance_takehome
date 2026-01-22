#!/usr/bin/env python3
import os
import random
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from problem import Tree, Input, build_mem_image, reference_kernel2, VLEN

MOD = 2**32


def u32(val: int) -> int:
    return val % MOD


def build_L3(forest_values):
    L3 = [forest_values[7 + i] for i in range(8)]
    for i in range(0, 8, 2):
        L3[i + 1] = u32(L3[i + 1] - L3[i])
    return L3


def build_L4(forest_values):
    L4 = [forest_values[15 + i] for i in range(16)]
    for i in range(0, 16, 2):
        L4[i + 1] = u32(L4[i + 1] - L4[i])
    return L4


def select_depth3(idx, L3):
    path = u32(idx - 7)
    b0 = path & 1
    r0 = u32(L3[0] + b0 * L3[1])
    r1 = u32(L3[2] + b0 * L3[3])
    r2 = u32(L3[4] + b0 * L3[5])
    r3 = u32(L3[6] + b0 * L3[7])

    b1 = (path >> 1) & 1
    s0 = u32(r0 + b1 * u32(r1 - r0))
    s1 = u32(r2 + b1 * u32(r3 - r2))

    b2 = (path >> 2) & 1
    return u32(s0 + b2 * u32(s1 - s0))


def select_depth4(idx, L4):
    path = u32(idx - 15)
    b0 = path & 1
    r0 = u32(L4[0] + b0 * L4[1])
    r1 = u32(L4[2] + b0 * L4[3])
    r2 = u32(L4[4] + b0 * L4[5])
    r3 = u32(L4[6] + b0 * L4[7])

    b1 = (path >> 1) & 1
    s0 = u32(r0 + b1 * u32(r1 - r0))
    s1 = u32(r2 + b1 * u32(r3 - r2))

    b2 = (path >> 2) & 1
    t0 = u32(s0 + b2 * u32(s1 - s0))

    r4 = u32(L4[8] + b0 * L4[9])
    r5 = u32(L4[10] + b0 * L4[11])
    r6 = u32(L4[12] + b0 * L4[13])
    r7 = u32(L4[14] + b0 * L4[15])

    s2 = u32(r4 + b1 * u32(r5 - r4))
    s3 = u32(r6 + b1 * u32(r7 - r6))
    t1 = u32(s2 + b2 * u32(s3 - s2))

    b3 = (path >> 3) & 1
    return u32(t0 + b3 * u32(t1 - t0))


def check_block(depth, round_id, trace, forest_values, batch_size, block_start=0, block_size=128):
    L3 = build_L3(forest_values)
    L4 = build_L4(forest_values)
    total = 0
    for k in range(block_size // VLEN):
        for vi in range(VLEN):
            i = block_start + k * VLEN + vi
            if i >= batch_size:
                continue
            idx = trace[(round_id, i, "idx")]
            expected = trace[(round_id, i, "node_val")]
            if depth == 3:
                got = select_depth3(idx, L3)
            else:
                got = select_depth4(idx, L4)
            total += 1
            if got != expected:
                path = u32(idx - (7 if depth == 3 else 15))
                print(
                    f"Mismatch depth={depth} round={round_id} i={i} idx={idx} path={path} "
                    f"got={got} expected={expected}"
                )
                return False
    print(f"Depth {depth} round {round_id}: {total} lanes OK in block {block_start}")
    return True


def main():
    forest_height = 10
    rounds = 16
    batch_size = 256
    block_start = 0

    random.seed(0)
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)

    trace = {}
    for _ in reference_kernel2(mem, trace):
        pass

    ok3 = check_block(3, 3, trace, forest.values, batch_size, block_start)
    ok4 = check_block(4, 4, trace, forest.values, batch_size, block_start)
    ok3 = ok3 and check_block(3, 14, trace, forest.values, batch_size, block_start)
    ok4 = ok4 and check_block(4, 15, trace, forest.values, batch_size, block_start)
    if not (ok3 and ok4):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
