# How I Did It: Performance Take‑Home Optimization

This document summarizes the work recorded in the current diff against `origin/main`. It focuses on what I changed, why, and how I validated it.

## Goals

- Reduce cycle count on the released single‑core simulator with `forest_height=10`, `rounds=16`, `batch_size=256`.
- Keep correctness intact and avoid changing anything under `tests/`.
- Build tooling to measure cycle counts and understand scheduling bottlenecks.

## High‑Level Strategy

1. Replace the scalar reference kernel with a fully vectorized kernel that fits the simulator’s VLEN=8 model.
2. Use precomputed constants and a shallow lookup strategy for depth 0–2, then gather‑load for deeper levels.
3. Move index tracking to an address‑vector model (`addr = base + idx`) to avoid scalar index maintenance.
4. Replace the one‑slot‑per‑bundle packing with a DAG‑based scheduler that packs across engines and overlaps load/valu.
5. Build local experiments for profiling and tuning the scheduler.

## Kernel Changes (`perf_takehome.py`)

### Vectorized kernel

- The old scalar kernel and helper `build_hash()` were removed. The new kernel emits vector instructions directly and delegates instruction packing to the scheduler.
- The kernel uses per‑vector scratch buffers for:
  - `p_val[k]`: current values
  - `v_nv[k]`: node values (gathered or selected)
  - `v_addr[k]`: per‑lane addresses (base + idx)
  - `v_path[k]`: 1‑bit path used for early depths
  - `v_a[k]`: scratch for node selection and address updates

### Shallow depth selection (d = 0–2)

- d=0: broadcast root value (`v_L0`) once and reuse.
- d=1: `flow.vselect` between two constants (`v_L1_base`, `v_L1_hi`) using `v_path`.
- d=2: `flow.vselect` tree over `v_L2[0..3]`, using `v_path` to select among four constants.

This keeps early rounds load‑free.

### Deeper depths (d >= 3)

- Use gather loads via `load_offset` (one per lane) into `v_nv[k]`.
- This trades VALU for LOAD, which is favorable at the current bottleneck mix.

### Address update via `v_addr`

- Maintain `v_addr` directly:
  - Precompute `v_off1 = 1 - base` and `v_off2 = 2 - base`.
  - For each round: select the appropriate offset with `flow.vselect`, then
    `v_addr = multiply_add(v_addr, v_two, offset)`.
- On wrap depth (`d == forest_height`), reset to base directly.

### Hash pipeline

- Keep the hash stages in VALU; use `multiply_add` for the fused stages (0,2,4).
- For the non‑fused stages, compute into `v_nv[k]` then combine into `p_val[k]`.

### Wavefront round ordering

- Use a `t`/`k` wavefront schedule for rounds to desynchronize vectors:
  - For `t in [0..rounds+real_nv-1]`, schedule vector `k` at round `r = t - k` if valid.
  - This improves overlap of gather loads with hash work.

### Debug support

- Retained debug hooks by emitting `debug.vcompare` for specific rounds/blocks when requested.

## Scheduler Changes (`optimizer.py`)

### DAG construction

- Build a node DAG from raw ops with explicit read/write sets.
- Track memory tags by address origin to reduce unnecessary load/store ordering.
- Identify contiguous `load_offset` sequences and tag them as a group for join‑aware scheduling.

### Scheduling policy

- Compute critical‑path heights and weighted heights (engine‑weighted).
- Prioritize ready ops based on:
  - weighted criticality
  - load group completion (finish groups rather than stripe across all groups)
  - load‑priority hints
- Cap the number of inflight load groups (`MAX_INFLIGHT_GROUPS`) to avoid starving VALU.

### Schedule compaction

- Add an optional compacting pass that attempts to pull ops earlier in a sliding window, respecting deps and slot limits.

## Experiment and Profiling Tooling (`experiments/`)

I added small, local scripts to keep tight feedback loops:

- `experiments/schedule_profile.py`: cycle/utilization profile + lower bounds.
- `experiments/tune_scheduler.py`: sweep scheduling heuristics and weights.
- `experiments/reproduce_perf.py`: convenience wrapper for benchmark cycles.
- `experiments/compare_kernels.py`, `compare_cores.py`: quick comparisons.
- `experiments/debug_depths.py`, `debug_kernel_checks.py`, `mem_diff.py`: correctness and debug helpers.

These are local helpers only; they do not change the simulator or tests.

## Validation

- Tests folder is untouched (`git diff origin/main tests/` is empty).
- `python tests/submission_tests.py` passes all correctness tests.
- Current speed: **1387 cycles** for `forest_height=10`, `rounds=16`, `batch_size=256`.

## Proof Printout (Submission Tests)

```text
......F..
======================================================================
FAIL: test_opus45_improved_harness (__main__.SpeedTests.test_opus45_improved_harness)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "/root/anthropic_kernel/original_performance_takehome/tests/submission_tests.py", line 115, in test_opus45_improved_harness
    assert cycles() < 1363
           ^^^^^^^^^^^^^^^
AssertionError

----------------------------------------------------------------------
Ran 9 tests in 12.337s

FAILED (failures=1)
Testing forest_height=10, rounds=16, batch_size=256
CYCLES:  1387
Testing forest_height=10, rounds=16, batch_size=256
CYCLES:  1387
Testing forest_height=10, rounds=16, batch_size=256
CYCLES:  1387
Testing forest_height=10, rounds=16, batch_size=256
CYCLES:  1387
Testing forest_height=10, rounds=16, batch_size=256
CYCLES:  1387
Testing forest_height=10, rounds=16, batch_size=256
CYCLES:  1387
Testing forest_height=10, rounds=16, batch_size=256
CYCLES:  1387
Testing forest_height=10, rounds=16, batch_size=256
CYCLES:  1387
Testing forest_height=10, rounds=16, batch_size=256
CYCLES:  1387
Speedup over baseline:  106.51333813987023
```

## Tooling Printout (Scheduler Profile)

```text
cycles: 1387
alu: util=0.235 peak=12/12 idle=63.7% full=2.2%
valu: util=0.880 peak=6/6 idle=0.6% full=65.5%
load: util=0.949 peak=2/2 idle=4.1% full=93.9%
store: util=0.012 peak=1/2 idle=97.7% full=0.0%
flow: util=0.578 peak=1/1 idle=42.2% full=57.8%
lower_bounds: alu:326, valu:1220, load:1316, store:16, flow:802 max: 1316
overlap(load+valu): 0.955 load_only=0.004 valu_only=0.040
```

## Known Tradeoffs / Limitations

- The kernel is still load‑bound at the current best (LB ~1316); remaining gains require either:
  - fewer loads per round (e.g., shallow selection for additional depths), or
  - further scheduling improvements to close the remaining slack.
- Scratch space is near the limit; adding new vector tables requires freeing other buffers.

## Next Steps (if pushing below 1363)

- Consider removing more load work by adding a depth‑3 table selection if scratch allows.
- Investigate ALU offload for parity extraction to reduce VALU pressure.
- Audit for any remaining vector moves implemented as VALU `+ 0` (convert to FLOW if slack).
