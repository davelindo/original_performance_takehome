import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from perf_takehome import KernelBuilder
from problem import Machine, build_mem_image, reference_kernel2, Tree, Input, N_CORES, VLEN

def run_test():
    forest_height = 10
    rounds = 16
    batch_size = 256
    
    print(f"Testing {forest_height=}, {rounds=}, {batch_size=}")
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)

    kb = KernelBuilder()
    kb.build_kernel(forest_height, len(forest.values), len(inp.indices), rounds)

    machine = Machine(mem, kb.instrs, kb.debug_info(), n_cores=N_CORES)
    machine.enable_pause = False
    machine.enable_debug = False
    machine.run()

    # Check correctness
    for ref_mem in reference_kernel2(mem):
        pass

    inp_values_p = ref_mem[6]
    output_correct = (
        machine.mem[inp_values_p : inp_values_p + len(inp.values)]
        == ref_mem[inp_values_p : inp_values_p + len(inp.values)]
    )
    
    if output_correct:
        print("Output Correct.")
        print("CYCLES: ", machine.cycle)
    else:
        print("Output INCORRECT.")

if __name__ == "__main__":
    run_test()
