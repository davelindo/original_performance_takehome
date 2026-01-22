"""
# Anthropic's Original Performance Engineering Take-home (Release version)

Copyright Anthropic PBC 2026. Permission is granted to modify and use, but not
to publish or redistribute your solutions so it's hard to find spoilers.

# Task

- Optimize the kernel (in KernelBuilder.build_kernel) as much as possible in the
  available time, as measured by test_kernel_cycles on a frozen separate copy
  of the simulator.

Validate your results using `python tests/submission_tests.py` without modifying
anything in the tests/ folder.

We recommend you look through problem.py next.
"""

from problem import (
    VLEN, SCRATCH_SIZE, HASH_STAGES,
)
from optimizer import schedule

class KernelBuilder:
    def __init__(self):
        self.all_ops = []
        self.scratch = {}
        self.scratch_debug = {}
        self.scratch_ptr = 0
        self.const_map = {}

    def debug_info(self):
        return None

    def build(self, slots):
        return schedule(slots)

    def add(self, engine, slot):
        self.all_ops.append((engine, slot))

    def alloc_scratch(self, name=None, length=1):
        addr = self.scratch_ptr
        self.scratch_ptr += length
        assert self.scratch_ptr <= SCRATCH_SIZE
        return addr

    def scratch_const(self, val):
        if val not in self.const_map:
            addr = self.alloc_scratch()
            self.add("load", ("const", addr, val))
            self.const_map[val] = addr
        return self.const_map[val]

    def vector_const(self, val):
        if not hasattr(self, 'vc_map'): self.vc_map = {}
        if val in self.vc_map: return self.vc_map[val]
        s_addr = self.scratch_const(val)
        v_addr = self.alloc_scratch(length=VLEN)
        self.add("valu", ("vbroadcast", v_addr, s_addr))
        self.vc_map[val] = v_addr
        return v_addr

    def get_vector_const(self, val):
        return self.vector_const(val)

    def build_kernel(self, forest_height, n_nodes, batch_size, rounds, debug_checks=None):
        self.vc_map = {}
        if debug_checks is None:
            debug_checks = []
        if isinstance(debug_checks, dict):
            debug_checks = [debug_checks]
        store_indices = any("post_store" in spec.get("stages", ()) for spec in debug_checks)

        def get_debug_spec(stage, round_id, block_start):
            for spec in debug_checks:
                if spec.get("round") != round_id:
                    continue
                if spec.get("block_start") != block_start:
                    continue
                if stage in spec.get("stages", ()):
                    return spec
            return None

        def add_vcompare(vec_addr, keys):
            self.add("debug", ("vcompare", vec_addr, keys))
        # Init
        s_vars = [self.alloc_scratch() for _ in range(7)]
        tmp1 = self.alloc_scratch()
        for i in range(7):
            self.add("load", ("const", tmp1, i))
            self.add("load", ("load", s_vars[i], tmp1))
        self.add("flow", ("pause",))

        v_zero = self.get_vector_const(0)
        v_one = self.get_vector_const(1)
        v_two = self.get_vector_const(2)
        
        v_f_base = self.alloc_scratch(length=VLEN)
        self.add("valu", ("vbroadcast", v_f_base, s_vars[4]))
        v_off1 = self.alloc_scratch(length=VLEN)
        v_off2 = self.alloc_scratch(length=VLEN)
        self.add("valu", ("-", v_off1, v_one, v_f_base))
        self.add("valu", ("-", v_off2, v_two, v_f_base))
        
        h_ks = {0: self.get_vector_const(4097), 2: self.get_vector_const(33), 4: self.get_vector_const(9)}

        v_L0 = self.alloc_scratch(length=VLEN)
        s_zero = self.scratch_const(0)
        s_one = self.scratch_const(1)
        self.add("alu", ("+", tmp1, s_vars[4], s_zero))
        self.add("load", ("load", tmp1, tmp1))
        self.add("valu", ("vbroadcast", v_L0, tmp1))
        
        v_L1_base = self.alloc_scratch(length=VLEN)
        v_L1_hi = self.alloc_scratch(length=VLEN)
        for i, v in enumerate([v_L1_base, v_L1_hi]):
            c = self.scratch_const(1+i)
            self.add("alu", ("+", tmp1, s_vars[4], c))
            self.add("load", ("load", tmp1, tmp1))
            self.add("valu", ("vbroadcast", v, tmp1))
        
        v_L2 = [self.alloc_scratch(length=VLEN) for _ in range(4)]
        for i in range(4):
            c = self.scratch_const(3+i)
            self.add("alu", ("+", tmp1, s_vars[4], c))
            self.add("load", ("load", tmp1, tmp1))
            self.add("valu", ("vbroadcast", v_L2[i], tmp1))

        BLOCK_SIZE = 256 
        nv = BLOCK_SIZE // VLEN
        p_val = [self.alloc_scratch(length=VLEN) for _ in range(nv)]
        v_nv = [self.alloc_scratch(length=VLEN) for _ in range(nv)]
        v_addr = [self.alloc_scratch(length=VLEN) for _ in range(nv)]
        v_path = [self.alloc_scratch(length=VLEN) for _ in range(nv)]
        v_a = [self.alloc_scratch(length=VLEN) for _ in range(nv)]
        addr_temps = [self.alloc_scratch() for _ in range(nv)]
        b_start_addr = self.alloc_scratch()
        
        for b_start in range(0, batch_size, BLOCK_SIZE):
            real_nv = (min(batch_size, b_start + BLOCK_SIZE) - b_start + VLEN - 1) // VLEN
            
            self.add("load", ("const", b_start_addr, b_start))
            for k in range(real_nv):
                t = addr_temps[k]
                self.add("alu", ("+", t, s_vars[6], b_start_addr))
                self.add("flow", ("add_imm", t, t, k*VLEN))
                self.add("load", ("vload", p_val[k], t))
                self.add("valu", ("+", v_addr[k], v_f_base, v_zero))
                self.add("valu", ("+", v_path[k], v_zero, v_zero))

            for t in range(rounds + real_nv - 1):
                for k in range(real_nv):
                    r = t - k
                    if r < 0 or r >= rounds:
                        continue
                    d = r % (forest_height + 1)
                    dbg_node = get_debug_spec("node_val", r, b_start)
                    dbg_hash = get_debug_spec("hashed_val", r, b_start)
                    dbg_wrap = get_debug_spec("wrapped_idx", r, b_start)

                    if d == 0:
                        self.add("valu", ("+", v_nv[k], v_L0, v_zero))
                    elif d == 1:
                        self.add("flow", ("vselect", v_nv[k], v_path[k], v_L1_hi, v_L1_base))
                    elif d == 2:
                        self.add("flow", ("vselect", v_a[k], v_path[k], v_L2[2], v_L2[0]))
                        self.add("flow", ("vselect", v_path[k], v_path[k], v_L2[3], v_L2[1]))
                        self.add("flow", ("vselect", v_a[k], v_nv[k], v_path[k], v_a[k]))
                    else:
                        for vi in range(VLEN):
                            self.add("load", ("load_offset", v_nv[k], v_addr[k], vi))

                    if d == 2:
                        node_reg = v_a[k]
                    else:
                        node_reg = v_nv[k]

                    if dbg_node is not None and k < dbg_node.get("max_vectors", real_nv):
                        if b_start + (k + 1) * VLEN <= batch_size:
                            keys = [
                                (r, b_start + k * VLEN + vi, "node_val")
                                for vi in range(VLEN)
                            ]
                            add_vcompare(node_reg, keys)
                    self.add("valu", ("^", p_val[k], p_val[k], node_reg))
                    for hi in range(6):
                        if hi in [0, 2, 4]:
                            self.add("valu", ("multiply_add", p_val[k], p_val[k], h_ks[hi], self.get_vector_const(HASH_STAGES[hi][1])))
                        else:
                            op1, val1, op2, op3, val3 = HASH_STAGES[hi]
                            self.add("valu", (op1, v_nv[k], p_val[k], self.get_vector_const(val1)))
                            self.add("valu", (op3, p_val[k], p_val[k], self.get_vector_const(val3)))
                            self.add("valu", (op2, p_val[k], v_nv[k], p_val[k]))

                    if dbg_hash is not None and k < dbg_hash.get("max_vectors", real_nv):
                        if b_start + (k + 1) * VLEN <= batch_size:
                            keys = [
                                (r, b_start + k * VLEN + vi, "hashed_val")
                                for vi in range(VLEN)
                            ]
                            add_vcompare(p_val[k], keys)

                    if d == forest_height:
                        self.add("valu", ("+", v_addr[k], v_f_base, v_zero))
                    else:
                        if d == 1:
                            for vi in range(VLEN):
                                self.add("alu", ("&", v_nv[k] + vi, p_val[k] + vi, s_one))
                            b_reg = v_nv[k]
                        else:
                            for vi in range(VLEN):
                                self.add("alu", ("&", v_path[k] + vi, p_val[k] + vi, s_one))
                            b_reg = v_path[k]
                        self.add("flow", ("vselect", v_a[k], b_reg, v_off2, v_off1))
                        self.add("valu", ("multiply_add", v_addr[k], v_addr[k], v_two, v_a[k]))


                    if dbg_wrap is not None and k < dbg_wrap.get("max_vectors", real_nv):
                        if b_start + (k + 1) * VLEN <= batch_size:
                            self.add("valu", ("-", v_a[k], v_addr[k], v_f_base))
                            keys = [
                                (r, b_start + k * VLEN + vi, "wrapped_idx")
                                for vi in range(VLEN)
                            ]
                            add_vcompare(v_a[k], keys)

            for k in range(real_nv):
                t = addr_temps[k]
                if store_indices:
                    self.add("valu", ("-", v_a[k], v_addr[k], v_f_base))
                    self.add("alu", ("+", t, s_vars[5], b_start_addr))
                    self.add("flow", ("add_imm", t, t, k*VLEN))
                    self.add("store", ("vstore", t, v_a[k]))
                self.add("alu", ("+", t, s_vars[6], b_start_addr))
                self.add("flow", ("add_imm", t, t, k*VLEN))
                self.add("store", ("vstore", t, p_val[k]))

            dbg = get_debug_spec("post_store", rounds - 1, b_start)
            if dbg is not None:
                max_vectors = dbg.get("max_vectors", real_nv)
                for k in range(min(real_nv, max_vectors)):
                    if b_start + (k + 1) * VLEN > batch_size:
                        break
                    t = addr_temps[k]
                    if store_indices:
                        self.add("alu", ("+", t, s_vars[5], b_start_addr))
                        self.add("flow", ("add_imm", t, t, k*VLEN))
                        self.add("load", ("vload", v_a[k], t))
                        keys = [
                            (rounds - 1, b_start + k * VLEN + vi, "wrapped_idx")
                            for vi in range(VLEN)
                        ]
                        add_vcompare(v_a[k], keys)

                    self.add("alu", ("+", t, s_vars[6], b_start_addr))
                    self.add("flow", ("add_imm", t, t, k*VLEN))
                    self.add("load", ("vload", v_nv[k], t))
                    keys = [
                        (rounds - 1, b_start + k * VLEN + vi, "hashed_val")
                        for vi in range(VLEN)
                    ]
                    add_vcompare(v_nv[k], keys)
        
        self.add("flow", ("halt",))
        self.instrs = self.build(self.all_ops)
