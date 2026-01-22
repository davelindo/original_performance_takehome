import random
from collections import defaultdict, deque

SLOT_LIMITS = {
    "alu": 12,
    "valu": 6,
    "load": 2,
    "store": 2,
    "flow": 1,
    "debug": 64,
}

VLEN = 8
SCHEDULE_SEEDS = (None,) + tuple(range(16))
WEIGHT_FLOW_MULT = 0.0
WEIGHT_LOAD_MULT = 0.02
MAX_INFLIGHT_GROUPS = 2
COMPACT_WINDOW = 2
COMPACT_PASSES = 1

def add_range(s, base, count=VLEN):
    for i in range(count):
        s.add(base + i)

def get_deps(engine, slot):
    reads = set()
    writes = set()
    mem_read = False
    mem_write = False
    
    op = slot[0]
    
    if engine == "alu":
        writes.add(slot[1])
        reads.add(slot[2])
        reads.add(slot[3])
        
    elif engine == "valu":
        if op == "vbroadcast":
            add_range(writes, slot[1])
            reads.add(slot[2])
        elif op == "multiply_add":
            add_range(writes, slot[1])
            add_range(reads, slot[2])
            add_range(reads, slot[3])
            add_range(reads, slot[4])
        else:
            add_range(writes, slot[1])
            add_range(reads, slot[2])
            add_range(reads, slot[3])
            
    elif engine == "load":
        if op == "const":
            writes.add(slot[1])
        elif op == "load":
            writes.add(slot[1])
            reads.add(slot[2])
            mem_read = True
        elif op == "load_offset":
            dest, addr, offset = slot[1], slot[2], slot[3]
            writes.add(dest + offset)
            reads.add(addr + offset)
            mem_read = True
        elif op == "vload":
            add_range(writes, slot[1])
            reads.add(slot[2])
            mem_read = True
            
    elif engine == "store":
        if op == "store":
            reads.add(slot[1])
            reads.add(slot[2])
            mem_write = True
        elif op == "vstore":
            reads.add(slot[1])
            add_range(reads, slot[2])
            mem_write = True
            
    elif engine == "flow":
        if op == "select":
            writes.add(slot[1])
            reads.add(slot[2])
            reads.add(slot[3])
            reads.add(slot[4])
        elif op == "add_imm":
            writes.add(slot[1])
            reads.add(slot[2])
        elif op == "vselect":
            add_range(writes, slot[1])
            add_range(reads, slot[2])
            add_range(reads, slot[3])
            add_range(reads, slot[4])
        elif op == "cond_jump":
            reads.add(slot[1])
        elif op == "cond_jump_rel":
            reads.add(slot[1])
        elif op == "jump_indirect":
            reads.add(slot[1])
        elif op == "trace_write":
            reads.add(slot[1])
        elif op == "coreid":
            writes.add(slot[1])
            
    elif engine == "debug":
        if op == "compare":
            reads.add(slot[1])
        elif op == "vcompare":
            add_range(reads, slot[1])

    return reads, writes, mem_read, mem_write

def combine_tags(tags):
    tag = None
    for t in tags:
        if t is None:
            continue
        if tag is None:
            tag = t
        elif tag != t:
            return None
    return tag

class Node:
    def __init__(self, id, engine, slot):
        self.id = id
        self.engine = engine
        self.slot = slot
        self.reads, self.writes, self.mem_read, self.mem_write = get_deps(engine, slot)
        self.successors = []
        self.predecessors = 0
        self.original_order = id
        self.group_id = None
        self.mem_tag = None
        self.war_predecessors = set()
        self.hard_predecessors = set()

class Dag:
    def __init__(self, nodes, group_counts):
        self.nodes = nodes
        self.engine_counts = defaultdict(int)
        for node in nodes:
            self.engine_counts[node.engine] += 1
        self.group_counts = group_counts

def build_dag(ops):
    nodes = [Node(i, op[0], op[1]) for i, op in enumerate(ops)]
    latest_writer = {} # addr -> Node
    latest_readers = defaultdict(list) # addr -> list[Node]
    group_counts = defaultdict(int)
    group_id = 0
    prev_load = None
    addr_tags = {}
    const_vals = {}
    last_store_by_tag = {}
    last_loads_by_tag = defaultdict(list)

    def set_tags(writes, tag):
        for w in writes:
            if tag is None:
                addr_tags.pop(w, None)
            else:
                addr_tags[w] = tag
    
    for node in nodes:
        op = node.slot[0]
        if node.engine == "load" and node.slot[0] == "load_offset":
            dest, addr, offset = node.slot[1], node.slot[2], node.slot[3]
            if (
                prev_load
                and prev_load["dest"] == dest
                and prev_load["addr"] == addr
                and prev_load["offset"] == offset - 1
            ):
                node.group_id = prev_load["group_id"]
            else:
                node.group_id = group_id
                group_id += 1
            group_counts[node.group_id] += 1
            prev_load = {
                "dest": dest,
                "addr": addr,
                "offset": offset,
                "group_id": node.group_id,
            }
        else:
            prev_load = None

        # Infer memory tag for the address operand (region hint).
        if node.mem_read or node.mem_write:
            if node.engine == "load":
                if op == "load":
                    node.mem_tag = addr_tags.get(node.slot[2])
                elif op == "load_offset":
                    node.mem_tag = addr_tags.get(node.slot[2] + node.slot[3])
                elif op == "vload":
                    node.mem_tag = addr_tags.get(node.slot[2])
            elif node.engine == "store":
                if op in ("store", "vstore"):
                    node.mem_tag = addr_tags.get(node.slot[1])

        # Track constant scalars to tag base pointers for known regions.
        if node.engine == "load" and op == "const":
            const_vals[node.slot[1]] = node.slot[2]

        # Propagate address region tags through simple address arithmetic.
        if node.engine == "load":
            tag_out = None
            if op == "load":
                const_val = const_vals.get(node.slot[2])
                if const_val == 4:
                    tag_out = "forest"
                elif const_val == 5:
                    tag_out = "indices"
                elif const_val == 6:
                    tag_out = "values"
            if op != "const":
                for w in node.writes:
                    const_vals.pop(w, None)
            set_tags(node.writes, tag_out)
        elif node.engine == "alu":
            tag_out = combine_tags([addr_tags.get(r) for r in node.reads])
            for w in node.writes:
                const_vals.pop(w, None)
            set_tags(node.writes, tag_out)
        elif node.engine == "flow":
            tag_out = None
            if op == "add_imm":
                tag_out = addr_tags.get(node.slot[2])
            elif op in ("select", "vselect"):
                tag_out = combine_tags([addr_tags.get(r) for r in node.reads])
            for w in node.writes:
                const_vals.pop(w, None)
            set_tags(node.writes, tag_out)
        elif node.engine == "valu":
            if op == "vbroadcast":
                tag_out = addr_tags.get(node.slot[2])
            elif op in ("+", "-", "multiply_add"):
                tag_out = combine_tags([addr_tags.get(r) for r in node.reads])
            else:
                tag_out = None
            for w in node.writes:
                const_vals.pop(w, None)
            set_tags(node.writes, tag_out)
        else:
            for w in node.writes:
                const_vals.pop(w, None)

        # 1. RAW: Depend on latest writer of my inputs
        dependencies = set()
        for r in node.reads:
            if r in latest_writer:
                dependencies.add(latest_writer[r])
        
        # 2. WAW: Depend on latest writer of my outputs
        for w in node.writes:
            if w in latest_writer:
                dependencies.add(latest_writer[w])
                
        # 3. WAR: Readers of my outputs must be in same or earlier cycle.
        for w in node.writes:
            if w in latest_readers:
                node.war_predecessors.update(latest_readers[w])
                    
        # 4. Memory Dependencies
        if node.mem_write:
            if node.mem_tag is None:
                dependencies.update(last_store_by_tag.values())
                for loads in last_loads_by_tag.values():
                    dependencies.update(loads)
            else:
                for tag in (node.mem_tag, None):
                    if tag in last_store_by_tag:
                        dependencies.add(last_store_by_tag[tag])
                    dependencies.update(last_loads_by_tag.get(tag, []))
            last_store_by_tag[node.mem_tag] = node
            last_loads_by_tag[node.mem_tag] = []
        elif node.mem_read:
            if node.mem_tag is None:
                dependencies.update(last_store_by_tag.values())
            else:
                for tag in (node.mem_tag, None):
                    if tag in last_store_by_tag:
                        dependencies.add(last_store_by_tag[tag])
            last_loads_by_tag[node.mem_tag].append(node)
            
        # Add edges
        for pred in dependencies:
            pred.successors.append(node)
            node.predecessors += 1
        node.hard_predecessors.update(dependencies)
            
        # Update state
        for w in node.writes:
            latest_writer[w] = node
            # Clear readers for this addr, as new writer acts as barrier?
            # Yes, subsequent writers will depend on THIS writer (WAW), so they transitively depend on previous readers.
            latest_readers[w] = []
            
        for r in node.reads:
            latest_readers[r].append(node)
            
    # Ensure halts cannot float ahead of other work.
    halt_nodes = [n for n in nodes if n.engine == "flow" and n.slot[0] == "halt"]
    for halt in halt_nodes:
        for node in nodes:
            if node is halt:
                continue
            node.successors.append(halt)
            halt.predecessors += 1

    for node in nodes:
        node.load_priority = any(succ.engine == "load" for succ in node.successors)

    return Dag(nodes, group_counts)


def compact_schedule(dag, schedule_nodes, window, passes):
    if window <= 0 or not schedule_nodes:
        return schedule_nodes

    for _ in range(max(1, passes)):
        cycle_of = {}
        for cycle, bundle in enumerate(schedule_nodes):
            for nodes in bundle.values():
                for node in nodes:
                    cycle_of[node.id] = cycle

        def can_move(node, target_cycle):
            for pred in node.hard_predecessors:
                if cycle_of[pred.id] >= target_cycle:
                    return False
            for pred in node.war_predecessors:
                if cycle_of[pred.id] > target_cycle:
                    return False
            return True

        num_cycles = len(schedule_nodes)
        for cycle in range(num_cycles):
            for engine, limit in SLOT_LIMITS.items():
                if engine == "debug":
                    continue
                nodes = schedule_nodes[cycle].get(engine, [])
                while len(nodes) < limit:
                    best = None
                    best_key = None
                    best_cycle = None
                    max_look = min(num_cycles, cycle + window + 1)
                    for look in range(cycle + 1, max_look):
                        for node in schedule_nodes[look].get(engine, []):
                            if not can_move(node, cycle):
                                continue
                            key = (
                                look,
                                -node.weighted_height,
                                -int(node.load_priority),
                                node.valu_distance,
                                node.tie,
                            )
                            if best_key is None or key < best_key:
                                best_key = key
                                best = node
                                best_cycle = look
                        if best_key is not None:
                            break
                    if best is None:
                        break
                    schedule_nodes[best_cycle][engine].remove(best)
                    schedule_nodes[cycle].setdefault(engine, []).append(best)
                    cycle_of[best.id] = cycle
                    nodes = schedule_nodes[cycle][engine]

    while schedule_nodes and all(len(v) == 0 for v in schedule_nodes[-1].values()):
        schedule_nodes.pop()
    return schedule_nodes

def schedule_once(ops, seed):
    """
    List scheduler with dependency analysis.
    ops: list of (engine, slot) tuples.
    Returns: list of instructions (dicts).
    """
    # 1. Build DAG scaffold
    dag = build_dag(ops)
    nodes = dag.nodes

    # Critical path heights for prioritization.
    pred_counts = [n.predecessors for n in nodes]
    topo = []
    q = deque([n for n in nodes if pred_counts[n.id] == 0])
    while q:
        node = q.popleft()
        topo.append(node)
        for succ in node.successors:
            pred_counts[succ.id] -= 1
            if pred_counts[succ.id] == 0:
                q.append(succ)
    heights = [1] * len(nodes)
    if len(topo) == len(nodes):
        for node in reversed(topo):
            if node.successors:
                heights[node.id] = 1 + max(heights[succ.id] for succ in node.successors)
    for node in nodes:
        node.height = heights[node.id]

    weights = {engine: (1.0 / SLOT_LIMITS[engine]) for engine in SLOT_LIMITS}
    weights["flow"] *= WEIGHT_FLOW_MULT
    weights["load"] *= WEIGHT_LOAD_MULT
    weighted_heights = [weights[n.engine] for n in nodes]
    if len(topo) == len(nodes):
        for node in reversed(topo):
            if node.successors:
                weighted_heights[node.id] = weights[node.engine] + max(
                    weighted_heights[succ.id] for succ in node.successors
                )
    for node in nodes:
        node.weighted_height = weighted_heights[node.id]

    weighted_depth = [weights[n.engine] for n in nodes]
    if len(topo) == len(nodes):
        for node in topo:
            for succ in node.successors:
                cand = weighted_depth[node.id] + weights[succ.engine]
                if cand > weighted_depth[succ.id]:
                    weighted_depth[succ.id] = cand
    critical_path = 0.0
    for node in nodes:
        path_len = weighted_depth[node.id] + weighted_heights[node.id] - weights[node.engine]
        if path_len > critical_path:
            critical_path = path_len
    for node in nodes:
        node.slack = critical_path - (
            weighted_depth[node.id] + weighted_heights[node.id] - weights[node.engine]
        )

    # Min steps to a VALU op (inf if no VALU reachable).
    inf_dist = len(nodes) + 1
    valu_dist = [inf_dist] * len(nodes)
    if len(topo) == len(nodes):
        for node in topo:
            if node.engine == "valu":
                valu_dist[node.id] = 0
        for node in reversed(topo):
            if node.engine == "valu":
                continue
            if node.successors:
                valu_dist[node.id] = 1 + min(valu_dist[succ.id] for succ in node.successors)
    for node in nodes:
        node.valu_distance = valu_dist[node.id]

    # Remaining VALU ops on the longest path to sink.
    valu_heights = [0] * len(nodes)
    if len(topo) == len(nodes):
        for node in reversed(topo):
            max_succ = 0
            if node.successors:
                max_succ = max(valu_heights[succ.id] for succ in node.successors)
            valu_heights[node.id] = max_succ + (1 if node.engine == "valu" else 0)
    for node in nodes:
        node.valu_height = valu_heights[node.id]

    if seed is None:
        for node in nodes:
            node.tie = (0.0, node.original_order)
    else:
        rng = random.Random(seed)
        for node in nodes:
            node.tie = (rng.random(), node.original_order)

    # 2. Schedule
    ready_queue = deque([n for n in nodes if n.predecessors == 0])
    bundles = []
    schedule_nodes = []

    remaining_counts = dict(dag.engine_counts)
    remaining_group_counts = dict(dag.group_counts)
    valu_idle_with_gather = 0
    valu_underfilled_cycles = 0
    ready_valu_total = 0
    ready_valu_hist = defaultdict(int)
    max_inflight_groups = MAX_INFLIGHT_GROUPS
    done_set = set()

    while True:
        current_bundle = defaultdict(list)
        current_nodes = defaultdict(list)
        in_progress_group_ids = {
            group_id
            for group_id, remaining in remaining_group_counts.items()
            if 0 < remaining < dag.group_counts[group_id]
        }
        in_progress_groups = bool(in_progress_group_ids)

        ready_list = list(ready_queue)
        ready_by_engine = defaultdict(list)
        for node in ready_list:
            ready_by_engine[node.engine].append(node)

        ready_valu = ready_by_engine.get("valu", [])
        ready_valu_count = len(ready_valu)
        ready_valu_total += ready_valu_count
        ready_valu_hist[ready_valu_count] += 1
        if ready_valu_count < SLOT_LIMITS["valu"]:
            valu_underfilled_cycles += 1
        use_valu_distance = remaining_counts.get("valu", 0) > 0
        scheduled_nodes = []
        scheduled_set = set()

        def war_ready(node):
            if not node.war_predecessors:
                return True
            for pred in node.war_predecessors:
                if pred not in done_set and pred not in scheduled_set:
                    return False
            return True

        def release_count(node):
            return sum(1 for succ in node.successors if succ.predecessors == 1)

        engine_pressure = {}
        for engine, count in remaining_counts.items():
            limit = SLOT_LIMITS[engine]
            engine_pressure[engine] = count / limit if limit else 0.0
        engine_order = sorted(
            [e for e in SLOT_LIMITS if e in engine_pressure],
            key=lambda e: (-engine_pressure[e], e),
        )
        progress = True
        while progress:
            progress = False
            for engine in engine_order:
                limit = SLOT_LIMITS[engine]
                if len(current_bundle[engine]) >= limit:
                    continue
                engine_ready = ready_by_engine.get(engine, [])
                if not engine_ready:
                    continue
                scheduled_in_engine = False
                if engine == "load":
                    def load_key(node):
                        if node.group_id is None:
                            return (
                                1,
                                -release_count(node),
                                node.valu_distance,
                                -node.weighted_height,
                                -int(node.load_priority),
                                node.tie,
                            )
                        remaining = remaining_group_counts.get(node.group_id, VLEN + 1)
                        return (
                            0,
                            remaining,
                            -release_count(node),
                            node.valu_distance,
                            -node.weighted_height,
                            -int(node.load_priority),
                            node.tie,
                        )

                    load_ready = [
                        node for node in engine_ready
                        if node not in scheduled_set and war_ready(node)
                    ]
                    if max_inflight_groups and len(in_progress_group_ids) >= max_inflight_groups:
                        constrained = [
                            node for node in load_ready
                            if node.group_id is None or node.group_id in in_progress_group_ids
                        ]
                        if constrained:
                            load_ready = constrained

                    load_ready.sort(key=load_key)
                    for node in load_ready:
                        if len(current_bundle[engine]) >= limit:
                            break
                        current_bundle[engine].append(node.slot)
                        current_nodes[engine].append(node)
                        scheduled_nodes.append(node)
                        scheduled_set.add(node)
                        if node.group_id is not None:
                            remaining_group_counts[node.group_id] -= 1
                        progress = True
                        scheduled_in_engine = True
                        break
                else:
                    if use_valu_distance:
                        engine_ready = [node for node in engine_ready if war_ready(node)]
                        engine_ready.sort(
                            key=lambda n: (
                                -int(n.load_priority),
                                -release_count(n),
                                n.valu_distance,
                                -n.weighted_height,
                                n.tie,
                            )
                        )
                    else:
                        engine_ready = [node for node in engine_ready if war_ready(node)]
                        engine_ready.sort(
                            key=lambda n: (
                                -int(n.load_priority),
                                -release_count(n),
                                -n.weighted_height,
                                n.tie,
                            )
                        )
                    for node in engine_ready:
                        if len(current_bundle[engine]) >= limit:
                            break
                        if node in scheduled_set:
                            continue
                        current_bundle[engine].append(node.slot)
                        current_nodes[engine].append(node)
                        scheduled_nodes.append(node)
                        scheduled_set.add(node)
                        progress = True
                        scheduled_in_engine = True
                        break

        next_ready = [node for node in ready_list if node not in scheduled_set]

        if not scheduled_nodes:
            if not next_ready and not ready_queue:
                break

        if current_bundle:
            bundles.append(dict(current_bundle))
            schedule_nodes.append(dict(current_nodes))
            if in_progress_groups and "valu" not in current_bundle:
                valu_idle_with_gather += 1

        newly_ready = []
        for node in scheduled_nodes:
            remaining_counts[node.engine] -= 1
            for succ in node.successors:
                succ.predecessors -= 1
                if succ.predecessors == 0:
                    newly_ready.append(succ)

        ready_queue = deque(next_ready + newly_ready)
        done_set.update(scheduled_nodes)

        if not current_bundle and ready_queue:
            pass

        if not current_bundle and not ready_queue:
            break

    if COMPACT_WINDOW > 0:
        schedule_nodes = compact_schedule(dag, schedule_nodes, COMPACT_WINDOW, COMPACT_PASSES)
        bundles = []
        for bundle_nodes in schedule_nodes:
            bundles.append(
                {engine: [node.slot for node in nodes] for engine, nodes in bundle_nodes.items()}
            )
        stats = {}
    else:
        stats = {
            "cycles": len(bundles),
            "valu_idle_with_gather": valu_idle_with_gather,
            "valu_underfilled_cycles": valu_underfilled_cycles,
            "ready_valu_total": ready_valu_total,
            "ready_valu_hist": dict(ready_valu_hist),
        }
    return bundles, stats


def schedule(ops):
    best_bundles = None
    best_stats = None
    best_cycles = None
    for seed in SCHEDULE_SEEDS:
        bundles, stats = schedule_once(ops, seed)
        cycles = len(bundles)
        if best_cycles is None or cycles < best_cycles:
            best_cycles = cycles
            best_bundles = bundles
            best_stats = stats
    schedule.last_stats = best_stats or {}
    return best_bundles
