#!/usr/bin/env python3
import sys
import json
import argparse
from pathlib import Path
import importlib.util


def load_oracle_map(jsonl_path: Path):
    mp = {}
    with open(jsonl_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            key = str(rec.get('id', '')).lower()
            # Use the raw solver program for interpretation to preserve semantics
            prog = rec.get('program')
            if key and prog:
                mp[key] = prog
    return mp


def _rewrite_var_calls_to_apply(program: str) -> str:
    """Rewrite higher-order calls into apply(...)/papply(...) to avoid interpreter issues.

    Handles two sugars seen in solver programs:
    1) Variable call sugar: x2(I) -> apply(x2, I)
    2) Expression call sugar: (branch(...))(I) -> apply(branch(...), I)
    Also maps two-arg var calls to papply: x2(a,b) -> papply(x2, a, b)
    """
    def rewrite_var_calls(s: str) -> str:
        i = 0
        out = []
        n = len(s)
        while i < n:
            ch = s[i]
            if ch == 'x':
                j = i + 1
                while j < n and s[j].isdigit():
                    j += 1
                if j > i + 1:
                    k = j
                    while k < n and s[k].isspace():
                        k += 1
                    if k < n and s[k] == '(':
                        depth = 0
                        arg_start = k
                        p = k
                        while p < n:
                            if s[p] == '(':
                                depth += 1
                            elif s[p] == ')':
                                depth -= 1
                                if depth == 0:
                                    break
                            p += 1
                        if p < n and depth == 0:
                            args = s[arg_start + 1:p].strip()
                            if ',' in args and args.count(',') == 1:
                                a, b = args.split(',', 1)
                                out.append(f"papply({s[i:j]}, {a.strip()}, {b.strip()})")
                            else:
                                out.append(f"apply({s[i:j]}, {args})")
                            i = p + 1
                            continue
            out.append(ch)
            i += 1
        return ''.join(out)

    def rewrite_expr_calls(s: str) -> str:
        i = 0
        n = len(s)
        while i < n:
            if s[i] == ')':
                j = i + 1
                while j < n and s[j].isspace():
                    j += 1
                if j < n and s[j] == '(':
                    depth = 1
                    k = i - 1
                    while k >= 0:
                        if s[k] == ')':
                            depth += 1
                        elif s[k] == '(':
                            depth -= 1
                            if depth == 0:
                                break
                        k -= 1
                    if k >= 0 and depth == 0:
                        callee = s[k + 1:i]
                        depth2 = 0
                        p = j
                        while p < n:
                            if s[p] == '(':
                                depth2 += 1
                            elif s[p] == ')':
                                depth2 -= 1
                                if depth2 == 0:
                                    break
                            p += 1
                        if p < n and depth2 == 0:
                            args = s[j + 1:p]
                            s = s[:k] + f"apply({callee}, {args})" + s[p + 1:]
                            n = len(s)
                            i = k + 1
                            continue
            i += 1
        return s

    prev = None
    cur = program
    while prev != cur:
        prev = cur
        cur = rewrite_var_calls(cur)
        cur = rewrite_expr_calls(cur)
    return cur


def get_task_id(task):
    tid = getattr(task, 'id', None) or getattr(task, 'uid', None) or getattr(task, 'name', None)
    if tid is None and hasattr(task, 'path'):
        tid = Path(task.path).stem
    return str(tid).lower() if tid is not None else None


def main():
    parser = argparse.ArgumentParser(description="Control evaluation: run oracle program per ARC task via dsl interpreter and report accuracy")
    parser.add_argument("--dataset", type=str, default="arcagi")
    parser.add_argument("--oracle_jsonl", type=str, default="data/arc/oracle/arc_oracle.jsonl")
    parser.add_argument("--show_failures", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of tasks")
    parser.add_argument("--compare_solvers", action="store_true", help="Also run official solver function solve_<id>(I) and compare vs interpreter")
    parser.add_argument("--use_solvers", action="store_true", help="Bypass interpreter and use official solver solve_<id>(I)")
    parser.add_argument("--only", type=str, default=None, help="Comma-separated list of task ids to run (e.g., 05269061,0dfd9992)")
    args = parser.parse_args()

    try:
        import arckit
        import arckit.vis as vis
        import numpy as np
    except Exception as e:
        print("arckit is required (pip install arc-kit)")
        print(e)
        sys.exit(1)

    project_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(project_root))
    from neural_lark.arc_interpreter import interpret_program
    # Prepare dynamic import of third_party solvers if requested
    solvers_mod = None
    if args.compare_solvers:
        arc_dsl_dir = project_root / "third_party" / "arc-dsl"
        if str(arc_dsl_dir) not in sys.path:
            sys.path.insert(0, str(arc_dsl_dir))
        solvers_path = arc_dsl_dir / "solvers.py"
        spec = importlib.util.spec_from_file_location("arc_solvers", solvers_path)
        if spec and spec.loader:
            solvers_mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(solvers_mod)  # type: ignore
        else:
            print("WARN: Could not import solvers.py for comparison")

    oracle_map = load_oracle_map(Path(args.oracle_jsonl))
    train_set, _ = arckit.load_data(args.dataset)
    all_tasks = list(train_set)
    tasks = all_tasks
    if args.only:
        only_ids = {tid.strip().lower() for tid in args.only.split(',') if tid.strip()}
        tasks = [t for t in tasks if get_task_id(t) in only_ids]
    if args.limit is not None:
        tasks = tasks[: args.limit]

    total_pairs = 0
    total_correct = 0
    matched_tasks = 0
    skipped_tasks = 0
    tasks_passed = 0

    for idx, task in enumerate(tasks, 1):
        tid = get_task_id(task)
        program = oracle_map.get(tid)
        if program is None:
            skipped_tasks += 1
            print(f"[{idx}] {tid}: skipped (no oracle program)")
            continue

        task_pairs = 0
        task_correct = 0
        for split_name, pairs in (("train", task.train), ("test", task.test)):
            for j, (inp, out) in enumerate(pairs):
                try:
                    if args.use_solvers and solvers_mod is not None:
                        fn = getattr(solvers_mod, f"solve_{tid}", None)
                        if callable(fn):
                            pred = fn(tuple(tuple(r) for r in inp.tolist()))
                        else:
                            raise RuntimeError("solver function not found")
                    else:
                        pred, _ = interpret_program(program, tuple(tuple(r) for r in inp.tolist()), project_root)
                    ok = (pred == tuple(tuple(r) for r in out.tolist()))
                except Exception as e:
                    ok = False
                    pred = None
                    if args.show_failures:
                        print(f"[{idx}:{tid}] {split_name}[{j}]: EXC {type(e).__name__}: {e}")
                solver_ok = None
                if args.compare_solvers and solvers_mod is not None:
                    fn_name = f"solve_{tid}"
                    fn = getattr(solvers_mod, fn_name, None)
                    if callable(fn):
                        try:
                            solver_pred = fn(tuple(tuple(r) for r in inp.tolist()))
                            solver_ok = (solver_pred == tuple(tuple(r) for r in out.tolist()))
                        except Exception as e:
                            solver_ok = False
                            if args.show_failures:
                                print(f"[{idx}:{tid}] {split_name}[{j}]: SOLVER_EXC {type(e).__name__}: {e}")
                task_pairs += 1
                total_pairs += 1
                if ok:
                    task_correct += 1
                    total_correct += 1
                # By default, print per-example only on failure (or solver mismatch when comparing)
                should_print_example = (not ok) or (args.compare_solvers and solver_ok is not None and not solver_ok)
                if should_print_example:
                    mark = "✓" if ok else "✗"
                    if solver_ok is not None:
                        mark = mark + ("|S✓" if solver_ok else "|S✗")
                    print(f"[{idx}:{tid}] {split_name}[{j}]: {mark}")
                    if args.show_failures and not ok and pred is not None:
                        print("Program:", program)
                        print("Input:")
                        vis.print_grid(inp)
                        print("Expected:")
                        vis.print_grid(out)
                        print("Got:")
                        vis.print_grid(np.array(pred))
                        print()

        matched_tasks += 1
        pct = 100.0 * task_correct / max(task_pairs, 1)
        if task_correct == task_pairs and task_pairs > 0:
            tasks_passed += 1
        # Color the per-task summary line: green if 100%, else red
        GREEN = "\033[92m"
        RED = "\033[91m"
        RESET = "\033[0m"
        line = f"Task {idx} ({tid}): {task_correct}/{task_pairs} = {pct:.1f}%"
        if task_pairs > 0 and task_correct == task_pairs:
            print(f"{GREEN}{line}{RESET}")
        else:
            print(f"{RED}{line}{RESET}")

    considered = len(tasks)
    dataset_total = len(all_tasks)
    print(f"Tasks considered: {considered}/{dataset_total}")
    print(f"Tasks ran: {matched_tasks}")
    print(f"Tasks skipped: {skipped_tasks}")
    tasks_pct = (100.0 * tasks_passed / max(considered, 1))
    print(f"Tasks passed: {tasks_passed}/{considered} = {tasks_pct:.2f}%")


if __name__ == "__main__":
    main()


