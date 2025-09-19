#!/usr/bin/env python3
import os
import sys
import json
import re
import argparse
from pathlib import Path


def _ensure_project_on_path():
    proj_root = Path(__file__).resolve().parent.parent
    if str(proj_root) not in sys.path:
        sys.path.insert(0, str(proj_root))
    return proj_root


def _load_arc_tasks(dataset_name: str):
    try:
        import arckit
    except Exception as e:
        print("Please install arckit to load ARC-AGI data (e.g., pip install arc-kit).\n", e)
        sys.exit(1)
    train_set, _ = arckit.load_data(dataset_name)
    tasks = list(train_set)
    def get_id(task):
        tid = getattr(task, 'id', None) or getattr(task, 'uid', None) or getattr(task, 'name', None)
        if tid is None and hasattr(task, 'path'):
            tid = Path(task.path).stem
        return tid
    ids = [get_id(t) for t in tasks]
    return ids


def _extract_solver_programs(solvers_path: Path):
    code = solvers_path.read_text()
    # Build mapping id -> program with variables renumbered to sequential x1..xN
    mapping = {}
    func_iter = re.finditer(r"^def\s+solve_([0-9a-f]+)\(I\):\n", code, flags=re.MULTILINE)
    for m in func_iter:
        func_id = m.group(1)
        start = m.end()
        next_m = re.search(r"^def\s+solve_[0-9a-f]+\(I\):\n", code[start:], flags=re.MULTILINE)
        block = code[start: start + next_m.start()] if next_m else code[start:]
        # Collect (lhs, rhs) assignments in order
        assigns = []
        for raw in block.splitlines():
            line = raw.strip()
            if not line or line.startswith('return '):
                continue
            if '#' in line:
                line = line.split('#', 1)[0].strip()
            if '=' in line:
                lhs, rhs = line.split('=', 1)
                lhs = lhs.strip()
                rhs = rhs.strip()
                assigns.append((lhs, rhs))
        if not assigns:
            continue
        # Renumber variables so references are to already-defined sequential names
        # Build mapping from original var names (e.g., x6) to new ones (x1, x2, ...)
        var_map = {}
        seq = 0
        renumbered_exprs = []
        var_token = re.compile(r"\bx\d+\b")
        for lhs, rhs in assigns:
            seq += 1
            new_var = f"x{seq}"
            var_map[lhs] = new_var
            # Replace variable references in rhs using current var_map
            def repl(tok: re.Match) -> str:
                name = tok.group(0)
                return var_map.get(name, name)
            rhs_renum = var_token.sub(repl, rhs)
            renumbered_exprs.append(rhs_renum)
        # Convert into assignment form with sequential x1..xN and final O if detectable
        stmts = []
        for i, rhs in enumerate(renumbered_exprs, start=1):
            stmts.append(f"x{i} = {rhs}")
        # Heuristic: if there is any occurrence of 'return xK', attach final assignment to O
        mret = re.search(r"return\s+(x\d+)", block)
        if mret:
            stmts.append(f"O = {mret.group(1)}")
        program = ' ## '.join(stmts)
        mapping[func_id] = program
    return mapping


def _compute_oracle_bnf(program: str, proj_root: Path) -> str:
    from minEarley.parser import EarleyParser
    from neural_lark.lark_utils import gen_min_lark, lark2bnf

    # Load the canonical grammar as text so we can optionally remove 'hocall'
    arc_lark_path = proj_root / 'grammars' / 'arc_1.lark'
    with open(arc_lark_path, 'r') as f:
        arc_lark_str = f.read()

    # Heuristic: if the program does NOT appear to use higher-order calls
    # (e.g., calling a variable like x1(...), or a parenthesized expression (...)(...)),
    # then remove 'hocall' from the grammar for extraction so concrete function
    # productions like 'objects(...)' are preferred and preserved in the minimal grammar.
    uses_ho_var_call = re.search(r"\bx\d+\s*\(", program) is not None
    uses_paren_callee = re.search(r"\)\s*\(", program) is not None

    if not (uses_ho_var_call or uses_paren_callee):
        # Remove ' | hocall' from expr alternatives (robust to spacing)
        arc_lark_str = re.sub(r"(\n\s*\|\s*)hocall(\s*\n)", r"\2", arc_lark_str)
        arc_lark_str = re.sub(r"(:\s*)([^\n]*?)\bhocall\b\s*\|\s*", r"\1\2", arc_lark_str)
        arc_lark_str = re.sub(r"(\n[^\n]*?)\s*\|\s*hocall\s*$", r"\1", arc_lark_str, flags=re.MULTILINE)

        # Remove the 'hocall:' production block entirely (single or multi-line)
        arc_lark_str = re.sub(
            r"\n\s*hocall\s*:\s*[^\n]*\n(?:\s*\|[^\n]*\n)*",
            "\n",
            arc_lark_str,
        )

    # Build parser from possibly-modified grammar string
    parser = EarleyParser(arc_lark_str, start=['start'], keep_all_tokens=True)
    lark_str = gen_min_lark(program, parser)
    return lark2bnf(lark_str)


def _rewrite_var_calls_to_apply(program: str) -> str:
    # No-op: grammar supports higher-order calls directly now
    return program


def main():
    _ensure_project_on_path()
    parser = argparse.ArgumentParser(description="Build ARC oracle dataset: ARC-AGI tasks annotated with solver program and oracle minimal grammar")
    parser.add_argument("--dataset", type=str, default="arcagi")
    parser.add_argument("--solvers", type=str, default="third_party/arc-dsl/solvers.py")
    parser.add_argument("--out_dir", type=str, default="data/arc/oracle")
    args = parser.parse_args()

    proj_root = Path(__file__).resolve().parent.parent
    out_dir = proj_root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    ids = _load_arc_tasks(args.dataset)
    solver_map = _extract_solver_programs(proj_root / args.solvers)

    jsonl_path = out_dir / 'arc_oracle.jsonl'
    src_path = out_dir / 'train.src'
    tgt_path = out_dir / 'train.tgt'
    gram_path = out_dir / 'train.grammar'

    n_total = 0
    n_matched = 0
    n_success = 0
    n_failed = 0
    failures = []
    with open(jsonl_path, 'w') as jf, open(src_path, 'w') as fs, open(tgt_path, 'w') as ft, open(gram_path, 'w') as fg:
        for tid in ids:
            n_total += 1
            # ARC IDs are 8-hex lowercase in solvers; normalize
            key = str(tid).lower()
            if key in solver_map:
                n_matched += 1
                program_raw = solver_map[key]
                try:
                    oracle_bnf = _compute_oracle_bnf(program_raw, proj_root)
                    n_success += 1
                except Exception as e:
                    # If grammar extraction fails, skip this task
                    n_failed += 1
                    failures.append({"id": key, "error": str(e)})
                    continue
                rec = {"id": key, "program": program_raw, "oracle_bnf": oracle_bnf}
                jf.write(json.dumps(rec) + "\n")
                fs.write(f"ARC task solve_{key}\n")
                ft.write(program_raw + "\n")
                fg.write(oracle_bnf.replace('\n', ' ') + "\n")

    # Write a failures file for inspection if any
    if failures:
        fail_path = out_dir / 'arc_oracle.failures.jsonl'
        with open(fail_path, 'w') as ff:
            for rec in failures:
                ff.write(json.dumps(rec) + "\n")

    print(f"ARC-AGI tasks: {n_total}, matched with solvers: {n_matched}")
    print(f"Oracle grammar success: {n_success}, failed: {n_failed}")
    print(f"Wrote: {jsonl_path}, {src_path}, {tgt_path}, {gram_path}")


if __name__ == "__main__":
    main()


