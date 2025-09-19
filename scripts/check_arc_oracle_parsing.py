import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Tuple, Optional

# Ensure project root is on sys.path when running directly
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from minEarley.parser import EarleyParser
from minEarley.earley_exceptions import UnexpectedInput


def bnf2lark(grammar: str) -> str:
    """Convert BNF (::=) to Lark (:) format."""
    return grammar.replace(" ::= ", " : ")


def decorate_grammar(grammar: str) -> str:
    """Append common terminals and whitespace ignore directives."""
    grammar += "\n%import common.DIGIT"
    grammar += "\n%import common.LCASE_LETTER"
    grammar += "\n%import common.UCASE_LETTER"
    grammar += "\n%import common.WS"
    grammar += "\n%ignore WS"
    return grammar


def split_program(program: str) -> List[str]:
    """
    Split a program into individual statements.
    Supports both ' ## ' separator and newline-separated statements.
    """
    if " ## " in program:
        parts = [s for s in program.split(" ## ") if s.strip()]
    elif "\n" in program:
        parts = [s for s in program.split("\n") if s.strip()]
    else:
        parts = [program.strip()] if program.strip() else []
    return parts
def _extract_name_tokens_from_bnf(bnf: str) -> set[str]:
    tokens: set[str] = set()
    if not bnf:
        return tokens
    for m in re.finditer(r"NAME\s*::=([^\n]+)", bnf):
        seg = m.group(1)
        tokens.update(re.findall(r'"([^"]+)"', seg))
    return tokens


def _augment_oracle_bnf_with_assignments(bnf: str) -> str:
    """Wrap an oracle minimal BNF to require assignment statements.

    Transforms the root 'start' into 'expr', and prepends:
      start ::= stmt
      stmt ::= assignment
      assignment ::= target "=" expr
      target ::= "O" | xvars_from_NAME
    """
    if not bnf or '::=' not in bnf:
        return bnf
    tokens = _extract_name_tokens_from_bnf(bnf)
    xvars = sorted([t for t in tokens if re.fullmatch(r"x\d+", t)], key=lambda s: (len(s), s))
    target_alts = ['"O"'] + [f'"{x}"' for x in xvars]
    target_rule = "target ::= " + " | ".join(target_alts)
    expr_bnf = re.sub(r"^start\s*::=", "expr ::=", bnf, count=1, flags=re.MULTILINE)
    header = "\n".join([
        "start ::= stmt",
        "stmt ::= assignment",
        "assignment ::= target \"=\" expr",
        target_rule,
    ])
    return f"{header}\n{expr_bnf}"



def build_parser_from_oracle_bnf(oracle_bnf: str) -> EarleyParser:
    """
    Convert oracle BNF to a Lark grammar, decorate it, and build an Earley parser.
    """
    # Ensure the oracle grammar accepts assignment-form programs
    augmented = _augment_oracle_bnf_with_assignments(oracle_bnf)
    lark_str = bnf2lark(augmented)
    lark_str = decorate_grammar(lark_str)
    parser = EarleyParser(lark_str, start=["start"], keep_all_tokens=True)
    return parser


def _eval_tree(node, env):
    """Evaluate a parsed AST (minEarley Tree) against the ARC DSL env.

    Supported nonterminals: start, hocall, callee, variable, constant, arglist.
    """
    from minEarley.tree import Tree
    if not isinstance(node, Tree):
        # punctuation/token - ignore at this level
        return node

    t = node.data
    if t == 'start':
        return _eval_tree(node.children[0], env)
    if t == 'stmt' or t == 'assignment':
        # assignment: target '=' expr
        # children: target, '=', expr
        # Evaluate RHS and bind to target in env
        # target is either 'O' or XVAR terminal under 'target'
        # Evaluate RHS expression (child index 2)
        value = _eval_tree(node.children[2], env)
        # Extract target name
        tgt_node = node.children[0]
        # target -> ('O') or XVAR token
        if hasattr(tgt_node, 'children') and tgt_node.children:
            tok = tgt_node.children[0]
            name = getattr(tok, 'value', str(tok))
        else:
            name = 'O'
        env[name] = value
        return value
    if t == 'hocall':
        # callee '(' arglist ')'
        func = _eval_tree(node.children[0], env)
        args = _eval_tree(node.children[2], env)
        return func(*args)
    if t == 'callee':
        return _eval_tree(node.children[0], env)
    if t == 'variable':
        tok = node.children[0]
        name = getattr(tok, 'value', str(tok))
        return env[name]
    if t == 'constant':
        tok = node.children[0]
        name = getattr(tok, 'value', str(tok))
        return env[name]
    if t in ('bool', 'digit', 'neg_digit', 'direction', 'vector', 'dims'):
        tok = node.children[0]
        name = getattr(tok, 'value', str(tok))
        return env[name]
    if t == 'arglist':
        args = []
        for ch in node.children:
            if hasattr(ch, 'data') and ch.data in ('variable', 'constant', 'hocall', 'callee'):
                args.append(_eval_tree(ch, env))
        return args
    # Support concrete function rules (e.g., 'objects', 'fill', ...)
    # If the node label matches a callable in env, treat children with data as argument expressions.
    try:
        fn = env.get(str(t))
    except Exception:
        fn = None
    if callable(fn):
        args = []
        for ch in node.children:
            if hasattr(ch, 'data'):
                args.append(_eval_tree(ch, env))
        return fn(*args)

    # Fallback: evaluate first child with structure
    for ch in node.children:
        if hasattr(ch, 'data'):
            return _eval_tree(ch, env)
    return None


def interpret_with_oracle(oracle_bnf: str, program: str, input_grid, project_root: str):
    """Interpret program by parsing with oracle_bnf and evaluating the parse trees sequentially."""
    # Build DSL environment
    sys.path.insert(0, project_root)
    from neural_lark.arc_interpreter import build_env, split_statements
    env = build_env(Path(project_root))
    env['I'] = input_grid

    parser = build_parser_from_oracle_bnf(oracle_bnf)
    last = None
    for idx, stmt in enumerate(split_statements(program), start=1):
        tree = parser.parse(stmt)
        last = _eval_tree(tree, env)
        env[f"x{idx}"] = last
    return last


def check_parse(oracle_bnf: str, program: str) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Try parsing each statement in program using the oracle BNF.
    Returns (ok, failing_stmt, error_message).
    """
    try:
        parser = build_parser_from_oracle_bnf(oracle_bnf)
    except Exception as e:
        return False, None, f"Failed to build parser from oracle_bnf: {e}"

    statements = split_program(program)
    for stmt in statements:
        try:
            parser.parse(stmt)
        except UnexpectedInput as e:
            return False, stmt, f"UnexpectedInput: {str(e)}"
        except Exception as e:
            return False, stmt, f"Parse error: {e}"
    return True, None, None


def main():
    ap = argparse.ArgumentParser(description="Check if oracle_bnf parses program for each JSONL line.")
    ap.add_argument("--file", default="data/arc/oracle/arc_oracle.jsonl", help="Path to arc_oracle.jsonl")
    ap.add_argument("--max", type=int, default=None, help="Max lines to check")
    ap.add_argument("--verbose", action="store_true", help="Print successes as well")
    ap.add_argument("--dataset", type=str, default="arcagi", help="arckit dataset name (e.g., arcagi)")
    args = ap.parse_args()

    args = ap.parse_args()

    # arckit is mandatory
    try:
        import arckit
    except Exception as e:
        print("arckit is required (pip install arc-kit)")
        print(e)
        sys.exit(1)

    total = 0
    both_pass = 0
    both_fail = 0
    mismatch_count = 0

    # Load arckit dataset for real grids
    tasks_by_id = {}
    train_set, _ = arckit.load_data(args.dataset)
    train_list = list(train_set)
    dataset_total = len([line for line in open(args.file, 'r') if line.strip()])
    for task in train_list:
        tid = getattr(task, 'id', None) or getattr(task, 'uid', None) or getattr(task, 'name', None)
        if tid is None and hasattr(task, 'path'):
            tid = Path(task.path).stem
        if tid is not None:
            tasks_by_id[str(tid).lower()] = task

    with open(args.file, "r") as f:
        for lineno, line in enumerate(f, start=1):
            if args.max is not None and total >= args.max:
                break
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Line {lineno}: JSON decode error: {e}")
                fail_count += 1
                total += 1
                continue

            ex_id = obj.get("id", f"line-{lineno}")
            program = obj.get("program")
            oracle_bnf = obj.get("oracle_bnf")

            if program is None or oracle_bnf is None:
                print(f"{ex_id}: missing required keys (program/oracle_bnf)")
                fail_count += 1
                total += 1
                continue

            ok, bad_stmt, err = check_parse(oracle_bnf, program)
            total += 1
            if ok:
                # Use arckit pairs to assess pass/fail per interpreter, then color per-line
                project_root = os.path.abspath(os.path.join(THIS_DIR, ".."))
                sys.path.insert(0, project_root)
                from neural_lark.arc_interpreter import interpret_program

                tid = str(ex_id).lower()
                GREEN = "\033[92m"
                RED = "\033[91m"
                YELLOW = "\033[93m"
                RESET = "\033[0m"

                line_idx = total + 1

                if tid in tasks_by_id:
                    task = tasks_by_id[tid]
                    pairs = list(task.train) + list(task.test)
                    master_all_ok = True
                    oracle_all_ok = True
                    for (inp, out) in pairs:
                        grid = tuple(tuple(r) for r in inp.tolist()) if hasattr(inp, 'tolist') else inp
                        expected = tuple(tuple(r) for r in out.tolist()) if hasattr(out, 'tolist') else out
                        # Master: must run and match expected
                        try:
                            master_pred, _ = interpret_program(program, grid, Path(project_root))
                            if master_pred != expected:
                                master_all_ok = False
                        except Exception:
                            master_all_ok = False
                            master_pred = None
                        # Oracle: must run and match expected output for the same grid
                        try:
                            oracle_pred = interpret_with_oracle(oracle_bnf, program, grid, project_root)
                            if oracle_pred != expected:
                                oracle_all_ok = False
                        except Exception:
                            oracle_all_ok = False

                    if master_all_ok and oracle_all_ok:
                        color = GREEN
                        both_pass += 1
                    elif (not master_all_ok) and (not oracle_all_ok):
                        color = RED
                        both_fail += 1
                    else:
                        color = YELLOW
                        mismatch_count += 1

                    m_mark = "✓" if master_all_ok else "✗"
                    o_mark = "✓" if oracle_all_ok else "✗"
                    print(f"{color}[{line_idx}/{dataset_total}] {tid}: master={m_mark} oracle={o_mark}{RESET}")
                else:
                    # No matching arckit task; count as both fail
                    both_fail += 1
                    print(f"{RED}[{line_idx}/{dataset_total}] {ex_id}: master=✗ oracle=✗ (no arckit task){RESET}")
            else:
                # Parse failure at least
                line_idx = total + 1
                both_fail += 1
                print(f"\033[91m[{line_idx}/{dataset_total}] {ex_id}: master=✗ oracle=✗ (parse/build fail)\033[0m")

    print("\n=== Summary ===")
    print(f"Checked:   {total}")
    print(f"Both pass: {both_pass}")
    print(f"Both fail: {both_fail}")
    print(f"Mismatch:  {mismatch_count}")


if __name__ == "__main__":
    main()


