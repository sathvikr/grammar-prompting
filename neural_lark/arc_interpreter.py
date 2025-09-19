import os
import sys
import ast
from typing import Any, Dict, List, Tuple
from pathlib import Path


def _ensure_arc_dsl_on_path(project_root: Path) -> None:
    arc_dsl_dir = project_root / "third_party" / "arc-dsl"
    if str(arc_dsl_dir) not in sys.path:
        sys.path.insert(0, str(arc_dsl_dir))


def build_env(project_root: Path) -> Dict[str, Any]:
    """Build a restricted evaluation environment from arc-dsl dsl.py and constants.py."""
    _ensure_arc_dsl_on_path(project_root)
    import dsl  # type: ignore
    import constants  # type: ignore

    env: Dict[str, Any] = {}
    # functions
    for name in dir(dsl):
        if name.startswith("_"):
            continue
        obj = getattr(dsl, name)
        if callable(obj):
            env[name] = obj
    # constants
    for name in dir(constants):
        if name.startswith("_"):
            continue
        env[name] = getattr(constants, name)
    # also allow basic tuple/ frozenset constructors through environment-only symbols
    env["frozenset"] = frozenset
    env["tuple"] = tuple
    def _is_grid(x: Any) -> bool:
        return isinstance(x, tuple) and len(x) > 0 and isinstance(x[0], (tuple, list))

    def _is_index_pair(x: Any) -> bool:
        return isinstance(x, tuple) and len(x) == 2 and all(isinstance(v, int) for v in x)

    def _is_patch(container: Any) -> bool:
        if not isinstance(container, frozenset):
            return False
        if len(container) == 0:
            return True
        sample = next(iter(container))
        return _is_index_pair(sample)

    def _is_set_of_pieces(container: Any) -> bool:
        if not isinstance(container, frozenset) or len(container) == 0:
            return False
        sample = next(iter(container))
        if not isinstance(sample, frozenset) or len(sample) == 0:
            return False
        inner = next(iter(sample))
        return _is_index_pair(inner)

    # Keep dsl semantics exactly; do not override apply/rbind/interval
    return env


def split_statements(program: str) -> List[str]:
    if " ## " in program:
        parts = [p.strip() for p in program.split(" ## ") if p.strip()]
    elif "\n" in program:
        parts = [p.strip() for p in program.split("\n") if p.strip()]
    else:
        parts = [program.strip()] if program.strip() else []
    return parts


def _parse_expr(expr: str) -> ast.AST:
    return ast.parse(expr, mode="eval")


def _eval_ast(node: ast.AST, env: Dict[str, Any]) -> Any:
    if isinstance(node, ast.Expression):
        return _eval_ast(node.body, env)
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        if node.id not in env:
            raise NameError(f"undefined name: {node.id}")
        return env[node.id]
    if isinstance(node, ast.Tuple):
        return tuple(_eval_ast(e, env) for e in node.elts)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        val = _eval_ast(node.operand, env)
        return -val
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.FloorDiv)):
        left = _eval_ast(node.left, env)
        right = _eval_ast(node.right, env)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        # FloorDiv
        return left // right
    if isinstance(node, ast.Call):
        func = _eval_ast(node.func, env)
        if not callable(func):
            raise TypeError(f"attempting to call non-callable: {func}")
        args = [_eval_ast(a, env) for a in node.args]
        kwargs = {kw.arg: _eval_ast(kw.value, env) for kw in node.keywords}
        return func(*args, **kwargs)
    # Disallow everything else
    raise ValueError(f"Unsupported syntax: {ast.dump(node, include_attributes=False)}")


def eval_expression(expr: str, env: Dict[str, Any]) -> Any:
    node = _parse_expr(expr)
    return _eval_ast(node, env)


def _rewrite_var_calls_to_apply(program: str) -> str:
    """Rewrite higher-order call sugar into apply/papply to match interpreter expectations.

    - x2(I) -> apply(x2, I)
    - x2(a,b) -> papply(x2, a, b)
    - (expr)(args) -> apply(expr, args)
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


def _describe_value(value: Any) -> str:
    try:
        if callable(value):
            name = getattr(value, "__name__", None) or value.__class__.__name__
            return f"<fn:{name}>"
        if isinstance(value, tuple):
            # Could be grid (tuple of rows) or vector/tuple
            if value and isinstance(value[0], (tuple, list)):
                h = len(value)
                w = len(value[0]) if value[0] is not None and hasattr(value[0], '__len__') else None
                return f"<grid {h}x{w}>"
            return f"<tuple len={len(value)}>"
        if isinstance(value, frozenset):
            return f"<frozenset size={len(value)}>"
        if isinstance(value, (int, float)):
            return f"<num {value}>"
        return f"<{type(value).__name__}>"
    except Exception:
        return f"<{type(value).__name__}>"


def interpret_program(program: str, input_grid: Any, project_root: Path, debug: bool = False) -> Tuple[Any, Dict[str, Any]]:
    """Interpret an ARC DSL program string against an input grid.

    Accepts assignment-form statements: 'xN = expr' or 'O = expr'.
    Multiple statements can be separated by ' ## ' or newlines.
    Returns (last_assigned_value, env).
    """
    env = build_env(project_root)
    env["I"] = input_grid
    if debug:
        env["_trace"] = []
    stmts = split_statements(program)
    last = None
    for idx, stmt in enumerate(stmts, start=1):
        if not stmt:
            continue
        # Expect 'target = expr'
        if '=' not in stmt:
            # Back-compat: treat as expression assigned to x{idx}
            target_name = f"x{idx}"
            rhs = stmt
        else:
            lhs, rhs = stmt.split('=', 1)
            target_name = lhs.strip()
            rhs = rhs.strip()
        try:
            value = eval_expression(rhs, env)
            env[target_name] = value
            last = value
            if debug:
                env["_trace"].append({"i": idx, "stmt": stmt, "target": target_name, "type": _describe_value(value)})
        except Exception as e:
            if debug:
                env.setdefault("_trace", []).append({"i": idx, "stmt": stmt, "error": f"{type(e).__name__}: {e}"})
            raise
    return last, env


