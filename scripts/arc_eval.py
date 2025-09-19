#!/usr/bin/env python3
import sys
import os
import json
import argparse
from typing import Optional, List, Set, Tuple, Dict
from pathlib import Path
import re
import numpy as np
from datetime import datetime, timezone


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Default dump locations (single files)
DEFAULT_PROGRAMS_DUMP = os.path.join(PROJECT_ROOT, "log", "oracle_programs.jsonl")
DEFAULT_PROMPTS_DUMP = os.path.join(PROJECT_ROOT, "log", "oracle_prompts.jsonl")


def require_arckit():
    try:
        import arckit  # noqa: F401
    except Exception as e:
        print("arckit is required (pip install arc-kit)")
        print(e)
        sys.exit(1)


def load_tasks(dataset: str):
    import arckit
    train_set, _ = arckit.load_data(dataset)
    tasks_by_id = {}
    for task in train_set:
        tid = getattr(task, 'id', None) or getattr(task, 'uid', None) or getattr(task, 'name', None)
        if tid is None and hasattr(task, 'path'):
            tid = Path(task.path).stem
        if tid is not None:
            tasks_by_id[str(tid).lower()] = task
    return tasks_by_id


def iter_jsonl(jsonl_path: str):
    with open(jsonl_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            yield rec


def _load_oracle_map(jsonl_path: str) -> Dict[str, str]:
    mp: Dict[str, str] = {}
    try:
        with open(jsonl_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                rid = str(obj.get('id', '')).lower()
                bnf = obj.get('oracle') or obj.get('oracle_bnf')
                if rid and bnf:
                    mp[rid] = bnf
    except Exception:
        pass
    return mp


def build_oracle_parser_and_run(oracle_bnf: str, program: str, grid, project_root: str):
    from scripts.check_arc_oracle_parsing import interpret_with_oracle
    return interpret_with_oracle(oracle_bnf, program, grid, project_root)


def run_master(program: str, grid, project_root: Path):
    from neural_lark.arc_interpreter import interpret_program
    last, _ = interpret_program(program, grid, project_root)
    return _normalize_grid(last)

def _normalize_grid(result):
    """Convert predicted grid-like structures to tuple-of-tuples for stable equality.

    Handles numpy arrays and list/tuple of lists/tuples.
    """
    try:
        if hasattr(result, 'tolist'):
            result = result.tolist()
        if isinstance(result, (list, tuple)) and result and isinstance(result[0], (list, tuple, np.ndarray)):
            # Convert inner to tuples as well
            return tuple(tuple(r.tolist() if hasattr(r, 'tolist') else tuple(r) if isinstance(r, (list, tuple)) else r for r in row) for row in result) if False else tuple(tuple(c for c in row) for row in result)
        return result
    except Exception:
        return result


def _lark_to_bnf(grammar: str) -> str:
    return grammar.replace(" : ", " ::= ")

def _bnf_to_lark(grammar: str) -> str:
    return grammar.replace(" ::= ", " : ")

def _decorate_lark(grammar: str) -> str:
    grammar += "\n%import common.DIGIT"
    grammar += "\n%import common.LCASE_LETTER"
    grammar += "\n%import common.UCASE_LETTER"
    grammar += "\n%import common.WS"
    grammar += "\n%ignore WS"
    return grammar

def _split_program_stmts(program: str) -> List[str]:
    if " ## " in program:
        return [s for s in program.split(" ## ") if s.strip()]
    if "\n" in program:
        return [s for s in program.split("\n") if s.strip()]
    return [program.strip()] if program.strip() else []

_KNOWN_CONSTANTS = {
    # booleans
    "T", "F",
    # numbers
    "ZERO", "ONE", "TWO", "THREE", "FOUR", "FIVE", "SIX", "SEVEN", "EIGHT", "NINE", "TEN",
    "NEG_ONE", "NEG_TWO",
    # directions & specials
    "DOWN", "RIGHT", "UP", "LEFT",
    "ORIGIN", "UNITY", "NEG_UNITY", "UP_RIGHT", "DOWN_LEFT",
    # sizes
    "ZERO_BY_TWO", "TWO_BY_ZERO", "TWO_BY_TWO", "THREE_BY_THREE",
}

_QUOTE_TOKEN_RE = re.compile(r'"([A-Za-z0-9_]+)"')

def _extract_name_tokens_from_bnf(bnf: str) -> Set[str]:
    tokens = set()
    if not bnf:
        return tokens
    for m in re.finditer(r'NAME\s*::=([^\n]+)', bnf):
        seg = m.group(1)
        tokens.update(re.findall(r'"([^"]+)"', seg))
    return tokens

def sanitize_program_text(text: str, oracle_bnf: Optional[str], strip_fences: bool = True, dequote: bool = True) -> str:
    s = text
    if strip_fences:
        # Remove leading code fence with optional language tag
        s = re.sub(r'^\s*```[A-Za-z0-9_-]*\s*\n', '', s)
        # Remove trailing fence
        s = re.sub(r'\n```\s*$', '', s)
        # Remove stray triple-backticks
        s = s.replace('```', '')

    if dequote:
        allowed = set(_KNOWN_CONSTANTS)
        allowed.update(_extract_name_tokens_from_bnf(oracle_bnf or ''))

        def _replace(m: re.Match) -> str:
            tok = m.group(1)
            if tok in allowed or re.fullmatch(r'x\d+', tok):
                return tok
            return m.group(0)

        s = _QUOTE_TOKEN_RE.sub(_replace, s)

    s = s.strip()

    # Fallback extraction if the model produced prose: try to pull assignment lines
    if not s or ('=' not in s and ' ## ' not in s):
        lines = [ln.strip() for ln in text.splitlines()]
        assign_lines = [ln for ln in lines if '=' in ln and not ln.startswith('#')]
        # Remove trailing punctuation and prose
        cleaned = []
        for ln in assign_lines:
            # keep up to first inline comment
            if '#' in ln:
                ln = ln.split('#', 1)[0].strip()
            # drop bullet markers
            ln = ln.lstrip('-').strip()
            # simple guard: looks like target = expr
            if re.match(r"^(O|x\d+)\s*=\s*.+", ln):
                cleaned.append(ln)
        if cleaned:
            s = ' ## '.join(cleaned)

    return s
def _augment_oracle_bnf_with_assignments(bnf: str) -> str:
    """Wrap an oracle minimal BNF to require assignment statements.

    Transforms the root 'start' into 'expr', and prepends:
      start ::= stmt
      stmt ::= assignment
      assignment ::= target "=" expr
      target ::= "O" | xvars_from_NAME

    xN variables are derived from the NAME token list inside the oracle BNF.
    """
    if not bnf or '::=' not in bnf:
        return bnf
    # Extract tokens and identify x-variables present in this oracle
    tokens = _extract_name_tokens_from_bnf(bnf)
    xvars = sorted([t for t in tokens if re.fullmatch(r"x\d+", t)], key=lambda s: (len(s), s))
    # Build target alternatives
    target_alts = ['"O"'] + [f'"{x}"' for x in xvars]
    target_rule = "target ::= " + " | ".join(target_alts)
    # Rename start ::= ... to expr ::= ... (only first definition)
    expr_bnf = re.sub(r"^start\s*::=", "expr ::=", bnf, count=1, flags=re.MULTILINE)
    header = "\n".join([
        "start ::= stmt",
        "stmt ::= assignment",
        "assignment ::= target \"=\" expr",
        target_rule,
    ])
    return f"{header}\n{expr_bnf}"



def build_prompt(rec, use_oracle: bool, master_lark_path: str, fewshot_text: Optional[str] = None, retrieval_text: Optional[str] = None) -> str:
    header = (
        "You are an expert ARC programmer. An ARC programming task consists of a 2D input grid and a 2D output grid. Each cell in a grid contains a number from ZERO (0) to TEN (10). There always exists a sequence of transformations that maps an input ARC grid to an output ARC grid.\n"
    )
    fewshot_block = ""
    if fewshot_text:
        fewshot_block = f"\nHere are some ARC input-output grid pairs that follow the same sequence of transformations:\n{fewshot_text}\n"
    retrieval_block = ""
    if retrieval_text:
        retrieval_block = f"\nHere are 16 similar tasks (retrieved by BM25). For each, we show its minimal BNF and the corresponding ARC program:\n{retrieval_text}\n"
    bnf_header = "Your output ARC program must conform to this BNF grammar:\n"
    if use_oracle:
        # Prefer explicit 'oracle' field; fall back to 'oracle_bnf'
        grammar = rec.get('oracle') or rec.get('oracle_bnf') or ""
        # Enforce assignment-form programs in the prompt by wrapping the oracle minimal grammar
        bnf = _augment_oracle_bnf_with_assignments(grammar)
    else:
        with open(master_lark_path, 'r') as f:
            lark_str = f.read()
        bnf = _lark_to_bnf(lark_str)
    delimiter = (
        "\n\nOutput the ARC program based on the BNF grammar rules:\n"
        "Respond with only the program, no explanation, no blank lines, and no code fences.\n"
        "If multiple statements, separate them with ' ## '.\n"
        # "You must use ALL of the primitive functions present in this BNF at least once.\n"
    )
    return f"{header}{fewshot_block}{retrieval_block}{bnf_header}{bnf}{delimiter}"


def prompt_program(rec, use_oracle: bool, engine: str, temperature: float, master_lark_path: str, disable_cache: bool,
                  constrain_parse: bool = False, max_retries: int = 2,
                  strip_fences: bool = True, dequote: bool = True,
                  oracle_map: Optional[Dict[str, str]] = None,
                  num_retrieved: int = 16):
    from neural_lark.llm_interface import setup_llm
    from minEarley.parser import EarleyParser
    llm = setup_llm(engine)

    fewshot_text = None
    retrieval_text = None
    # Build few-shot block using arckit train pairs for this id
    try:
        import arckit
        rid = str(rec.get('id','')).lower()
        # Attempt to reuse dataset from FLAGS if set via environment; otherwise default to arcagi
        dataset = os.environ.get('ARC_DATASET_NAME', 'arcagi')
        tasks_by_id = load_tasks(dataset)
        task = tasks_by_id.get(rid)
        if task is not None:
            lines = []
            for idx, (inp, out) in enumerate(list(task.train)):
                ijson = inp.tolist() if hasattr(inp, 'tolist') else inp
                ojson = out.tolist() if hasattr(out, 'tolist') else out
                lines.append(f"- Pair {idx+1}:\n  Input:\n  {ijson}\n  Output:\n  {ojson}\n")
            fewshot_text = "\n".join(lines)
    except Exception:
        fewshot_text = None

    # Build BM25 retrieval block: similar tasks with oracle_bnf and solver program
    try:
        from neural_lark.dataset import _extract_arc_examples_from_solvers
        from neural_lark.retriever import BM25Retriever
        # Merge BNFs from canonical file and passed-in map so most tasks have BNFs
        merged_oracle_map: Dict[str, str] = {}
        try:
            canonical_jsonl = os.path.join(PROJECT_ROOT, 'data', 'arc', 'oracle', 'arc_oracle.jsonl')
            merged_oracle_map.update(_load_oracle_map(canonical_jsonl))
        except Exception:
            pass
        if oracle_map:
            merged_oracle_map.update(oracle_map)

        # Build examples from solvers
        solvers_path = os.path.join(PROJECT_ROOT, 'third_party', 'arc-dsl', 'solvers.py')
        examples = _extract_arc_examples_from_solvers(solvers_path)

        def _rid_from_source(src: str) -> str:
            m = re.search(r'solve_([0-9a-f]+)', src)
            return m.group(1) if m else ''

        # Keep only examples that have a known BNF
        examples_with_bnf = [ex for ex in examples if merged_oracle_map.get(_rid_from_source(ex.source), '')]
        if examples_with_bnf:
            # Normalize BNF to single-line for BM25
            def _norm_bnf(txt: str) -> str:
                return re.sub(r"\s+", " ", txt).strip()

            def _ex2doc(ex_obj):
                rid_local = _rid_from_source(ex_obj.source)
                if not rid_local:
                    # Query path: use the provided source string (expected to be BNF)
                    return ex_obj.source
                bnf_local = merged_oracle_map.get(rid_local, '')
                return _norm_bnf(bnf_local)

            bm25 = BM25Retriever(examples_with_bnf, ex2doc=_ex2doc)
            # Build query from this record's oracle_bnf only
            bnf_q = rec.get('oracle') or rec.get('oracle_bnf') or ''
            if not bnf_q:
                raise ValueError('No oracle_bnf available for retrieval query')
            # Use the BNF string directly as the query document
            retrieved, _ = bm25.retrieve_by_src(_norm_bnf(bnf_q), n=num_retrieved + 2)
            # Filter out the same task id if present; keep distinct 16 others
            cur_id = str(rec.get('id', '')).lower()
            filtered = []
            seen = set()
            for ex in retrieved:
                rid_local = _rid_from_source(ex.source)
                if not rid_local or rid_local == cur_id or rid_local in seen:
                    continue
                filtered.append(ex)
                seen.add(rid_local)
                if len(filtered) >= num_retrieved:
                    break
            # If nothing retrieved, fallback to 16 random examples with known BNF
            if not filtered:
                import random as _random
                fallback = _random.sample(examples_with_bnf, min(num_retrieved, len(examples_with_bnf)))
                filtered = fallback
            # Assemble lines for prompt
            lines = []
            # Load tasks to render input/output grids for each retrieved task
            try:
                dataset = os.environ.get('ARC_DATASET_NAME', 'arcagi')
                tasks_by_id_for_ret = load_tasks(dataset)
            except Exception:
                tasks_by_id_for_ret = {}

            for ex in filtered:
                rid = _rid_from_source(ex.source)
                bnf_show = merged_oracle_map.get(rid, None)
                program_text = ex.target
                # Add input/output grids from arckit for this rid
                io_block = ""
                task_obj = tasks_by_id_for_ret.get(rid) if rid else None
                if task_obj is not None:
                    try:
                        pair_lines = []
                        for idx_io, (inp_io, out_io) in enumerate(list(task_obj.train)):
                            ijson = inp_io.tolist() if hasattr(inp_io, 'tolist') else inp_io
                            ojson = out_io.tolist() if hasattr(out_io, 'tolist') else out_io
                            pair_lines.append(f"  Input:\n  {ijson}\n  Output:\n  {ojson}")
                        if pair_lines:
                            io_block = "\n" + "\n".join(pair_lines)
                    except Exception:
                        io_block = ""

                if bnf_show:
                    lines.append(f"- Task {rid}:{io_block}\n  Oracle BNF:\n  {bnf_show}\n  Program:\n  {program_text}")
                else:
                    lines.append(f"- Task {rid}:{io_block}\n  Program:\n  {program_text}")
            retrieval_text = "\n".join(lines)
        else:
            retrieval_text = None
    except Exception:
        retrieval_text = None

    prompt = build_prompt(rec, use_oracle=use_oracle, master_lark_path=master_lark_path, fewshot_text=fewshot_text, retrieval_text=retrieval_text)
    # Build oracle parser if constraining
    parser = None
    if constrain_parse and use_oracle:
        oracle_bnf = rec.get('oracle') or rec.get('oracle_bnf')
        if oracle_bnf:
            lark_str = _bnf_to_lark(oracle_bnf)
            lark_str = _decorate_lark(lark_str)
            parser = EarleyParser(lark_str, start=["start"], keep_all_tokens=True)

    attempts = max(1, max_retries if constrain_parse else 1)
    last_text = None
    for _ in range(attempts):
        # Stop at double newline to keep single program
        responses = llm.sample_completions(prompt, temperature, stop_token="\n\n", num_completions=1, disable_cache=disable_cache)
        raw_text = responses[0].response_text
        text = sanitize_program_text(raw_text, rec.get('oracle') or rec.get('oracle_bnf'), strip_fences=strip_fences, dequote=dequote)
        last_text = text
        if not parser:
            return text, ("LLM" if disable_cache else "CACHE_OR_LLM"), prompt
        # validate parseability per-statement
        ok = True
        try:
            for stmt in _split_program_stmts(text):
                parser.parse(stmt)
        except Exception:
            ok = False
        if ok:
            return text, ("LLM" if disable_cache else "CACHE_OR_LLM"), prompt
    # If we reach here, return last attempt even if invalid
    return (last_text or ""), ("LLM" if disable_cache else "CACHE_OR_LLM"), prompt


def eval_line(rec, tasks_by_id, mode_master: bool, mode_oracle: bool, project_root: Path):
    ex_id = str(rec.get('id', '')).lower()
    program = rec.get('program')
    oracle_bnf = rec.get('oracle_bnf')

    task = tasks_by_id.get(ex_id)
    if task is None:
        return False, False, ex_id, 0, 0

    pairs = list(task.train) + list(task.test)
    master_ok = True if mode_master else None
    oracle_ok = True if mode_oracle else None

    for (inp, out) in pairs:
        grid = tuple(tuple(r) for r in inp.tolist()) if hasattr(inp, 'tolist') else inp
        expected = tuple(tuple(r) for r in out.tolist()) if hasattr(out, 'tolist') else out

        if mode_master and master_ok:
            try:
                pred = run_master(program, grid, project_root)
                if pred != expected:
                    master_ok = False
                    # Visualize the first failure for this task
                    try:
                        import arckit.vis as vis
                        print("[MASTER FAIL] Input:")
                        vis.print_grid(np.array(grid))
                        print("Expected:")
                        vis.print_grid(np.array(expected))
                        print("Predicted:")
                        vis.print_grid(np.array(pred))
                    except Exception:
                        pass
            except Exception:
                master_ok = False

        if mode_oracle and oracle_ok:
            try:
                pred = build_oracle_parser_and_run(oracle_bnf, program, grid, str(project_root))
                pred = _normalize_grid(pred)
                if pred != expected:
                    oracle_ok = False
                    try:
                        import arckit.vis as vis
                        print("[ORACLE FAIL] Input:")
                        vis.print_grid(np.array(grid))
                        print("Expected:")
                        vis.print_grid(np.array(expected))
                        print("Predicted:")
                        vis.print_grid(np.array(pred))
                    except Exception:
                        pass
            except Exception:
                oracle_ok = False

        # Early break if relevant modes already failed
        if (mode_master and master_ok is False) and (mode_oracle and oracle_ok is False):
            break

    total = len(pairs)
    ok_count = 0
    if mode_master and master_ok:
        ok_count += total
    if mode_oracle and oracle_ok:
        ok_count += total
    return master_ok, oracle_ok, ex_id, ok_count, total


def colorize(master_ok, oracle_ok, idx, total_lines, ex_id):
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    RESET = "\033[0m"

    m = None if master_ok is None else ("✓" if master_ok else "✗")
    o = None if oracle_ok is None else ("✓" if oracle_ok else "✗")

    # Determine color
    if master_ok is not None and oracle_ok is not None:
        if master_ok and oracle_ok:
            color = GREEN
        elif (not master_ok) and (not oracle_ok):
            color = RED
        else:
            color = YELLOW
    elif (master_ok is True) or (oracle_ok is True):
        color = GREEN
    else:
        color = RED if (master_ok is False) or (oracle_ok is False) else RESET

    left = f"[{idx}/{total_lines}] {ex_id}:"
    right = []
    if m is not None:
        right.append(f"master={m}")
    if o is not None:
        right.append(f"oracle={o}")
    return f"{color}{left} {' '.join(right)}{RESET}"


def main():
    parser = argparse.ArgumentParser(description="Evaluate ARC programs against arckit expected outputs using master and/or oracle interpreters")
    parser.add_argument("--file", default="data/arc/oracle/arc_oracle.jsonl")
    parser.add_argument("--dataset", default="arcagi")
    parser.add_argument("--master", action="store_true", help="Evaluate with master interpreter")
    parser.add_argument("--oracle", action="store_true", default=True, help="Evaluate with oracle_bnf interpreter (default)")
    parser.add_argument("--prompt", action="store_true", default=True, help="Generate programs via LLM prompt instead of using 'program' field (default)")
    parser.add_argument("--engine", type=str, default="openai/gpt-4o-mini", help="LLM engine spec platform/model")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=128)
    parser.add_argument("--freq_penalty", type=float, default=0.0)
    parser.add_argument("--llm_cache_dir", type=str, default="llm_cache")
    parser.add_argument("--disable_cache", action="store_true", default=True, help="Force live LLM calls (no cache)")
    parser.add_argument("--show_source", action="store_true", default=True, help="Show how the program was obtained (LLM/cache/fallback) (default)")
    parser.add_argument("--debug_prompt", action="store_true", help="Print prompt and any LLM exception when prompting fails")
    parser.add_argument("--constrain_parse", action="store_true", help="Retry prompting until the program parses under the oracle BNF")
    parser.add_argument("--max_retries", type=int, default=2, help="Max retries for constrained prompting")
    args = parser.parse_args()

    # Convenience: if no mode flags provided explicitly, defaults already set oracle=True
    # Convenience: use /tmp/arc3.jsonl if present and user didn't override the default file
    default_file = "data/arc/oracle/arc_oracle.jsonl"
    if args.file == default_file and os.path.exists("/tmp/arc3.jsonl"):
        args.file = "/tmp/arc3.jsonl"

    # Initialize LLM FLAGS for neural_lark.llm_interface
    try:
        from neural_lark.flags import FLAGS as NL_FLAGS
        NL_FLAGS.engine = args.engine
        NL_FLAGS.temperature = args.temperature
        NL_FLAGS.freq_penalty = args.freq_penalty
        NL_FLAGS.max_tokens = args.max_tokens
        NL_FLAGS.llm_cache_dir = args.llm_cache_dir
    except Exception:
        pass

    require_arckit()
    tasks_by_id = load_tasks(args.dataset)

    project_root = Path(__file__).resolve().parent.parent
    lines = [rec for rec in iter_jsonl(args.file)]
    total_lines = len(lines)

    both_pass = 0
    both_fail = 0
    mismatch = 0

    master_lark_path = os.path.join(PROJECT_ROOT, "grammars", "arc_1.lark")

    # If prompting, augment each record with generated program per requested mode
    if args.prompt:
        augmented = []
        # Precompute oracle_bnf map for retrieval display
        # Build oracle map from the same input JSONL to ensure retrieval has BNF text
        oracle_map: Dict[str, str] = _load_oracle_map(args.file)

        for _idx, rec in enumerate(lines, 1):
            new_rec = dict(rec)
            if args.oracle:
                try:
                    # Stream progress before prompting
                    print(f"[PROMPT][{_idx}/{total_lines}] {new_rec.get('id')}: prompting with oracle grammar...", flush=True)
                    prog, src, used_prompt = prompt_program(
                        rec,
                        use_oracle=True,
                        engine=args.engine,
                        temperature=args.temperature,
                        master_lark_path=master_lark_path,
                        disable_cache=args.disable_cache,
                        constrain_parse=args.constrain_parse,
                        max_retries=args.max_retries,
                        strip_fences=True,
                        dequote=True,
                        oracle_map=oracle_map,
                        num_retrieved=16,
                    )
                    new_rec['program'] = prog
                    new_rec['_prog_source'] = src
                    new_rec['_prompt_used'] = used_prompt
                    print(f"[PROMPT][{_idx}/{total_lines}] {new_rec.get('id')}: generated program (source={src}, len={len(prog)})", flush=True)
                except Exception as e:
                    if args.debug_prompt:
                        import traceback
                        print(f"[PROMPT ERROR][{new_rec.get('id')}] engine={args.engine} disable_cache={args.disable_cache}")
                        print("--- PROMPT START ---")
                        try:
                            print(build_prompt(rec, use_oracle=True, master_lark_path=master_lark_path))
                        except Exception:
                            print("<prompt build failed>")
                        print("--- PROMPT END ---")
                        print(type(e).__name__, e)
                        traceback.print_exc()
                    # leave as is; eval will fail accordingly
                    pass
            if args.master and not args.oracle:
                try:
                    print(f"[PROMPT][{_idx}/{total_lines}] {new_rec.get('id')}: prompting with master grammar...", flush=True)
                    # When only master requested, allow using full grammar prompt
                    prog, src, used_prompt = prompt_program(
                        rec,
                        use_oracle=False,
                        engine=args.engine,
                        temperature=args.temperature,
                        master_lark_path=master_lark_path,
                        disable_cache=args.disable_cache,
                        constrain_parse=False,
                        strip_fences=True,
                        dequote=True,
                        oracle_map=oracle_map,
                        num_retrieved=16,
                    )
                    new_rec['program'] = prog
                    new_rec['_prog_source'] = src
                    new_rec['_prompt_used'] = used_prompt
                    print(f"[PROMPT][{_idx}/{total_lines}] {new_rec.get('id')}: generated program (source={src}, len={len(prog)})", flush=True)
                except Exception as e:
                    if args.debug_prompt:
                        import traceback
                        print(f"[PROMPT ERROR][{new_rec.get('id')}] engine={args.engine} disable_cache={args.disable_cache}")
                        print("--- PROMPT START ---")
                        try:
                            print(build_prompt(rec, use_oracle=False, master_lark_path=master_lark_path))
                        except Exception:
                            print("<prompt build failed>")
                        print("--- PROMPT END ---")
                        print(type(e).__name__, e)
                        traceback.print_exc()
                    pass
            augmented.append(new_rec)
        lines = augmented

    # Always dump to default single files when prompting
    dumper = None
    prompt_dumper = None
    if args.prompt:
        os.makedirs(os.path.dirname(DEFAULT_PROGRAMS_DUMP), exist_ok=True)
        os.makedirs(os.path.dirname(DEFAULT_PROMPTS_DUMP), exist_ok=True)
        dumper = open(DEFAULT_PROGRAMS_DUMP, 'w')
        prompt_dumper = open(DEFAULT_PROMPTS_DUMP, 'w')

    for idx, rec in enumerate(lines, 1):
        master_ok, oracle_ok, ex_id, _, _ = eval_line(rec, tasks_by_id, args.master, args.oracle, project_root)
        # Tally
        if args.master and args.oracle:
            if master_ok and oracle_ok:
                both_pass += 1
            elif (master_ok is False) and (oracle_ok is False):
                both_fail += 1
            else:
                mismatch += 1
        # Stream line
        src_label = None
        if args.prompt and args.show_source:
            src_label = rec.get('_prog_source') or "FALLBACK"
        print(colorize(master_ok, oracle_ok, idx, total_lines, ex_id) + (f" src={src_label}" if src_label else ""), flush=True)

        if dumper is not None:
            obj = {
                'id': rec.get('id'),
                'program_generated': rec.get('program'),
                'source': rec.get('_prog_source'),
                'mode': 'oracle' if args.oracle else ('master' if args.master else 'unknown'),
                'generated_at': datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
            }
            import json as _json
            dumper.write(_json.dumps(obj) + "\n")
        if prompt_dumper is not None:
            pobj = {
                'id': rec.get('id'),
                'prompt': rec.get('_prompt_used') or '',
                'mode': 'oracle' if args.oracle else ('master' if args.master else 'unknown'),
                'engine': args.engine,
                'temperature': args.temperature,
                'disable_cache': args.disable_cache,
            }
            import json as _json
            prompt_dumper.write(_json.dumps(pobj) + "\n")

    print("\n=== Summary ===")
    print(f"Checked:   {total_lines}")
    if args.master and args.oracle:
        print(f"Both pass: {both_pass}")
        print(f"Both fail: {both_fail}")
        print(f"Mismatch:  {mismatch}")

    if dumper is not None:
        dumper.close()
    if prompt_dumper is not None:
        prompt_dumper.close()


if __name__ == "__main__":
    main()


