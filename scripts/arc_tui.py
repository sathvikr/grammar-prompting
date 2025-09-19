#!/usr/bin/env python3
import os
import sys
import json
import argparse
import curses
from datetime import datetime
try:
    from zoneinfo import ZoneInfo  # Python 3.9+
except Exception:
    ZoneInfo = None  # fallback handled below
from typing import List, Dict, Optional
import re
from pathlib import Path


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def load_jsonl(path: str) -> List[Dict]:
    items = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def load_program_dump(path: Optional[str]) -> Dict[str, Dict]:
    if not path:
        return {}
    if not os.path.exists(path):
        return {}
    mp: Dict[str, Dict] = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            _id = str(rec.get("id", ""))
            if _id:
                mp[_id.lower()] = rec
    return mp


def load_prompt_dump(path: Optional[str]) -> Dict[str, Dict]:
    if not path:
        return {}
    if not os.path.exists(path):
        return {}
    mp: Dict[str, Dict] = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            _id = str(rec.get("id", ""))
            if _id:
                mp[_id.lower()] = rec
    return mp

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


def _extract_solver_programs(solvers_path: str) -> Dict[str, str]:
    code = open(solvers_path, 'r').read()
    mapping: Dict[str, str] = {}
    for m in re.finditer(r"^def\s+solve_([0-9a-f]+)\(I\):\n", code, flags=re.MULTILINE):
        func_id = m.group(1)
        start = m.end()
        next_m = re.search(r"^def\s+solve_[0-9a-f]+\(I\):\n", code[start:], flags=re.MULTILINE)
        block = code[start: start + next_m.start()] if next_m else code[start:]
        stmts: List[str] = []
        for raw in block.splitlines():
            line = raw.strip()
            if not line or line.startswith('return '):
                continue
            if '=' in line:
                lhs, rhs = line.split('=', 1)
                lhs = lhs.strip()
                rhs = rhs.strip()
                if '#' in rhs:
                    rhs = rhs.split('#', 1)[0].strip()
                stmts.append(f"{lhs} = {rhs}")
        if stmts:
            mapping[func_id] = " \n".join(stmts)
    return mapping


def build_prompt_for_record(rec: Dict, dataset: Optional[str]) -> str:
    # Reuse arc_eval logic to build prompts deterministically and include few-shot and retrieval examples
    from scripts.arc_eval import build_prompt, load_tasks
    from neural_lark.retriever import BM25Retriever
    master_lark_path = os.path.join(PROJECT_ROOT, "grammars", "arc_1.lark")

    fewshot_text = None
    retrieval_text = None
    if dataset:
        try:
            import arckit  # noqa: F401
            tasks_by_id = load_tasks(dataset)
            rid = str(rec.get('id', '')).lower()
            task = tasks_by_id.get(rid)
            if task is not None:
                lines: List[str] = []
                for idx, (inp, out) in enumerate(list(task.train)):
                    ijson = inp.tolist() if hasattr(inp, 'tolist') else inp
                    ojson = out.tolist() if hasattr(out, 'tolist') else out
                    lines.append(f"- Pair {idx+1}:\n  Input:\n  {ijson}\n  Output:\n  {ojson}\n")
                fewshot_text = "\n".join(lines)
        except Exception:
            fewshot_text = None

    # Build retrieval block: 16 other tasks with oracle BNF and solver program, plus IO grids
    try:
        # Oracle BNFs: canonical file
        merged_oracle_map: Dict[str, str] = {}
        try:
            canonical_jsonl = os.path.join(PROJECT_ROOT, 'data', 'arc', 'oracle', 'arc_oracle.jsonl')
            merged_oracle_map.update(_load_oracle_map(canonical_jsonl))
        except Exception:
            pass

        # Extract programs from solvers
        solvers_path = os.path.join(PROJECT_ROOT, 'third_party', 'arc-dsl', 'solvers.py')
        solver_prog_map = _extract_solver_programs(solvers_path)

        # Build examples list with ids we have BNFs for
        class _Ex:  # minimal container
            def __init__(self, src: str, rid: str):
                self.source = src
                self.rid = rid

        examples = []
        for rid in solver_prog_map.keys():
            if rid in merged_oracle_map:
                examples.append(_Ex(f"solve_{rid}", rid))

        def _ex2doc(ex_obj: _Ex) -> str:
            bnf_local = merged_oracle_map.get(ex_obj.rid, '')
            return re.sub(r"\s+", " ", bnf_local).strip()

        if examples:
            bm25 = BM25Retriever(examples, ex2doc=_ex2doc)
            bnf_q = rec.get('oracle') or rec.get('oracle_bnf') or ''
            if bnf_q:
                retrieved, _ = bm25.retrieve_by_src(re.sub(r"\s+", " ", bnf_q).strip(), n=18)
            else:
                retrieved = examples[:18]
            # Filter current id and dedupe to 16
            cur_id = str(rec.get('id', '')).lower()
            filtered: List[_Ex] = []
            seen = set()
            for ex in retrieved:
                rid_local = ex.rid
                if not rid_local or rid_local == cur_id or rid_local in seen:
                    continue
                filtered.append(ex)
                seen.add(rid_local)
                if len(filtered) >= 16:
                    break

            # Load tasks for IO grids display
            try:
                tasks_by_id_ret = load_tasks(dataset or 'arcagi')
            except Exception:
                tasks_by_id_ret = {}

            lines_r = []
            for ex in filtered:
                rid_local = ex.rid
                bnf_show = merged_oracle_map.get(rid_local)
                prog_show = solver_prog_map.get(rid_local, '')
                io_block = ""
                task_obj = tasks_by_id_ret.get(rid_local)
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
                    lines_r.append(f"- Task {rid_local}:{io_block}\n  Oracle BNF:\n  {bnf_show}\n  Program:\n  {prog_show}")
                else:
                    lines_r.append(f"- Task {rid_local}:{io_block}\n  Program:\n  {prog_show}")
            retrieval_text = "\n".join(lines_r) if lines_r else None
    except Exception:
        retrieval_text = None

    return build_prompt(rec, use_oracle=True, master_lark_path=master_lark_path, fewshot_text=fewshot_text, retrieval_text=retrieval_text)


class ListView:
    def __init__(self, stdscr, items: List[Dict], programs_by_id: Dict[str, Dict], status_by_id: Optional[Dict[str, Optional[bool]]] = None):
        self.stdscr = stdscr
        self.items = items
        self.programs_by_id = programs_by_id
        self.status_by_id = status_by_id or {}
        self.idx = 0
        self.top = 0

    def draw(self):
        self.stdscr.clear()
        h, w = self.stdscr.getmaxyx()
        # Compute accuracy if we have statuses
        total = len(self.items)
        known = [v for v in self.status_by_id.values() if v is not None]
        acc = (100.0 * sum(1 for v in known if v) / max(1, len(known))) if known else 0.0
        passed = sum(1 for v in known if v)
        title = f"ARC Viewer - oracle | {passed}/{total} = {acc:.1f}% | ↑/↓ navigate  P:Prompt  S:Solution  Q:Quit"
        self.stdscr.addnstr(0, 0, title, w - 1, curses.A_REVERSE)
        visible_h = h - 2
        if self.idx < self.top:
            self.top = self.idx
        if self.idx >= self.top + visible_h:
            self.top = self.idx - visible_h + 1
        for i in range(visible_h):
            j = self.top + i
            if j >= len(self.items):
                break
            rec = self.items[j]
            ex_id = str(rec.get("id", ""))
            extra = ""
            if ex_id.lower() in self.programs_by_id:
                src = self.programs_by_id[ex_id.lower()].get("source", "?")
                extra = f"  [prog:{src}]"
            # Lookup stored generation timestamp for this id (UTC ISOZ) and render as PST
            ts_str = ""
            prog_rec = self.programs_by_id.get(ex_id.lower())
            if prog_rec:
                iso = prog_rec.get('generated_at')
                if iso:
                    try:
                        if iso.endswith('Z'):
                            dt = datetime.fromisoformat(iso.replace('Z', '+00:00'))
                        else:
                            dt = datetime.fromisoformat(iso)
                        if ZoneInfo is not None:
                            dt_la = dt.astimezone(ZoneInfo("America/Los_Angeles"))
                        else:
                            dt_la = dt
                        month_map = {1: "Jan.", 2: "Feb.", 3: "Mar.", 4: "Apr.", 5: "May", 6: "Jun.", 7: "Jul.", 8: "Aug.", 9: "Sept.", 10: "Oct.", 11: "Nov.", 12: "Dec."}
                        mname = month_map.get(dt_la.month, dt_la.strftime('%b'))
                        try:
                            hm = dt_la.strftime('%-I:%M %p')
                        except Exception:
                            hm = dt_la.strftime('%I:%M %p').lstrip('0')
                        ts_str = f"  {mname} {dt_la.day}, {hm} (PST)"
                    except Exception:
                        ts_str = ""
            line = f"[{j+1}/{len(self.items)}] {ex_id}{extra}{ts_str}"
            attr = curses.A_REVERSE if j == self.idx else curses.A_NORMAL
            # color-code pass/fail if known
            rid = ex_id.lower()
            status = self.status_by_id.get(rid, None)
            if status is True:
                self.stdscr.addnstr(1 + i, 0, "✓ ", 2, curses.color_pair(1) | attr)
                self.stdscr.addnstr(1 + i, 2, line, w - 3, attr)
            elif status is False:
                self.stdscr.addnstr(1 + i, 0, "✗ ", 2, curses.color_pair(2) | attr)
                self.stdscr.addnstr(1 + i, 2, line, w - 3, attr)
            else:
                self.stdscr.addnstr(1 + i, 0, line, w - 1, attr)
        self.stdscr.refresh()

    def current(self) -> Dict:
        return self.items[self.idx]

    def move(self, delta: int):
        self.idx = max(0, min(len(self.items) - 1, self.idx + delta))


class TextViewer:
    def __init__(self, stdscr, text: str, title: str):
        self.stdscr = stdscr
        self.text_lines = text.splitlines() or [""]
        self.title = title
        self.top = 0
        self._wrapped_len = len(self.text_lines)

    def loop(self):
        while True:
            self.draw()
            ch = self.stdscr.getch()
            if ch in (ord('q'), ord('Q'), 27, curses.KEY_LEFT, ord('b'), ord('B')):
                return
            elif ch in (curses.KEY_DOWN, ord('j')):
                self.top = min(max(0, self._wrapped_len - 1), self.top + 1)
            elif ch in (curses.KEY_UP, ord('k')):
                self.top = max(0, self.top - 1)
            elif ch in (curses.KEY_NPAGE, ord(' ')):
                h, _ = self.stdscr.getmaxyx()
                self.top = min(max(0, self._wrapped_len - 1), self.top + (h - 2))
            elif ch in (curses.KEY_PPAGE,):
                h, _ = self.stdscr.getmaxyx()
                self.top = max(0, self.top - (h - 2))

    def draw(self):
        self.stdscr.clear()
        h, w = self.stdscr.getmaxyx()
        header = f"{self.title}  (wrapped)  (q/B/Left to go back)"
        self.stdscr.addnstr(0, 0, header, w - 1, curses.A_REVERSE)
        visible_h = h - 2

        # Soft-wrap long lines to the terminal width
        wrap_w = max(1, w - 1)
        wrapped: list[str] = []
        for ln in self.text_lines:
            if len(ln) <= wrap_w:
                wrapped.append(ln)
            else:
                # hard slice to width; avoids relying on whitespace
                for i in range(0, len(ln), wrap_w):
                    wrapped.append(ln[i:i+wrap_w])

        self._wrapped_len = len(wrapped)
        # clamp top within range
        if self.top > max(0, self._wrapped_len - 1):
            self.top = max(0, self._wrapped_len - 1)

        for i in range(visible_h):
            j = self.top + i
            if j >= self._wrapped_len:
                break
            self.stdscr.addnstr(1 + i, 0, wrapped[j], w - 1)
        self.stdscr.refresh()


def run_tui(stdscr, file: str, programs_path: Optional[str], dataset: Optional[str], prompts_path: Optional[str]):
    curses.curs_set(0)
    curses.start_color()
    try:
        curses.use_default_colors()
    except Exception:
        pass
    curses.init_pair(1, curses.COLOR_GREEN, -1)
    curses.init_pair(2, curses.COLOR_RED, -1)

    items = load_jsonl(file)
    # Enforce single default dump paths used by arc_eval
    from scripts.arc_eval import DEFAULT_PROGRAMS_DUMP, DEFAULT_PROMPTS_DUMP
    programs_path = DEFAULT_PROGRAMS_DUMP
    prompts_path = DEFAULT_PROMPTS_DUMP
    if not os.path.exists(programs_path):
        raise SystemExit(f"Programs dump not found: {programs_path}. Run arc_eval with --prompt to generate it.")
    if not os.path.exists(prompts_path):
        raise SystemExit(f"Prompts dump not found: {prompts_path}. Run arc_eval with --prompt to generate it.")
    programs_by_id = load_program_dump(programs_path)
    prompts_by_id = load_prompt_dump(prompts_path)

    # Build pass/fail status if dataset provided and arckit available
    status_by_id: Dict[str, Optional[bool]] = {}
    if dataset:
        try:
            from scripts.arc_eval import load_tasks, build_oracle_parser_and_run
            import arckit  # noqa: F401
            tasks_by_id = load_tasks(dataset)

            def _normalize_grid(result):
                try:
                    if hasattr(result, 'tolist'):
                        result = result.tolist()
                    if isinstance(result, (list, tuple)) and result and isinstance(result[0], (list, tuple)):
                        return tuple(tuple(c for c in row) for row in result)
                    return result
                except Exception:
                    return result
            for rec in items:
                rid = str(rec.get('id', '')).lower()
                task = tasks_by_id.get(rid)
                if task is None:
                    status_by_id[rid] = None
                    continue
                prog = None
                if rid in programs_by_id:
                    prog = programs_by_id[rid].get('program_generated')
                if not prog:
                    prog = rec.get('program')
                if not prog:
                    status_by_id[rid] = None
                    continue
                ok_all = True
                pairs = list(task.train) + list(task.test)
                for (inp, out) in pairs:
                    grid = tuple(tuple(r) for r in inp.tolist()) if hasattr(inp, 'tolist') else inp
                    expected = tuple(tuple(r) for r in out.tolist()) if hasattr(out, 'tolist') else out
                    try:
                        pred = build_oracle_parser_and_run(rec.get('oracle_bnf', rec.get('oracle', '')), prog, grid, PROJECT_ROOT)
                        pred = _normalize_grid(pred)
                        if pred != expected:
                            ok_all = False
                            break
                    except Exception:
                        ok_all = False
                        break
                status_by_id[rid] = ok_all
        except Exception:
            status_by_id = {}

    list_view = ListView(stdscr, items, programs_by_id, status_by_id)

    while True:
        list_view.draw()
        ch = stdscr.getch()
        if ch in (ord('q'), ord('Q')):
            break
        elif ch in (curses.KEY_DOWN, ord('j')):
            list_view.move(+1)
        elif ch in (curses.KEY_UP, ord('k')):
            list_view.move(-1)
        elif ch in (curses.KEY_NPAGE,):
            h, _ = stdscr.getmaxyx()
            list_view.move(h - 2)
        elif ch in (curses.KEY_PPAGE,):
            h, _ = stdscr.getmaxyx()
            list_view.move(-(h - 2))
        elif ch in (ord('g'),):
            list_view.idx = 0
        elif ch in (ord('G'),):
            list_view.idx = len(items) - 1
        elif ch in (ord('p'), ord('P')):
            rec = list_view.current()
            rid = str(rec.get('id', '')).lower()
            if rid not in prompts_by_id:
                raise SystemExit(f"Missing prompt for {rid} in {prompts_path}")
            prompt = prompts_by_id[rid].get('prompt', '')
            viewer = TextViewer(stdscr, prompt, f"Prompt: {rec.get('id')}")
            viewer.loop()
        elif ch in (ord('s'), ord('S')):
            rec = list_view.current()
            rid = str(rec.get('id', '')).lower()
            if rid not in programs_by_id:
                raise SystemExit(f"Missing program for {rid} in {programs_path}")
            prog = programs_by_id[rid].get('program_generated', '')
            viewer = TextViewer(stdscr, str(prog), f"Program: {rec.get('id')}")
            viewer.loop()


def main():
    ap = argparse.ArgumentParser(description="Terminal UI to browse ARC tasks, prompts, and solutions")
    ap.add_argument("--file", default="data/arc/oracle/arc_oracle.jsonl", help="Path to ARC JSONL input")
    ap.add_argument("--dataset", default="arcagi", help="arckit dataset name to color-code pass/fail (default: arcagi)")
    args = ap.parse_args()

    # Convenience: auto-use /tmp/arc3.jsonl if present and user didn't override default file
    default_file = "data/arc/oracle/arc_oracle.jsonl"
    file_path = args.file
    if file_path == default_file and os.path.exists("/tmp/arc3.jsonl"):
        file_path = "/tmp/arc3.jsonl"

    curses.wrapper(run_tui, file_path, None, args.dataset, None)


if __name__ == "__main__":
    main()


