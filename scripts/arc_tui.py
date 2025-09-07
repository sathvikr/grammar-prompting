#!/usr/bin/env python3
import os
import sys
import json
import argparse
import curses
from typing import List, Dict, Optional
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

def build_prompt_for_record(rec: Dict, dataset: Optional[str]) -> str:
    # Reuse arc_eval logic to build prompts deterministically and include few-shot examples
    from scripts.arc_eval import build_prompt, load_tasks
    master_lark_path = os.path.join(PROJECT_ROOT, "grammars", "arc_1.lark")

    fewshot_text = None
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

    return build_prompt(rec, use_oracle=True, master_lark_path=master_lark_path, fewshot_text=fewshot_text)


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
            line = f"[{j+1}/{len(self.items)}] {ex_id}{extra}"
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
    programs_by_id = load_program_dump(programs_path)
    prompts_by_id = load_prompt_dump(prompts_path)

    # Build pass/fail status if dataset provided and arckit available
    status_by_id: Dict[str, Optional[bool]] = {}
    if dataset:
        try:
            from scripts.arc_eval import load_tasks, build_oracle_parser_and_run
            import arckit  # noqa: F401
            tasks_by_id = load_tasks(dataset)
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
            if rid in prompts_by_id:
                prompt = prompts_by_id[rid].get('prompt', '')
            else:
                prompt = build_prompt_for_record(rec, dataset)
            viewer = TextViewer(stdscr, prompt, f"Prompt: {rec.get('id')}")
            viewer.loop()
        elif ch in (ord('s'), ord('S')):
            rec = list_view.current()
            rid = str(rec.get('id', '')).lower()
            prog = None
            if rid in programs_by_id:
                prog = programs_by_id[rid].get('program_generated')
            if not prog:
                prog = rec.get('program', '<no program available>')
            viewer = TextViewer(stdscr, str(prog), f"Program: {rec.get('id')}")
            viewer.loop()


def main():
    ap = argparse.ArgumentParser(description="Terminal UI to browse ARC tasks, prompts, and solutions")
    ap.add_argument("--file", default="data/arc/oracle/arc_oracle.jsonl", help="Path to ARC JSONL input")
    ap.add_argument("--programs", default=None, help="Path to JSONL dump of generated programs (from arc_eval --dump_programs)")
    ap.add_argument("--dataset", default=None, help="arckit dataset name to color-code pass/fail (e.g., arcagi)")
    ap.add_argument("--prompts", default=None, help="Path to JSONL dump of prompts (from arc_eval --dump_prompts)")
    args = ap.parse_args()

    curses.wrapper(run_tui, args.file, args.programs, args.dataset, args.prompts)


if __name__ == "__main__":
    main()


