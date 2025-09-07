#!/usr/bin/env python3
import os
import sys
import re
import json
import argparse
from pathlib import Path


TEMPLATE = """# Auto-generated from {results_path}
from pathlib import Path
from typing import List


def p(grid: List[List[int]]) -> List[List[int]]:
    # Convert to tuples for ARC DSL
    G = tuple(tuple(r) for r in grid)
    # Program predicted by the model
    program = {program_json}
    # Run interpreter
    from neural_lark.arc_interpreter import interpret_program
    project_root = Path(__file__).resolve().parents[1]
    result, _ = interpret_program(program, G, project_root)
    # Back to lists
    return [list(r) for r in result]
"""


def counter2pred(counter_obj):
    if not counter_obj:
        return None
    items = list(counter_obj.items())
    items.sort(key=lambda kv: (-kv[1], len(kv[0])))
    return items[0][0]


def main():
    parser = argparse.ArgumentParser(description="Create submission/*.py p() files from results.json")
    parser.add_argument("--results", type=str, required=True, help="Path to results.json")
    parser.add_argument("--out_dir", type=str, default="submission", help="Output directory for taskNNN.py files")
    args = parser.parse_args()

    results_path = Path(args.results)
    data = json.load(open(results_path))
    preds = data.get("test_predictions", [])
    prompts = data.get("test_prompts", [])

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mapping = []
    for i, counter in enumerate(preds):
        program = counter2pred(counter)
        if program is None:
            program = "I"  # identity fallback
        # Write taskNNN.py
        nnn = f"task{i+1:03d}.py"
        content = TEMPLATE.format(results_path=str(results_path), program_json=json.dumps(program))
        (out_dir / nnn).write_text(content)
        mapping.append({
            "index": i,
            "file": nnn,
            "program": program,
            "prompts": prompts[i] if i < len(prompts) else None,
        })

    (out_dir / "mapping.json").write_text(json.dumps(mapping, indent=2))
    print(f"Wrote {len(mapping)} files to {out_dir}")


if __name__ == "__main__":
    main()


