#!/usr/bin/env python3
import os
import sys
import json
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Run ARC DSL interpreter on a program and input grid")
    parser.add_argument("--program", type=str, required=True, help="ARC program string (use ' ## ' separator for multiple statements)")
    parser.add_argument("--grid", type=str, required=True, help="Path to JSON grid file (nested lists)")
    parser.add_argument("--show_env", action="store_true", help="Print final environment keys")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(project_root))
    from neural_lark.arc_interpreter import interpret_program

    with open(args.grid, "r") as f:
        grid = json.load(f)

    result, env = interpret_program(args.program, grid, project_root)
    print(json.dumps({"result": result}, indent=2))
    if args.show_env:
        # only show x* bindings and I
        xs = {k: v for k, v in env.items() if k == "I" or (k.startswith("x") and k[1:].isdigit())}
        print(json.dumps({"env": xs}, indent=2))


if __name__ == "__main__":
    main()


