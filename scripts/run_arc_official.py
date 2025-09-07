#!/usr/bin/env python3
import sys
import argparse
import importlib.util


def main():
    parser = argparse.ArgumentParser(description="Run an ARC submission against the official ARC-AGI data")
    parser.add_argument("task_num", type=int, help="1-based task index within the ARC-AGI training set")
    parser.add_argument("--submission_dir", type=str, default="submission", help="Directory containing taskNNN.py modules")
    parser.add_argument("--dataset", type=str, default="arcagi", help="Dataset name for arckit.load_data (default: arcagi)")
    args = parser.parse_args()

    try:
        import arckit
        import arckit.vis as vis
        import numpy as np
    except Exception as e:
        print("Failed to import arckit. Install via `pip install arc-kit` (or your local package).\n", e)
        sys.exit(1)

    task_num = int(args.task_num)
    train_set, _ = arckit.load_data(args.dataset)
    task = list(train_set)[task_num - 1]

    submission_path = f"{args.submission_dir}/task{task_num:03d}.py"
    spec = importlib.util.spec_from_file_location("solution", submission_path)
    if spec is None or spec.loader is None:
        print(f"Could not load submission module: {submission_path}")
        sys.exit(1)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    for i, (inp, out) in enumerate(task.train):
        result = module.p(inp.tolist())
        status = "✓" if result == out.tolist() else "✗"
        print(f"train[{i}]: {status}")
        print("Input:")
        vis.print_grid(inp)
        print("Expected:")
        vis.print_grid(out)
        print("Got:")
        vis.print_grid(np.array(result))
        print()

    for i, (inp, out) in enumerate(task.test):
        result = module.p(inp.tolist())
        status = "✓" if result == out.tolist() else "✗"
        print(f"test[{i}]: {status}")
        print("Input:")
        vis.print_grid(inp)
        print("Expected:")
        vis.print_grid(out)
        print("Got:")
        vis.print_grid(np.array(result))
        print()


if __name__ == "__main__":
    main()


