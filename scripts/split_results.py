#!/usr/bin/env python3
import os
import sys
import re
import json
import argparse
from pathlib import Path


def _counter2pred(counter_obj):
    if not counter_obj:
        return None
    # counter_obj is a dict: {string: count}
    items = list(counter_obj.items())
    items.sort(key=lambda kv: (-kv[1], len(kv[0])))
    return items[0][0]


def _load_sources_for_dataset(dataset: str):
    if dataset == "arc":
        src_file = Path("data/arc/generated/test.src")
        if not src_file.exists():
            raise FileNotFoundError(f"Missing {src_file}. Build the dataset first.")
        with open(src_file, "r") as f:
            sources = [line.rstrip("\n") for line in f]
        return sources
    raise ValueError(f"Unsupported dataset for split: {dataset}")


def main():
    parser = argparse.ArgumentParser(description="Split results.json into per-example files with input, predicted grammar, and predicted program")
    parser.add_argument("--run_dir", type=str, required=True, help="Path to run directory containing results.json")
    parser.add_argument("--dataset", type=str, default="arc", help="Dataset name to recover inputs (default: arc)")
    parser.add_argument("--out_dir", type=str, required=True, help="Directory to write per-example files")
    parser.add_argument("--include_prompts", action="store_true", help="Also include prompts in each output JSON")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    results_path = run_dir / "results.json"
    if not results_path.exists():
        print(f"results.json not found at {results_path}")
        sys.exit(1)

    with open(results_path, "r") as f:
        data = json.load(f)

    predictions = data.get("test_predictions", [])
    grammars = data.get("test_grammars", [])
    prompts = data.get("test_prompts", [])

    sources = _load_sources_for_dataset(args.dataset)

    n = len(predictions)
    if len(sources) < n:
        print(f"Warning: fewer sources ({len(sources)}) than predictions ({n}); truncating to min length")
        n = len(sources)
    elif len(sources) > n:
        print(f"Warning: more sources ({len(sources)}) than predictions ({n}); extra sources ignored")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for i in range(n):
        source = sources[i]
        pred_prog = _counter2pred(predictions[i]) if i < len(predictions) else None
        pred_gram = _counter2pred(grammars[i]) if i < len(grammars) and grammars[i] is not None else None
        item = {
            "index": i,
            "source": source,
            "predicted_program": pred_prog,
            "predicted_grammar": pred_gram,
        }
        if args.include_prompts and i < len(prompts):
            item["prompts"] = prompts[i]

        # Prefer ARC task id in filename when available
        m = re.search(r"solve_([0-9a-f]+)", source)
        if m:
            fname = f"{i:04d}_solve_{m.group(1)}.json"
        else:
            fname = f"{i:04d}.json"
        with open(out_dir / fname, "w") as fo:
            json.dump(item, fo, indent=2)

    print(f"Wrote {n} files to {out_dir}")


if __name__ == "__main__":
    main()


