#!/usr/bin/env python3
import os
import sys
import json
import argparse
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def run_cmd(cmd: str, extra_env=None):
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    if extra_env:
        env.update(extra_env)
    print(cmd)
    proc = subprocess.Popen(cmd, shell=True, env=env)
    proc.communicate()
    if proc.returncode != 0:
        sys.exit(proc.returncode)


def cmd_build(args):
    solvers = args.solvers or str(PROJECT_ROOT / "third_party/arc-dsl/solvers.py")
    out_dir = args.out_dir or str(PROJECT_ROOT / "data/arc/generated")
    cmd = f"/usr/bin/env python3 {PROJECT_ROOT / 'scripts' / 'build_arc_dataset.py'} --solvers {solvers} --out_dir {out_dir} --show_sample {args.show_sample}"
    run_cmd(cmd)


def cmd_run(args):
    engine = args.engine
    mode = args.mode
    template = "wrule" if mode == "rot" else "std"

    flags = [
        f"--dataset arc",
        f"--engine {engine}",
        f"--prompt_mode {mode}",
        f"--prompt_template {template}",
        f"--retrieve_fn {args.retrieve_fn}",
        f"--batch_size {args.batch_size}",
        f"--num_shot {args.num_shot}",
    ]
    if mode == "rot" and args.add_rule_instruction:
        flags.append("--add_rule_instruction_flag")
    if args.constrain_prog:
        flags.append("--constrain_prog_gen_flag")
    if args.oracle:
        flags.append("--use_oracle_rule_flag")

    cmd = f"/usr/bin/env python3 {PROJECT_ROOT / 'neural_lark' / 'main.py'} " + " ".join(flags)
    run_cmd(cmd, extra_env={"WANDB_MODE": "offline"})


def _latest_results_dir():
    log_root = PROJECT_ROOT / "log" / "sempar-rule-icl"
    if not log_root.exists():
        return None
    runs = sorted(log_root.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True)
    return runs[0] if runs else None


def _latest_log_file(run_dir: Path):
    cand = sorted(run_dir.glob("log_*.txt"), key=lambda p: p.stat().st_mtime, reverse=True)
    return cand[0] if cand else None


def cmd_show(args):
    run_dir = Path(args.dir) if args.dir else _latest_results_dir()
    if not run_dir:
        print("No results found.")
        sys.exit(1)
    results = run_dir / "results.json"
    print(f"Reading {results}")
    data = json.load(open(results))

    if args.what in ("summary", "all"):
        print(f"#examples: {len(data.get('test_predictions', []))}")

    if args.what in ("prompts", "all"):
        idx = args.index
        prompts = data.get("test_prompts", [])
        if 0 <= idx < len(prompts):
            print(f"\n=== prompts[{idx}] ===")
            for j, p in enumerate(prompts[idx]):
                print(f"[prompt {j}]\n{p}\n")
        else:
            print("No prompts at that index.")

    if args.what in ("predictions", "all"):
        idx = args.index
        preds = data.get("test_predictions", [])
        if 0 <= idx < len(preds):
            print(f"\n=== predictions[{idx}] ===\n{json.dumps(preds[idx], indent=2)}")
        else:
            print("No predictions at that index.")

    if args.what in ("grammars", "all"):
        idx = args.index
        grams = data.get("test_grammars", [])
        if 0 <= idx < len(grams):
            print(f"\n=== grammars[{idx}] ===\n{json.dumps(grams[idx], indent=2)}")
        else:
            print("No grammars at that index.")


def cmd_split(args):
    run_dir = Path(args.dir) if args.dir else _latest_results_dir()
    if not run_dir:
        print("No results found.")
        sys.exit(1)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = f"/usr/bin/env python3 {PROJECT_ROOT / 'scripts' / 'split_results.py'} --run_dir '{run_dir}' --dataset arc --out_dir '{out_dir}'"
    if args.include_prompts:
        cmd += " --include_prompts"
    run_cmd(cmd)


def cmd_tail(args):
    run_dir = Path(args.dir) if args.dir else _latest_results_dir()
    if not run_dir:
        print("No run directory found.")
        sys.exit(1)
    log_file = _latest_log_file(run_dir)
    if not log_file:
        print(f"No log file yet in {run_dir}")
        sys.exit(1)
    print(f"Tailing {log_file} (Ctrl-C to stop)...")
    # Use system tail -f for smooth streaming
    os.execvp("/usr/bin/env", ["/usr/bin/env", "bash", "-lc", f"tail -f '{str(log_file)}'"])


def main():
    parser = argparse.ArgumentParser(description="ARC CLI: build dataset, run prompting, inspect results")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_build = sub.add_parser("build", help="Build ARC dataset from solvers.py")
    p_build.add_argument("--solvers", type=str, default=None)
    p_build.add_argument("--out_dir", type=str, default=None)
    p_build.add_argument("--show_sample", type=int, default=2)
    p_build.set_defaults(func=cmd_build)

    p_run = sub.add_parser("run", help="Run grammar prompting on ARC")
    p_run.add_argument("--engine", type=str, required=True)
    p_run.add_argument("--mode", type=str, choices=["std", "rot"], default="rot")
    p_run.add_argument("--retrieve_fn", type=str, default="bm25")
    p_run.add_argument("--batch_size", type=int, default=8)
    p_run.add_argument("--num_shot", type=int, default=-1)
    p_run.add_argument("--constrain_prog", action="store_true", default=True)
    p_run.add_argument("--no_constrain_prog", dest="constrain_prog", action="store_false")
    p_run.add_argument("--add_rule_instruction", action="store_true", default=True)
    p_run.add_argument("--no_add_rule_instruction", dest="add_rule_instruction", action="store_false")
    p_run.add_argument("--oracle", action="store_true", help="Use oracle minimal grammar (use_oracle_rule_flag)")
    p_run.set_defaults(func=cmd_run)

    p_show = sub.add_parser("show", help="Inspect latest run results")
    p_show.add_argument("--dir", type=str, default=None, help="Explicit run dir (defaults to latest)")
    p_show.add_argument("--what", type=str, choices=["summary", "prompts", "predictions", "grammars", "all"], default="summary")
    p_show.add_argument("--index", type=int, default=0, help="Example index to show")
    p_show.set_defaults(func=cmd_show)

    p_split = sub.add_parser("split", help="Split latest results.json into per-example files")
    p_split.add_argument("--dir", type=str, default=None, help="Explicit run dir (defaults to latest)")
    p_split.add_argument("--out_dir", type=str, required=True, help="Output folder for per-example files")
    p_split.add_argument("--include_prompts", action="store_true", help="Include prompts in each output JSON")
    p_split.set_defaults(func=cmd_split)

    p_tail = sub.add_parser("tail", help="Stream the latest run log in real time")
    p_tail.add_argument("--dir", type=str, default=None, help="Explicit run dir (defaults to latest)")
    p_tail.set_defaults(func=cmd_tail)

    def cmd_export(args):
        run_dir = Path(args.dir) if args.dir else _latest_results_dir()
        if not run_dir:
            print("No results found.")
            sys.exit(1)
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = f"/usr/bin/env python3 {PROJECT_ROOT / 'scripts' / 'make_submissions_from_results.py'} --results '{run_dir / 'results.json'}' --out_dir '{out_dir}'"
        run_cmd(cmd)

    p_export = sub.add_parser("export", help="Export latest results.json to submission/taskNNN.py files")
    p_export.add_argument("--dir", type=str, default=None, help="Explicit run dir (defaults to latest)")
    p_export.add_argument("--out_dir", type=str, default="submission", help="Output folder for submissions")
    p_export.set_defaults(func=cmd_export)

    def cmd_official(args):
        task = args.task
        cmd = f"/usr/bin/env python3 {PROJECT_ROOT / 'scripts' / 'run_arc_official.py'} {task} --submission_dir '{args.submission_dir}' --dataset '{args.dataset}'"
        run_cmd(cmd)

    p_off = sub.add_parser("official", help="Run a generated submission against official ARC data")
    p_off.add_argument("task", type=int, help="1-based index in ARC-AGI train set")
    p_off.add_argument("--submission_dir", type=str, default="submission")
    p_off.add_argument("--dataset", type=str, default="arcagi")
    p_off.set_defaults(func=cmd_official)

    def cmd_control(args):
        cmd = f"/usr/bin/env python3 {PROJECT_ROOT / 'scripts' / 'arc_control_eval.py'} --dataset '{args.dataset}' --oracle_jsonl '{args.oracle_jsonl}'"
        if args.show_failures:
            cmd += " --show_failures"
        if args.limit is not None:
            cmd += f" --limit {args.limit}"
        run_cmd(cmd)

    p_ctl = sub.add_parser("control", help="Run control evaluation using oracle programs + dsl interpreter")
    p_ctl.add_argument("--dataset", type=str, default="arcagi")
    p_ctl.add_argument("--oracle_jsonl", type=str, default=str(PROJECT_ROOT / 'data' / 'arc' / 'oracle' / 'arc_oracle.jsonl'))
    p_ctl.add_argument("--show_failures", action="store_true")
    p_ctl.add_argument("--limit", type=int, default=None)
    p_ctl.set_defaults(func=cmd_control)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()


