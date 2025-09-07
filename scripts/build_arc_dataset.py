import os
import sys
import re
import random
import argparse


def _extract_arc_examples_from_solvers(solvers_path: str):
    with open(solvers_path, 'r') as f:
        code = f.read()

    def rewrite_var_calls_to_apply(expr: str) -> str:
        def replace_match(m: re.Match) -> str:
            var_name = m.group(1)
            rest = m.group(2)
            depth = 0
            for i, ch in enumerate(rest):
                if ch == '(':
                    depth += 1
                elif ch == ')':
                    depth -= 1
                    if depth == 0:
                        args_str = rest[1:i]
                        remainder = rest[i+1:]
                        return f"apply({var_name}, {args_str}){remainder}"
            return f"{var_name}{rest}"

        prev = None
        cur = expr
        for _ in range(10):
            if prev == cur:
                break
            prev = cur
            m = re.search(r"\b(x\d+)\s*(\(.*)", cur)
            if not m:
                break
            prefix = cur[:m.start()]
            replaced = replace_match(m)
            cur = prefix + replaced
        return cur

    blocks = []
    for m in re.finditer(r"^def\s+solve_([0-9a-f]+)\(I\):\n", code, flags=re.MULTILINE):
        func_id = m.group(1)
        next_m = re.search(r"^def\s+solve_[0-9a-f]+\(I\):\n", code[m.end():], flags=re.MULTILINE)
        if next_m:
            block = code[m.end(): m.end() + next_m.start()]
        else:
            block = code[m.end():]
        blocks.append((func_id, block))

    class Example:
        def __init__(self, source: str, target: str):
            self.source = source
            self.target = target

    examples = []
    for func_id, block in blocks:
        lines = []
        for raw in block.splitlines():
            line = raw.strip()
            if not line or line.startswith('return '):
                continue
            if '=' in line:
                _, rhs = line.split('=', 1)
                rhs = rhs.split('#', 1)[0].strip()
                rhs = rewrite_var_calls_to_apply(rhs)
                lines.append(rhs)
        if not lines:
            continue
        source = f"ARC task solve_{func_id}"
        target = " ## ".join(lines)
        examples.append(Example(source, target))
    return examples


def _write_arc_splits(examples, out_dir: str, train_ratio=0.8, dev_ratio=0.1, seed=13):
    os.makedirs(out_dir, exist_ok=True)
    rng = random.Random(seed)
    shuffled = examples[:]
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * train_ratio)
    n_dev = int(n * dev_ratio)
    train, dev, test = shuffled[:n_train], shuffled[n_train:n_train+n_dev], shuffled[n_train+n_dev:]
    def write_split(name, split):
        src_path = os.path.join(out_dir, f"{name}.src")
        tgt_path = os.path.join(out_dir, f"{name}.tgt")
        with open(src_path, 'w') as fs, open(tgt_path, 'w') as ft:
            for ex in split:
                fs.write(ex.source + "\n")
                ft.write(ex.target + "\n")
        return f"{src_path},{tgt_path}"
    return write_split('train', train), write_split('dev', dev), write_split('test', test)


def main():
    parser = argparse.ArgumentParser(description="Build ARC dataset (train/dev/test) from third_party/arc-dsl/solvers.py")
    parser.add_argument("--solvers", type=str, default="third_party/arc-dsl/solvers.py", help="Path to solvers.py")
    parser.add_argument("--out_dir", type=str, default="data/arc/generated", help="Output directory for splits")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Train split ratio")
    parser.add_argument("--dev_ratio", type=float, default=0.1, help="Dev split ratio; test = 1 - train - dev")
    parser.add_argument("--seed", type=int, default=13, help="Random seed for shuffling")
    parser.add_argument("--show_sample", type=int, default=0, help="Print N sample examples from the built dataset")
    args = parser.parse_args()

    print(f"Reading solvers from: {args.solvers}")
    examples = _extract_arc_examples_from_solvers(args.solvers)
    print(f"Parsed {len(examples)} solver programs into ARC examples")

    train_file, dev_file, test_file = _write_arc_splits(examples, args.out_dir, train_ratio=args.train_ratio, dev_ratio=args.dev_ratio, seed=args.seed)
    print("Wrote splits:")
    print(f"  train: {train_file}")
    print(f"  dev:   {dev_file}")
    print(f"  test:  {test_file}")

    if args.show_sample > 0:
        # Show a few lines from each split
        def head(path, n):
            fn_src, fn_tgt = path.split(",")
            with open(fn_src) as fs, open(fn_tgt) as ft:
                for i, (s, t) in enumerate(zip(fs, ft)):
                    if i >= n:
                        break
                    print("- source:")
                    print(s.rstrip())
                    print("  target:")
                    print(t.rstrip())
                    print()

        print("\nSample from train:")
        head(train_file, args.show_sample)


if __name__ == "__main__":
    main()


