import os
import json
import random
from dataclasses import dataclass
import re

from minEarley.parser import EarleyParser
from neural_lark.flags import FLAGS
from neural_lark.lark_utils import * 
# from third_party.structg.eval import check_equiv

@dataclass
class Example:
    source: str
    target: str

    grammar = None
    label = None

def load_examples(filename):
    examples = []
    assert len(filename.split(",")) == 2
    src_filename = filename.split(",")[0]
    trg_filename = filename.split(",")[1]
    with open(src_filename) as f1, open(trg_filename) as f2:
        for line1, line2 in zip(f1, f2):
            examples.append(Example(source=line1.strip(), target=line2.strip(),))
    return examples

def _rewrite_var_calls_to_apply(expr: str) -> str:
    """
    Rewrite higher-order variable function calls like `x2(I)` into `apply(x2, I)`
    so that they conform to the textual DSL grammar which lacks generic NAME(...)
    call syntax for variables.
    """
    # Greedy-safe replacement using a small parser for balanced parentheses
    def replace_match(m: re.Match) -> str:
        var_name = m.group(1)
        rest = m.group(2)
        # rest starts with '(' and contains the full balanced arg list up to matching ')'
        # We need to find the balanced parentheses span
        depth = 0
        args = []
        start = 0
        for i, ch in enumerate(rest):
            if ch == '(':
                depth += 1
            elif ch == ')':
                depth -= 1
                if depth == 0:
                    # include up to i (inclusive)
                    args_str = rest[1:i]  # inside parentheses
                    remainder = rest[i+1:]
                    return f"apply({var_name}, {args_str}){remainder}"
        # If unbalanced, return original
        return f"{var_name}{rest}"

    # Replace repeatedly until no change
    pattern = re.compile(r"\b(x\d+)\s*(\(.*)\Z")
    prev = None
    cur = expr
    # To avoid catastrophic behavior, limit iterations
    for _ in range(10):
        if prev == cur:
            break
        prev = cur
        # Apply replacement only once per loop from leftmost occurrence
        m = re.search(r"\b(x\d+)\s*(\(.*)", cur)
        if not m:
            break
        prefix = cur[:m.start()]
        replaced = replace_match(m)
        cur = prefix + replaced
    return cur

def _extract_arc_examples_from_solvers(solvers_path: str):
    """
    Parse third_party/arc-dsl/solvers.py and convert each solver into an Example.
    - source: a synthetic description containing the solver id
    - target: a multi-line textual DSL program, one expression per line (RHS),
              with higher-order variable calls rewritten to apply(var, arg)
    """
    with open(solvers_path, 'r') as f:
        code = f.read()

    # Find all solver functions
    # Capture def solve_<id>(I): blocks
    blocks = []
    for m in re.finditer(r"^def\s+solve_([0-9a-f]+)\(I\):\n", code, flags=re.MULTILINE):
        func_start = m.start()
        func_id = m.group(1)
        # Find next def or EOF
        next_m = re.search(r"^def\s+solve_[0-9a-f]+\(I\):\n", code[m.end():], flags=re.MULTILINE)
        if next_m:
            block = code[m.end(): m.end() + next_m.start()]
        else:
            block = code[m.end():]
        blocks.append((func_id, block))

    examples = []
    for func_id, block in blocks:
        lines = []
        for raw in block.splitlines():
            line = raw.strip()
            if not line:
                continue
            if line.startswith('return '):
                continue
            # Keep only assignments of the form xN = <expr> or O = <expr>
            if '=' in line:
                lhs, rhs = line.split('=', 1)
                lhs = lhs.strip()
                rhs = rhs.strip()
                # Remove trailing comments if any
                if '#' in rhs:
                    rhs = rhs.split('#', 1)[0].strip()
                # Rewrite var function calls to apply(var, ...)
                rhs_rewritten = _rewrite_var_calls_to_apply(rhs)
                lines.append(rhs_rewritten)
        if not lines:
            continue
        source = f"ARC task solve_{func_id}"
        # Join statements into a single line, consistent with other datasets
        target = " ## ".join(lines)
        examples.append(Example(source=source, target=target))
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

def load_sempar_data(config):
    """
    Note: 
    1. Uncomment the normalize_program_for_all function to normalize the programs, which would be useful for the baseline using linearized_tree (e.g., need to make sure programs can be predicted in a round-trip fashion)
    2. num_shot = -1 means using all the training data
    """

    def normalize_program_for_all(exs, parser):
        for ex in exs:
            try:
                normalized_target = normalize_program(ex.target, parser)
                ex.target = normalized_target
            except:
                logger.warning(f"failed to normalize program: {ex.target}")
                pass

    if config["dataset"] == "geoquery":
        split_name = config["split_name"]
        num_shot = config["num_shot"] 
        train_filename = f"data/geoquery/{split_name}/train.src,data/geoquery/{split_name}/train.tgt"
        dev_filename = f"data/geoquery/{split_name}/dev.src,data/geoquery/{split_name}/dev.tgt"
        test_filename = f"data/geoquery/{split_name}/test.src,data/geoquery/{split_name}/test.tgt"

        train_examples = load_examples(train_filename)
        dev_examples = load_examples(dev_filename)
        test_examples = load_examples(test_filename)

        # normalize_program_for_all(train_examples + dev_examples + test_examples, parser)

        if num_shot != -1:
            train_examples = train_examples[:num_shot]

    elif config["dataset"] == "smc":
        num_shot = config["num_shot"] 
        split_name = config["split_name"]

        # no need to escape 
        def unescape(ex):
            if "\"\\\"" in ex.target:
                new_target = ex.target.replace("\"\\\"", "\"")
                new_target = new_target.replace("\\\"", "")
                ex.target = new_target
        
        def unescape_all(examples):
            for ex in examples:
                unescape(ex)
            
        if split_name == "indomain":
            data_dir = f"data/smcalflow_cs/calflow.orgchart.event_create_v2/source_domain_with_target_num0"
            train_filename = f"{data_dir}/train.canonical.src,{data_dir}/train.canonical.tgt"
            dev_filename = f"{data_dir}/valid.canonical.indist.src,{data_dir}/valid.canonical.indist.tgt"
            test_filename = f"{data_dir}/test.canonical.indist.src,{data_dir}/test.canonical.indist.tgt"

            train_examples = load_examples(train_filename)
            dev_examples = load_examples(dev_filename)
            test_examples = load_examples(test_filename)
            unescape_all(train_examples + dev_examples + test_examples)

            # normalize_program_for_all(train_examples + dev_examples + test_examples, parser)

            if num_shot != -1:
                train_examples = train_examples[:num_shot]

        else:
            assert split_name == "comp"
            data_dir = f"data/smcalflow_cs/calflow.orgchart.event_create_v2/source_domain_with_target_num{num_shot}"
            train_filename = f"{data_dir}/train.canonical.src,{data_dir}/train.canonical.tgt"
            dev_filename = f"{data_dir}/valid.canonical.outdist.src,{data_dir}/valid.canonical.outdist.tgt"
            test_filename = f"{data_dir}/test.canonical.outdist.src,{data_dir}/test.canonical.outdist.tgt"

            train_examples = load_examples(train_filename)
            dev_examples = load_examples(dev_filename)
            test_examples = load_examples(test_filename)
            unescape_all(train_examples + dev_examples + test_examples)
            # normalize_program_for_all(train_examples + dev_examples + test_examples, parser)

            if num_shot != -1:
                logger.info(f"Only use {num_shot} comp examples")
                train_examples = train_examples[-num_shot:]
    
    elif config["dataset"] == "overnight":
        domain = config["domain"]
        num_shot = config["num_shot"]
        data_dir = f"data/overnight/{domain}"
        train_filename = f"{data_dir}/train.src,{data_dir}/train.tgt"
        dev_filename = f"{data_dir}/dev.src,{data_dir}/dev.tgt"
        test_filename = f"{data_dir}/test.src,{data_dir}/test.tgt"

        train_examples = load_examples(train_filename)
        dev_examples = load_examples(dev_filename)
        test_examples = load_examples(test_filename)

        if num_shot != -1:
            train_examples = train_examples[-num_shot:]

    elif config["dataset"] == "mtop":
        split_name = config["split_name"]
        num_shot = config["num_shot"]
        data_dir = f"data/mtop/{split_name}-numshot{num_shot}"
        train_filename = f"{data_dir}/train.src,{data_dir}/train.tgt"
        dev_filename = f"{data_dir}/dev.src,{data_dir}/dev.tgt"
        test_filename = f"{data_dir}/test.src,{data_dir}/test.tgt"

        train_examples = load_examples(train_filename)
        dev_examples = load_examples(dev_filename)
        test_examples = load_examples(test_filename)

        if "indomain" in split_name and num_shot != -1:
            train_examples = train_examples[:num_shot]
        else:
            assert num_shot == -1
    
    elif config["dataset"] == "regex":
        num_shot = config["num_shot"]
        data_dir = f"data/regex/fewshot_num{num_shot}"
        train_filename = f"{data_dir}/train.src,{data_dir}/train.tgt"
        dev_filename = f"{data_dir}/valid.src,{data_dir}/valid.tgt"
        test_filename = f"{data_dir}/testi.src,{data_dir}/testi.tgt"

        train_examples = load_examples(train_filename)
        dev_examples = load_examples(dev_filename)
        test_examples = load_examples(test_filename)

        # normalize_program_for_all(train_examples + dev_examples + test_examples, parser)
    elif config["dataset"] == "arc":
        # Build ARC examples from solvers.py
        solvers_path = "third_party/arc-dsl/solvers.py"
        arc_examples = _extract_arc_examples_from_solvers(solvers_path)
        out_dir = "data/arc/generated"
        train_file, dev_file, test_file = _write_arc_splits(arc_examples, out_dir)
        train_examples = load_examples(train_file)
        dev_examples = load_examples(dev_file)
        test_examples = load_examples(test_file)
        if config["num_shot"] != -1:
            train_examples = train_examples[:config["num_shot"]]
    elif config["dataset"] == "folio":
        def load_fol_examples(filename):
            examples = []
            with open(filename) as f:
                for line in f:
                    example = json.loads(line)

                    nl_l = example["premises"] + [example["conclusion"]]
                    nl = "\n".join(nl_l)

                    fol_l = example["premises-FOL"] + [example["conclusion-FOL"]]
                    fol = "\n".join(fol_l)

                    label = example["label"]

                    example = Example(nl, fol)
                    example.label = label
                    examples.append(example)
            return examples
        
        # train_filename = f"data/folio/folio-train.jsonl"
        dev_filename = f"data/folio/folio-validation.jsonl"
        orig_dev_examples = load_fol_examples(dev_filename)
        random.shuffle(orig_dev_examples)

        assert config["num_shot"] != -1
        train_examples = orig_dev_examples[:config["num_shot"]]
        dev_examples = []
        test_examples = orig_dev_examples[config["num_shot"]:]

    else:
        raise ValueError(f"dataset {config['dataset']} not supported")

    ## dryrun mode
    if getattr(FLAGS, "dryrun", False):
        os.environ["WANDB_MODE"] = "dryrun"
        batch_size = 20
        dev_examples = dev_examples[:batch_size // 2]
        test_examples = test_examples[:batch_size // 2]
    
    if getattr(FLAGS, "quickrun", False):
        os.environ["WANDB_MODE"] = "dryrun"
        test_examples = test_examples[:100]
    
    logger.info(f"num train examples: {len(train_examples)}, num dev examples: {len(dev_examples)}, num test examples: {len(test_examples)}")
    
    return train_examples, dev_examples, test_examples

def load_sem_parser(config):
    if config["dataset"] == "geoquery":
        grammar_file = "grammars/geo.lark"
        global_parser = EarleyParser.open(grammar_file, start='query', keep_all_tokens=True)
    elif config["dataset"] == "smc":
        grammar_file, start_symbol = "grammars/lispress_full_3.lark", "call"
        global_parser = EarleyParser.open(grammar_file, start=start_symbol, keep_all_tokens=True)
    elif config["dataset"] == "mtop":
        domain = config["split_name"].split("-")[1]
        grammar_file, start_symbol = f"grammars/mtop/{domain}.lark", "query"
        global_parser = EarleyParser.open(grammar_file, start=start_symbol, keep_all_tokens=True)
    elif config["dataset"] == "regex":
        # grammar_file = "grammars/regex_simple.lark"
        grammar_file = "grammars/regex_medium.lark"
        # grammar_file = "grammars/regex_hard.lark"
        global_parser = EarleyParser.open(grammar_file, start='regex', keep_all_tokens=True)
    elif config["dataset"] == "overnight":
        domain = config["domain"]
        grammar_file, start_symbol = f"grammars/overnight/{domain}.lark", "list_value"
        global_parser = EarleyParser.open(grammar_file, start=start_symbol, keep_all_tokens=True)
    elif config["dataset"] == "folio":
        grammar_file, start_symbol = f"grammars/fol.lark", "formula"
        global_parser = EarleyParser.open(grammar_file, start=start_symbol, keep_all_tokens=True)
    elif config["dataset"] == "arc":
        grammar_file, start_symbol = "grammars/arc_1.lark", "start"
        global_parser = EarleyParser.open(grammar_file, start=start_symbol, keep_all_tokens=True)
    else:
        raise ValueError(f"dataset {config['dataset']} not supported")
    global_rules, _ = collect_rules_from_larkfile(grammar_file)
    return global_parser, global_rules

def counter2pred(counter):
    if len(counter) == 0:
        return None
    else:
        return counter.most_common(1)[0][0]

def evaluate_programs(predictions, examples):
    if len(examples) == 0:
        return 0.0

    counter = 0
    for prediction_counter, example in zip(predictions, examples):
        prediction = counter2pred(prediction_counter) 
        if prediction == example.target:
            counter += 1
    return counter / len(predictions)


def evaluate_grammars(grammars, examples, global_parser):
    if grammars is None or len(grammars) == 0:
        return 0.0
    
    counter = 0
    for grammar_counter, example in zip(grammars, examples):
        if grammar_counter is None or len(grammar_counter) == 0:
            continue

        lark_grammar = counter2pred(grammar_counter)
        try:
            parse_tree = global_parser.parse(example.target)
        except Exception as e:
            logger.warning(f"failed to parse target program:\n{example.target}")
            continue
        _, min_rules = extract_min_grammar_from_trees(parse_tree, return_rules=True)
        if check_grammar_correctness(min_rules, lark_grammar):
            counter += 1
    return counter / len(grammars)


def evaluate_dfa(predictions, examples):
    def unnaturalize(text):
        text = text.replace("notcontain", "notcc")
        text = text.replace("<letter>", "<let>")
        text = text.replace("<lowercase>", "<low>")
        text = text.replace("<uppercase>", "<cap>")
        text = text.replace("<number>", "<num>")
        text = text.replace("<special>", "<spec>")
        text = text.replace("constant(", "const(")
        return text

    if len(examples) == 0:
        return 0.0

    counter = 0
    for prediction_counter, example in zip(predictions, examples):
        prediction = counter2pred(prediction_counter) 
        if prediction:
            prediction = unnaturalize(prediction)
            target = unnaturalize(example.target)
            if check_equiv(prediction, target):
                counter += 1
    return counter / len(predictions)

def evaluate_fol(predictions, examples, parser):
    if len(examples) == 0:
        return 0.0
    
    counter = 0
    for prediction_counter, example in zip(predictions, examples):
        prediction = counter2pred(prediction_counter) 

        try:
            logger.disabled = True
            prediction = cleanup_fol(prediction)
            pred_premises, pred_hyp = parse_fol(prediction, parser)
            pred_res = execute_fol(pred_premises, pred_hyp)

            if isinstance(pred_res, dict):
                all_vals = list(set(v.responseStr() for v in pred_res.values()))
                if len(all_vals) == 1:
                    pred_val = all_vals[0]
                elif "Yes." in all_vals:
                    pred_val = "Yes."
                else:
                    pred_val = "I don't know."
            else:
                pred_val = pred_res.responseStr()
            logger.disabled = False
        except Exception as e:
            logger.warning(f"failed to execute predicted program:\n{prediction}")
            logger.warning(str(e))
            pred_val = "I don't know."

        tgt_premises, tgt_hyp = parse_fol(example.target, parser)
        tgt_res = execute_fol(tgt_premises, tgt_hyp)
        tgt_val = tgt_res.responseStr()

        logger.info(f"NL sentences:\n{example.source}")
        logger.info(f"predicted program:\n{prediction}")
        logger.info(f"predicted value: {pred_val}")
        logger.info(f"target program:\n{example.target}")
        logger.info(f"target value: {tgt_val}")

        if pred_val == tgt_val:
            counter += 1
    return counter / len(predictions)
