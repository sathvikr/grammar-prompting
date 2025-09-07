# Auto-generated from log/sempar-rule-icl/add_True-bat_8-con_True-con_False-dat_arc-dom_None-dry_False-eng_openai/gpt-4-fre_0.0-ite_None-ite_iter-ite_None-laz_False-llm_llm_cache-max_100-num_1-num_100-num_-1-pro_None-pro_rot-pro_wrule-qui_False-ret_bm25-rul_0.0-see_1-sep_False-spl_None-tem_0.0-use_False-use_False-use_False/results.json
from pathlib import Path
from typing import List


def p(grid: List[List[int]]) -> List[List[int]]:
    # Convert to tuples for ARC DSL
    G = tuple(tuple(r) for r in grid)
    # Program predicted by the model
    program = "objects(I, T, F, F) ## colorfilter(x1, ZERO) ## rbind(bordering, I) ## compose(flip, x3) ## mfilter(x2, x4) ## fill(I, FOUR, x5)"
    # Run interpreter
    from neural_lark.arc_interpreter import interpret_program
    project_root = Path(__file__).resolve().parents[1]
    result, _ = interpret_program(program, G, project_root)
    # Back to lists
    return [list(r) for r in result]
