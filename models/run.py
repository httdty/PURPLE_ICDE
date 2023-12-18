# -*- coding: utf-8 -*-
# @Time    : 2023/5/6 19:44
# @Author  : Ray
# @Email   : httdty2@163.com
# @File    : run.py
# @Software: PyCharm
import os
import copy
import json
import argparse

from datasets import Dataset
from loguru import logger
from torch.utils.data import DataLoader
from tqdm import tqdm

from bug_fix.post_fix import BugFix
from eval.spider_evaluator import EvaluateTool
from llms import model_init
from models.consistency import consistency
from models.load_data import data, load_ori_data
from models.utils import clean_output, load_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, help="Model name")
    parser.add_argument("--gpu", action="store_true", help="Enable gpu")
    parser.add_argument("--toy", action="store_true", help="Toy setting for very few instances")
    parser.add_argument("--exp_name", type=str, default="exp", help="Experiment name")
    parser.add_argument("--shot", choices=["zero", "few"], required=True, help="Zero/Few shot")
    parser.add_argument("--prompt",
                        choices=[
                            "random", "purple",
                        ],
                        default="default",
                        help="demonstration ranking")
    parser.add_argument("--enable_domain",
                        action="store_true", help="Enable domain for purple")
    parser.add_argument("--enable_skeleton",
                        action="store_true", help="Enable skeleton for purple")
    parser.add_argument("--enable_distinct",
                        action="store_true", help="Enable distinct for purple")
    parser.add_argument("--data",
                        choices=["default", "tree"],
                        default="default",
                        help="Data load strategy")
    parser.add_argument("--output_dir",
                        type=str,
                        default="./output",
                        help="Output dir")
    parser.add_argument("--train_file",
                        type=str,
                        default="./datasets/spider/train_spider.json",
                        help="train file")
    parser.add_argument("--ori_dev_file",
                        type=str,
                        default="./datasets/spider/dev.json",
                        help="dev file")
    parser.add_argument("--dev_file",
                        type=str,
                        default="./datasets/spider/dev_pruned.json",
                        help="dev file")
    parser.add_argument("--pred_skeleton",
                        type=str,
                        default="./datasets/spider/dev_skeleton.json",
                        help="dev pred skeleton file")
    parser.add_argument("--pred_domain",
                        type=str,
                        default="",
                        help="dev pred domain file")
    parser.add_argument("--table_file",
                        type=str,
                        default="./datasets/spider/tables.json",
                        help="tables file")
    parser.add_argument("--db_dir",
                        type=str,
                        default="./datasets/spider/database",
                        help="db_dir")
    parser.add_argument("--sample_db_dir",
                        type=str,
                        default="",
                        help="sample_db_dir for test suite")
    parser.add_argument("--bug_fix",
                        action="store_true", help="Enable bug fix for purple")
    parser.add_argument("--batch_size",
                        type=int,
                        default=2,
                        help="batch size")
    parser.add_argument("--consistency_num",
                        type=int,
                        default=1,
                        help="consistency size")
    parser.add_argument("--prompt_length",
                        type=int,
                        default=2048,
                        help="prompt length")
    parser.add_argument("--stage",
                        required=True,
                        choices=["dev", "test"],
                        help="LLMs inference stage")
    parser.add_argument("--api_key",
                        default="",
                        help="LLMs api key")

    # parser.add_argument("--number_param", type=int, default=0,
    #                     help="This is a number parameter with default value `0`")
    # parser.add_argument("--action_param", action="store_true",
    #                     help="This is an action parameter with action means True")
    args_ = parser.parse_args()
    return args_


def log_args(args_):
    args_dict = vars(copy.deepcopy(args_))
    arg_str = "\n"
    for k, v in args_dict.items():
        if isinstance(v, bool) and v:
            arg_str += f"--{k} "
        else:
            arg_str += f"--{k}={v} "
    logger.info("Running with parameters:" + arg_str)


def output_name(args_):
    args_dict = vars(copy.deepcopy(args_))
    name = args_dict['exp_name']
    keys = [
        'model_name', 'gpu', 'toy', 'shot', 'prompt',
        'enable_domain', 'enable_skeleton', 'enable_distinct',
        'bug_fix', 'consistency_num', 'prompt_length', 'stage'
    ]
    for k in keys:
        v = args_dict[k]
        if isinstance(v, bool):
            if v:
                name += f"_{k}"
        else:
            name += f"_{k}_{v}"
    name = name.replace(os.sep, '_')
    return name


def model_args(args_):
    return {
        "gpu": args_.gpu,
        "api_key": args_.api_key
    }


def main() -> None:
    args = parse_args()
    log_args(args)
    exp_name = output_name(args)

    # Load dev data
    # dev = Dataset.from_generator(data, gen_kwargs={"args": args})
    dev_data = data(args)
    dev = Dataset.from_list(dev_data)
    dev_loader = DataLoader(dev, batch_size=args.batch_size)

    # Init model
    model = model_init(args.model_name, **model_args(args))

    # Bug fix
    bug_fixer = None
    if args.bug_fix:
        bug_fixer = BugFix(args.db_dir, load_data(args.dev_file)[::3] if args.toy else load_data(args.dev_file))

    # Exp records
    if args.stage == 'dev':
        evaluator = EvaluateTool()
        dev_ori = load_ori_data(args)
        evaluator.register_golds(dev_ori, args.db_dir, args.sample_db_dir)
    else:
        evaluator = None
    out = open(os.path.join(args.output_dir, f"{exp_name}.txt"), 'w')
    out_log = []
    em = []
    ex = []
    ts = []

    idx = 0
    batches = tqdm(dev_loader)
    for batch_ins in batches:
        batch_raw_output = model.infer(batch_ins['prompt'], stop=";\n", n=args.consistency_num)
        for i, raw_output in enumerate(batch_raw_output):
            # Out put clean
            results = clean_output(raw_output)
            if bug_fixer:
                for res_idx in range(len(results)):
                    results[res_idx] = bug_fixer.online_fix(idx, results[res_idx])
            if len(results) > 1:
                result = consistency(results, batch_ins['db_id'][i], args.db_dir)
            else:
                result = results[0]

            # Eval
            if evaluator:
                score = evaluator.evaluate_one(idx=idx, prediction=result)
            else:
                score = {
                    'exact_match': 1,
                    'exec_match': 1,
                    'test_suite_match': 1,
                }
            em.append(score['exact_match'])
            ex.append(score['exec_match'])
            ts.append(score['test_suite_match'])
            idx += 1

            # Log info
            logger.info(batch_ins['prompt'][i])
            logger.info(result)
            logger.info(ts[-1] != 0)
            batches.desc = f"EM: {sum(em) / idx * 100:.2f}%   " \
                           f"EX: {sum(ex) / idx * 100:.2f}%   " \
                           f"TS: {sum(ts) / idx * 100:.2f}%   " \
                           f"IN: {model.prompt_length / model.count:.1f}   " \
                           f"OUT: {model.completion_length / model.count:.1f} "

            # File log
            out_log.append({
                "prompt": batch_ins['prompt'][i],
                "result": result,
                "raw_result": raw_output,
                "mark": score
            })
            out.write(f"{str(result)}\n")
            out.flush()
    # Stat info
    if evaluator:
        evaluator.print_score()
    logger.info(
        f"\nExact match\t{sum(em) / idx * 100:.2f}%"
        f"\nExec match \t{sum(ex) / idx * 100:.2f}%"
        f"\nTest suite \t{sum(ts) / idx * 100:.2f}%"
        f"\nPrompt     \t{model.prompt_length / model.count:.1f}"
        f"\nCompletion \t{model.completion_length / model.count:.1f}"
    )
    logger.info(f"Exp name: {exp_name}")
    logger.info(f"Output dir: {args.output_dir}")
    with open(os.path.join(args.output_dir, f"{exp_name}.json"), 'w') as f:
        json.dump(out_log, f, indent=4)
    if bug_fixer:
        logger.info(f"Fix and pass number: {bug_fixer.fix_pass}")
        logger.info(f"Fix but fail number: {bug_fixer.fix_fail}")
        reasons = '   '.join(list(set(bug_fixer.fail_reason)))
        logger.info(f"Failed reasons: {reasons}")

    logger.info("Prepare file for eval...")
    exp_abs = os.path.abspath(os.path.join(args.output_dir, f"{exp_name}.txt"))
    target_file = os.path.join(os.path.abspath("./"), f"predicted_sql_{args.model_name}.txt")
    os.system(f"cp {exp_abs} {target_file}")
    logger.info("Finished")


if __name__ == "__main__":
    main()
