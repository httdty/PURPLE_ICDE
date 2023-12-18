# -*- coding: utf-8 -*-
# @Time    : 2023/5/6 19:47
# @Author  : Ray
# @Email   : httdty2@163.com
# @File    : load_data.py
# @Software: PyCharm
import json

from tqdm import tqdm

from models.few_shot import RandomFewShotPrompter, PurpleFewShotPrompter
# from models.naive_few_shot import get_few_shot
from models.naive_zero_shot import get_zero_shot
from models.utils import load_data, load_schema  # , load_processed_data


def data(args):
    load_data_strategies = {
        'default': load_data_default,
    }
    return load_data_strategies[args.data](args)


def load_data_default(args):
    dev = load_data(args.dev_file)
    # dev = load_data(args.dev_file)[:9]
    if args.toy:
        dev = dev[::3]

    if args.shot == "zero":
        instances = load_zero_shot(dev, args)
    elif args.shot == "few":
        instances = load_few_shot(dev, args)
    else:
        raise ValueError(f"Can not handle shot as '{args.shot}'")

    return instances

def load_ori_data(args):
    dev = load_data(args.ori_dev_file)
    # dev = load_data(args.ori_dev_file)[:9]
    if args.toy:
        dev = dev[::3]

    return dev


def load_zero_shot(dev, args):
    schema = load_schema(args.table_file, args.db_dir)

    instances = []
    for instance in dev:
        prompt = get_zero_shot(instance, schema, args.prompt)
        instances.append(
            {
                "prompt": prompt,
                "gold": instance["query"],
                "db_id": instance["db_id"],
                "instance": instance
            }
        )
    return instances

def load_few_shot(dev, args):
    # Preprocess dev set
    if args.enable_skeleton:
        with open(args.pred_skeleton, 'r') as f:
            pred_skeleton = json.load(f)
            # pred_skeleton = json.load(f)[:9]
            if args.toy:
                pred_skeleton = pred_skeleton[::3]
        assert len(pred_skeleton) == len(dev), "pred_skeleton must match the dev number"
        for ins, skeleton in zip(dev, pred_skeleton):
            # ins['sql_skeleton'] = skeleton[0]['generated_text']
            ins['sql_skeleton'] = skeleton

    # Init prompter
    demonstrations = load_data(args.train_file)
    prompter = None
    if args.prompt == 'random':
        prompter = RandomFewShotPrompter(
            demonstrations,
            max_length=args.prompt_length,
            model_name=args.model_name,
            db_dir=args.db_dir
        )
    elif args.prompt == 'purple':
        prompter = PurpleFewShotPrompter(
            demonstrations,
            max_length=args.prompt_length,
            model_name=args.model_name,
            enable_domain=args.enable_domain,
            enable_skeleton=args.enable_skeleton,
            enable_distinct=args.enable_distinct,
        )

    # Prompt gen
    instances = []
    print("Prepare prompt for the input...")
    for idx, instance in tqdm(enumerate(dev)):
    # for idx, instance in tqdm(enumerate(dev[200:])):
        prompt = prompter.get_prompt(instance)
        print(idx)
        print(prompt)
        instances.append(
            {
                "prompt": prompt,
                "gold": instance["sql"],
                "db_id": instance["db_id"],
            }
        )
    return instances
