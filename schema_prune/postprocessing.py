# -*- coding: utf-8 -*-
# @Time    : 2023/7/6 09:52
# @Author  : Ray
# @Email   : httdty2@163.com
# @File    : postprocessing.py
# @Software: PyCharm

import argparse
import copy
import json
import os
import random

import numpy as np

from tqdm import tqdm

from schema_prune.bridge_content_encoder import get_column_picklist


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_file', type=str, default="./datasets/spider/train_spider_with_probs.json")
    parser.add_argument('--db_dir', type=str, default="./datasets/spider/database")
    parser.add_argument('--force_true', action="store_true", default=False)
    parser.add_argument('--output_file', type=str)

    args_ = parser.parse_args()
    return args_


def prune(ins_with_prob: dict, db_dir: str, force_true: bool = False):
    # Select tables
    selected_tables = []
    table_prob = np.array(ins_with_prob['table_pred_probs'])
    if force_true:
        table_prob += np.array(ins_with_prob['table_labels'])
    table_rank_idx = table_prob.argsort()[::-1]
    adj_tables = set()
    added_table_idx = set()
    for idx in table_rank_idx:
        if table_prob[idx] >= 0.5:
            added_table_idx.add(idx)
            table = copy.deepcopy(ins_with_prob['db_schema'][idx])
            table['column_pred_probs'] = ins_with_prob['column_pred_probs'][idx]
            selected_tables.append(table)
            table_name = ins_with_prob['db_schema'][idx]['table_name_original']
            for fk in ins_with_prob['fk']:
                if table_name == fk['source_table_name_original']:
                    adj_tables.add(fk['target_table_name_original'])
                elif table_name == fk['target_table_name_original']:
                    adj_tables.add(fk['source_table_name_original'])
        else:
            table_name = ins_with_prob['db_schema'][idx]['table_name_original']
            if table_name in adj_tables:
                table = copy.deepcopy(ins_with_prob['db_schema'][idx])
                table['column_pred_probs'] = ins_with_prob['column_pred_probs'][idx]
                selected_tables.append(table)
                added_table_idx.add(idx)
            break
    if 'table_labels' in ins_with_prob and len(ins_with_prob['table_pred_probs']) == len(ins_with_prob['db_schema']):
        for idx in added_table_idx:
            ins_with_prob['table_labels'][idx] = 0
        if sum(ins_with_prob['table_labels']) > 0:
            print(json.dumps(ins_with_prob, indent=2))

    # column prune
    all_columns = set()
    db_path = os.path.join(db_dir, ins_with_prob['db_id'], f"{ins_with_prob['db_id']}.sqlite")
    pks = [f"{pk['table_name_original']}.{pk['column_name_original']}" for pk in ins_with_prob['pk']]
    for table in selected_tables:
        column_prob = np.array(table['column_pred_probs'])
        column_rank_idx = column_prob.argsort()[::-1]
        column_names = []
        column_names_original = []
        column_types = []
        db_contents = []
        for i, idx in enumerate(column_rank_idx):
            full_name = f"{table['table_name_original']}.{table['column_names_original'][idx]}"
            if column_prob[idx] >= 0.5 or i <= 5 or full_name in pks:
                column_names.append(table['column_names'][idx])
                column_names_original.append(table['column_names_original'][idx])
                column_types.append(table['column_types'][idx])
                # db_contents.append(table['db_contents'][idx])
                all_columns.add(f"{table['table_name_original']}.{table['column_names_original'][idx]}")
                values = get_column_picklist(table['table_name_original'], table['column_names_original'][idx], db_path)
                random.shuffle(values)
                values = ["" if str(v) in table['db_contents'][idx] else str(v) for v in values[:5]]
                while "" in values:
                    values.remove("")
                values = table['db_contents'][idx] + values
                db_contents.append(values[:5])

        table['column_names'] = column_names
        table['column_names_original'] = column_names_original
        table['column_types'] = column_types
        table['db_contents'] = db_contents
        table.pop('column_pred_probs')
    ins_with_prob['db_schema'] = selected_tables

    # fks prune
    fks = []
    for fk in ins_with_prob['fk']:
        if f"{fk['source_table_name_original']}.{fk['source_column_name_original']}" in all_columns and \
            f"{fk['target_table_name_original']}.{fk['target_column_name_original']}" in all_columns:
            fks.append(fk)
    ins_with_prob['fk'] = fks

    # pks prune
    pks = []
    for pk in ins_with_prob['pk']:
        if f"{pk['table_name_original']}.{pk['column_name_original']}" in all_columns:
            pks.append(pk)
    ins_with_prob['pk'] = pks
    if "table_labels" in ins_with_prob:
        ins_with_prob.pop('table_labels')
    if "column_labels" in ins_with_prob:
        ins_with_prob.pop('column_labels')
    ins_with_prob.pop('column_pred_probs')
    ins_with_prob.pop('table_pred_probs')
    return ins_with_prob


def main():
    args = parse_args()
    with open(args.input_file, 'r') as f:
        data = json.load(f)
    res = []
    for idx, ins in tqdm(enumerate(data)):
        res.append(prune(ins, args.db_dir, args.force_true))

    with open(args.output_file, 'w') as f:
        json.dump(res, f, indent=2)


if __name__ == '__main__':
    main()
