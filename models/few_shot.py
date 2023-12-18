# -*- coding: utf-8 -*-
# @Time    : 2023/4/10 09:12
# @Author  : Ray
# @Email   : httdty2@163.com
# @File    : naive_few_shot.py
# @Software: PyCharm
import copy
import os
import time

import numpy as np
import random
import tiktoken

from functools import lru_cache
from typing import List, Dict

from models.purple_ranker import PurpleRanker
from schema_prune.bridge_content_encoder import get_column_picklist

class FewShotPrompter:
    def __init__(self, demonstrations: List, max_length: int = 2048, **kwargs):
        self.demonstrations = [self._demonstration_init(demo) for demo in demonstrations]
        self.max_length = max_length
        self.kwargs = kwargs
        try:
            self.tokenizer = tiktoken.encoding_for_model(self.kwargs.get("model_name", 'gpt-3.5-turbo'))
        except KeyError:
            self.tokenizer = tiktoken.encoding_for_model('gpt-3.5-turbo')

    @lru_cache(maxsize=20000, typed=False)
    def get_token_len(self, input_str: str):
        return len(self.tokenizer.encode(input_str))

    def _demonstration_init(self, demo):
        demo['prompt'] = self._serialize(demo, is_inference=False)
        return demo

    def _serialize(self, ins: Dict, is_inference: bool = True) -> str:
        raise NotImplementedError

    def get_prompt(self, ins):
        # Init
        task_desc = "Text2SQL task: Give you database schema and NL question, " \
                    "generate an executable SQL query for me.\n\n"
        ins_prompt = self._serialize(ins)
        icl_prompt = ""
        budget = self.max_length - self.get_token_len(ins_prompt) - self.get_token_len(task_desc)

        # Ranking demonstrations here
        choices = self._ranking(ins)

        # Budget limit
        patience = 5
        while patience > 0:
            idx = choices.pop(0)
            demo_prompt = self.demonstrations[idx]['prompt'] + "\n\n"
            demo_len = self.get_token_len(demo_prompt)
            if demo_len > budget:
                patience -= 1
                continue
            else:
                budget -= demo_len
                # icl_prompt += demo_prompt
                icl_prompt = demo_prompt + icl_prompt  # reverse ranking
        return task_desc + icl_prompt + ins_prompt

    def _ranking(self, ins) -> List[int]:
        raise NotImplementedError


class RandomFewShotPrompter(FewShotPrompter):
    def __init__(self, demonstrations: List, **kwargs):
        self.db_dir = kwargs['db_dir']
        super().__init__(demonstrations, **kwargs)

    def _serialize(self, ins: Dict, is_inference: bool = True):
        db_path = os.path.join(self.db_dir, ins['db_id'], f"{ins['db_id']}.sqlite")

        # DB info
        table_lines = []
        for table in ins['db_schema']:
            table_line = f"Here are some typical values for each column in table {table['table_name_original']}:\n" \
                         f"Table: {table['table_name_original']}\n"
            for c_idx, column in enumerate(table['column_names_original']):
                db_contents = table['db_contents'][c_idx]
                if len(db_contents) < 3:
                    vals = get_column_picklist(
                        table_name=table['table_name_original'], column_name=column, db_path=db_path
                    )
                    db_contents += vals
                table_line += f"{column}: {' , '.join([str(v) for v in db_contents[:3]])}\n"
            table_lines.append(table_line)

        if len(ins['fk']) > 0:
            fk_line = "The foreign keys:\n"
            for fk in ins['fk']:
                line = f"{fk['source_table_name_original']}.{fk['source_column_name_original']} = " \
                       f"{fk['target_table_name_original']}.{fk['target_column_name_original']}\n"
                fk_line += line
            # table_lines.append(fk_line)

        db_info = '\n'.join(table_lines)
        db_info = f"\n'''{db_info}'''"
        nl_info = f"The question is '{ins['question']}';"
        if is_inference:
            answer_info = "The SQL query is: "
        else:
            answer_info = f"The SQL query is: {ins['sql']};"
        prompt = f"{db_info}\n{nl_info}\n{answer_info}"
        return prompt

    def _ranking(self, ins) -> List[int]:
        idx_list = list(range(len(self.demonstrations)))
        random.shuffle(idx_list)
        return idx_list


class PurpleFewShotPrompter(FewShotPrompter):
    def __init__(self, demonstrations: List, **kwargs):
        self.enable_domain = kwargs.pop("enable_domain")
        self.enable_skeleton = kwargs.pop("enable_skeleton")
        self.enable_distinct = kwargs.pop("enable_distinct")
        super().__init__(demonstrations, **kwargs)
        self.purple_ranker = PurpleRanker(self.demonstrations)

    def _serialize(self, ins: Dict, is_inference: bool = True):
        # DB info
        table_lines = []
        tables = copy.deepcopy(ins['db_schema'])
        # random.shuffle(tables)
        for table in tables:
            table_line = f"Here are some typical values for each column in table '{table['table_name_original']}':\n" \
                         f"Table: {table['table_name_original']}\n"
            for c_idx, column in enumerate(table['column_names_original']):
                db_contents = table['db_contents'][c_idx]
                if is_inference:
                    table_line += f"{column}: {' , '.join([str(v) for v in db_contents[:3]])}\n"
                else:
                    table_line += f"{column}: {' , '.join([str(v) for v in db_contents[:2]])}\n"
            table_lines.append(table_line)
        if len(ins['fk']) > 0:
            fk_line = "The foreign keys:\n"
            for fk in ins['fk']:
                line = f"{fk['source_table_name_original']}.{fk['source_column_name_original']} = " \
                       f"{fk['target_table_name_original']}.{fk['target_column_name_original']}\n"
                fk_line += line
            table_lines.append(fk_line)
        db_info = '\n'.join(table_lines)
        db_info = f"'''\n{db_info}'''"
        nl_info = f"The question is '{ins['question']}';"
        if is_inference:
            answer_info = "The SQL query is: "
        else:
            answer_info = f"The SQL query is: {ins['sql']};"  # Format the SQL
        prompt = f"{db_info}\n{nl_info}\n{answer_info}"
        return prompt

    def _ranking(self, ins) -> List[int]:

        res = self.purple_ranker.demo_rank(dev_ins=ins)

        # ins_sql_skeletons = ins['sql_skeleton']
        # structures = [keyword_extract(s['generated_text'].split()) for s in ins_sql_skeletons]
        # structure_scores = [s['scores'] for s in ins_sql_skeletons]
        # weights = [1 - (structure_scores[0] - s) for s in structure_scores]
        # weights = [1 for i in range(len(weights))]
        # domain = ins['domain_linking']
        #
        # # domain_weights = [0.5, 0.1, 0.0, 0.0]  # weight_4
        # domain_weights = [0.0c, 0.0, 0.0, 0.0]  # weight_4
        #
        # scores = []
        # added_sql = set()
        # for idx, demo in enumerate(self.demonstrations):
        #     domain_scores = []
        #     skeleton_scores = []
        #
        #     # Top-k skeleton inference
        #     for structure in structures:
        #         # hardness_level = min(len(structure) // 5, 3)  # Hardness 1
        #         hardness_level = min(max(len(structure) - 1, 0) // 3, 3)  # Hardness 2
        #         domain_weight = domain_weights[hardness_level]
        #         domain_scores.append(int(idx in domain) * domain_weight)
        #         if 'structure' not in demo:
        #             demo['structure'] = keyword_extract(demo['sql_skeleton'].split())
        #         skeleton_scores.append(len(structure & demo['structure']) / len(structure | demo['structure']))
        #     structure_scores = [weights[i] * (skeleton_scores[i] + domain_scores[i]) for i in range(len(weights))]
        #     max_score = max(structure_scores)
        #     max_idx = structure_scores.index(max_score)
        #     domain_score = domain_scores[max_idx]
        #     skeleton_score = skeleton_scores[max_idx]
        #
        #     score = 0
        #     if self.enable_domain:
        #         score += domain_score
        #     elif random.random() < 0.001:
        #         print("[WARN] No domain")
        #
        #     if self.enable_skeleton:
        #         score += skeleton_score
        #     elif random.random() < 0.001:
        #         print("[WARN] No skeleton")
        #
        #     if not self.enable_domain and not self.enable_skeleton:
        #         print("[WARN] Random!")
        #         score += random.random()
        #
        #     if self.enable_distinct and demo['sql'] in added_sql:
        #         score *= 0.5
        #     scores.append(score)
        #     added_sql.add(demo['sql'])
        #
        # scores = np.array(scores) * -1
        # idx = scores.argsort()
        # res = []
        # for i in idx:
        #     if scores[i] < -0.95:
        #         res.append(i)
        # random.shuffle(res)
        # res += list(idx)[len(res):]
        return res

    # def _ranking(self, ins) -> List[int]:
    #     structure = keyword_extract(ins['sql_skeleton'].split())
    #     domain = ins['domain_linking']
    #     hardness_level = min(len(structure) // 5, 3)
    #     # domain_weights = [0.5, 0.25, 0.1, 0.0]  # old weight
    #     domain_weights = [0.5, 0.1, 0.01, 0.0]  # new weight
    #     domain_weight = domain_weights[hardness_level]
    #
    #     scores = []
    #     added_sql = set()
    #     for idx, demo in enumerate(self.demonstrations):
    #         domain_score = int(idx in domain) * domain_weight
    #         if 'structure' not in demo:
    #             demo['structure'] = keyword_extract(demo['sql_skeleton'].split())
    #         skeleton_score = len(structure & demo['structure']) / len(structure | demo['structure'])
    #         score = 0
    #         if self.enable_domain:
    #             score += domain_score
    #         if self.enable_skeleton:
    #             score += skeleton_score
    #         if self.enable_distinct and demo['sql'] in added_sql:
    #             score *= 0.5
    #         scores.append(score)
    #         added_sql.add(demo['sql'])
    #
    #     scores = np.array(scores) * -1
    #     res = scores.argsort()
    #     return list(res)


def keyword_extract(sql_skeleton: List[str]):
    idx = 0
    depth = 0
    sql_skeleton = " ".join(sql_skeleton).replace(" . ", ".").replace(" distinct ", " ").split()
    KEYWORDS = {'select', 'from', 'where', 'group', 'order', 'by', 'limit', 'intersect', 'union', 'except',
                'join', 'not', 'between', '=', '>', '<', '>=', '<=', '!=', 'in', 'like', 'is',
                'exists', '-', '+', '/', 'max', 'min', 'count', 'sum', 'avg', 'and', 'or', 'desc'}
    CLAUSE = {'where', 'intersect', 'union', 'except', 'having'}
    last_clause = None

    kw_cnt = {"": 0}
    for kw in KEYWORDS:
        kw_cnt[kw] = 0
    keywords = set()
    select_num = 0
    while idx < len(sql_skeleton):
        if sql_skeleton[idx] == '(':
            depth += 1
        elif sql_skeleton[idx] == ')':
            depth -= 1
        elif sql_skeleton[idx].lower() in KEYWORDS:
            # this_kw = sql_skeleton[idx].lower()
            # if last_clause is not None:
            #     keywords.add(f"{this_kw}_{last_clause}_{kw_cnt[this_kw]}")
            # else:
            #     keywords.add(f"{this_kw}_{kw_cnt[this_kw]}")
            # if sql_skeleton[idx].lower() in CLAUSE:
            #     last_clause = f"{this_kw}_{kw_cnt[this_kw]}"
            keywords.add(f"{sql_skeleton[idx].lower()}_{depth}_{kw_cnt[sql_skeleton[idx].lower()]}")
            kw_cnt[sql_skeleton[idx].lower()] += 1
        elif kw_cnt["from"] == 0 and depth == 0 and sql_skeleton[idx].lower() != ',':
            select_num += 1
        idx += 1
    keywords.add(f"select_col_{select_num}")
    return keywords
