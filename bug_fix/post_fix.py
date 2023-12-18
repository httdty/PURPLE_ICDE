# -*- coding: utf-8 -*-
# @Time    : 2023/7/24 19:13
# @Author  : Ray
# @Email   : httdty2@163.com
# @File    : post_fix.py
# @Software: PyCharm

import argparse
import copy
import json
import os
import sqlite3
import time
from typing import List
from rapidfuzz import fuzz


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_file", type=str, required=True,
                        help="Prediction file generated by LLMs")
    parser.add_argument("--input_file", type=str, required=True,
                        help="processed file, such as path to `dev.json`")
    parser.add_argument("--db_dir", type=str, required=True,
                        help="DB dir")
    parser.add_argument("--output_file", type=str, default="",
                        help="output file path")

    args_ = parser.parse_args()
    return args_


class BugFix:
    COL_TAB_MISMATCH = "no such column: "
    NO_FUNCTION_CALL = "no such function: "
    AMBIGUOUS_COLUMN = "ambiguous column name: "
    FUN_CALL_NUMBER = "wrong number of arguments to function "


    def __init__(self, db_dir: str, ins_list=None, patience: int = 5, verbose: int = False):
        if ins_list is None:
            ins_list = []
        self.db_dir = db_dir
        self.patience = patience
        self.fix_pass = 0
        self.fix_fail = 0
        self.fail_reason = []
        self.ins_list = copy.deepcopy(ins_list)
        self.verbose = verbose

    def online_fix(self, idx: int, sql: str):
        if not self.ins_list:
            return sql

        ins = self.ins_list[idx]
        try:
            sql = self.try_fix(sql, ins)
        except Exception as e:
            print(e)
        return sql

    def offline_fix(self, sql, ins):
        try:
            sql = self.try_fix(sql, ins)
        except Exception as e:
            print(e)
        return sql

    def try_fix(self, sql, ins):
        p = self.patience
        db_path = os.path.join(self.db_dir, ins['db_id'], ins['db_id'] + ".sqlite")
        has_bug = True
        while has_bug and p > 0:
            try:
                conn = sqlite3.connect(db_path)
                conn.text_factory = bytes
                c = conn.cursor()
                c.execute(sql)
                c.fetchall()
                has_bug = False
                if p < self.patience:
                    self.fix_pass += 1
            except Exception as e:
                error_msg = str(e)
                if sql.endswith(";"):
                    sql = sql[:-1].strip()
                if error_msg.startswith(self.COL_TAB_MISMATCH):
                    error_col = error_msg.replace(self.COL_TAB_MISMATCH, "").strip()
                    sql = self.fix_column(error_col, sql, ins)
                elif error_msg.startswith(self.AMBIGUOUS_COLUMN):
                    error_col = error_msg.replace(self.AMBIGUOUS_COLUMN, "").strip()
                    sql = self.fix_column(error_col, sql, ins)
                elif error_msg.startswith(self.NO_FUNCTION_CALL):
                    error_col = error_msg.replace(self.NO_FUNCTION_CALL, "").strip()
                    sql = self.fix_function(error_col, sql, ins)
                elif error_msg.startswith(self.FUN_CALL_NUMBER):
                    error_col = error_msg.replace(self.FUN_CALL_NUMBER, "").strip()
                    error_col = error_col.replace("()", "").strip()
                    sql = self.fix_function_number(error_col, sql, ins)
                else:
                    # error_col = error_msg.replace(self.NO_FUNCTION_CALL, "").strip()
                    # sql = self.fix_function(error_col, sql, ins)
                    self.fail_reason.append(str(e))
                p -= 1
        if not sql.strip().endswith(";"):
            sql = sql.strip() + ";"
        if p == 0 and has_bug:
            self.fix_fail += 1
        if p < self.patience and self.verbose:
            print(sql)
        return sql

    @staticmethod
    def get_table_alias(sql_toks: List[str], used_tables):
        """Scan the index of 'as' and build the map for all alias"""
        as_idx_list = [idx for idx, tok in enumerate(sql_toks) if tok.lower() == "as"]
        alias = {}
        reversed_alias = {}
        for idx in as_idx_list:
            if sql_toks[idx - 1].lower() in used_tables:
                alias[sql_toks[idx + 1]] = sql_toks[idx - 1].lower()
                reversed_alias[sql_toks[idx - 1].lower()] = sql_toks[idx + 1]
        return alias, reversed_alias

    @staticmethod
    def get_from_table(sql_toks: List[str], ins):
        all_tables = {tab['table_name_original'].lower() for tab in ins['db_schema']}
        used_tables = set()
        keywords = {"select", "from", "where", "group", "order", "limit"}
        current_keyword = ""
        for idx, tok in enumerate(sql_toks):
            if tok.lower() in keywords:
                current_keyword = tok.lower()
            elif current_keyword == 'from' and tok.lower() in all_tables:
                used_tables.add(tok.lower())
        return used_tables


    def fix_column(self, error_col, sql, ins) -> str:
        sql = sql.replace(",", " , ").replace("(", " ( ").replace(")", " ) ")
        sql_tokens = sql.split()
        error_idx = sql_tokens.index(error_col)
        # Get basic info
        used_tables = self.get_from_table(sql_tokens, ins)
        tables = {
            tab['table_name_original'].lower(): {
                col.lower() for col in tab['column_names_original']
            }
            for tab in ins['db_schema'] if tab['table_name_original'] in used_tables
        }

        _, reversed_alias = self.get_table_alias(sql_tokens, used_tables)
        error_col = error_col.split(".")[-1]
        fixed_tab = ""

        # Find the col's table
        for tab, tab_col in tables.items():
            if error_col in tab_col:
                fixed_tab = tab.lower()
                break

        if not fixed_tab:  # Not find such col
            # Ambiguous fix
            most_sim_col = ""
            score = 0
            for tab, cols in tables.items():
                for c in cols:
                    tmp_score = fuzz.ratio(error_col, c)
                    if tmp_score > score:
                        score = tmp_score
                        most_sim_col = c
                        fixed_tab = tab
            error_col = most_sim_col.lower()

        # Replace alias
        if fixed_tab in reversed_alias:
            fixed_tab = reversed_alias[fixed_tab]

        # Token fix
        fixed_tok = f"{fixed_tab}.{error_col}" if fixed_tab else f"{error_col}"
        sql_tokens[error_idx] = fixed_tok
        return " ".join(sql_tokens).replace(" , ", ", ").replace(" ( ", "(").replace(" ) ", ") ")


    def fix_function(self, error_fun, sql: str, ins) -> str:
        # Get all columns
        used_tables = self.get_from_table(sql.split(), ins)
        all_columns = set()
        for tab in ins['db_schema']:
            if tab['table_name_original'].lower() in used_tables:
                for col in tab['column_names_original']:
                    all_columns.add(col.lower())

        # Location function call
        sql_segs = sql.split(error_fun, maxsplit=1)
        start_idx = sql_segs[1].index('(')
        end_idx = 0
        depth = 0

        # Get the call body
        for idx, c in enumerate(sql_segs[1][start_idx:]):
            if c == '(':
                depth += 1
            elif c == ')':
                depth -= 1
            if depth == 0:
                end_idx = start_idx + idx 
                break

        # Get tail part
        tail = sql_segs[1][end_idx + 1:].strip().split()
        if tail[0].lower() == "as":
            tail = tail[2:]
        tail = " ".join(tail)

        # Fix
        fun_params = sql_segs[1][start_idx + 1: end_idx].split(",")
        fix_params = []
        for param in fun_params:
            param = param.strip()
            if "." in param:
                param = param.split(".")[-1]
            if param.lower() in all_columns:
                fix_params.append(param.lower())
        fix_params = ", ".join(fix_params)

        sql = " ".join((sql_segs[0] + " " + fix_params + " " + tail).split())

        # Reload
        return sql

    def fix_function_number(self, error_fun, sql: str, ins) -> str:
        # Get all columns
        used_tables = self.get_from_table(sql.split(), ins)
        all_columns = set()
        for tab in ins['db_schema']:
            if tab['table_name_original'].lower() in used_tables:
                for col in tab['column_names_original']:
                    all_columns.add(col.lower())

        # Location function call
        sql_segs = sql.split(error_fun, maxsplit=1)
        start_idx = sql_segs[1].index('(')
        end_idx = 0
        depth = 0

        # Get the call body
        for idx, c in enumerate(sql_segs[1][start_idx:]):
            if c == '(':
                depth += 1
            elif c == ')':
                depth -= 1
            if depth == 0:
                end_idx = start_idx + idx
                break

        # Get tail part
        tail = sql_segs[1][end_idx + 1:].strip().split()
        if tail[0].lower() == "as":
            tail = tail[2:]
        tail = " ".join(tail)

        # Fix
        fun_params = sql_segs[1][start_idx + 1: end_idx].replace(",", " ").split()
        have_distinct = False
        fix_params = []
        for param in fun_params:
            if param.lower() == 'distinct':
                have_distinct = True
            elif param.lower() in all_columns:
                param = f"{error_fun.upper()}(DISTINCT {param})" if have_distinct else f"{error_fun}({param})"
                fix_params.append(param)
        fix_params = ", ".join(fix_params)
        sql = " ".join((sql_segs[0] + " " + fix_params + " " + tail).split())
        return sql


def main():
    args = parse_args()
    with open(args.pred_file, 'r') as f:
        preds = f.readlines()
    with open(args.input_file, 'r') as f:
        dev_ori = json.load(f)
    assert len(preds) == len(dev_ori)
    fixer = BugFix(db_dir=args.db_dir)
    fixed_sqls = []
    for idx, (sql, ins) in enumerate(list(zip(preds, dev_ori))[:]):
        fixed_sqls.append(fixer.offline_fix(sql.strip(), ins))

    print("Fix and pass number:", fixer.fix_pass)
    print("Fix but fail number:", fixer.fix_fail)
    print("Failed reasons:", json.dumps(fixer.fail_reason, indent=2))
    if not args.output_file:
        args.output_file = "".rsplit(args.pred_file, maxsplit=1)[0] + "_fixed.txt"
    with open(args.output_file, 'w') as f:
        f.writelines(fixed_sqls)


if __name__ == '__main__':
    main()