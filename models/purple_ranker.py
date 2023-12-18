# -*- coding: utf-8 -*-
# @Time    : 2023/10/27 16:01
# @Author  : Ray
# @Email   : httdty2@163.com
# @File    : purple_ranker.py
# @Software: PyCharm
import copy
import json
import random
# import sqlfluff
from typing import List

from anytree import Node
# from sqlfluff.api import APIParsingError
from tqdm import tqdm


class PurpleRanker:
    all_units = [
        'START',
        'count', 'max', 'avg', 'sum', 'min',
        'except', 'union', 'intersect',
        'select', 'from', 'where', 'group', 'having', 'order', 'limit',
        'asc', 'desc',
        'and', 'or', 'on', 'join',
        '<=', '!=', '<', '>', '=', '>=', 'in', 'like', 'not', 'between',
        '_', '-', '+', ')', '(', 'by', 'distinct',
        'END',
        ','
    ]

    abs_1 = {
        ',', '',
        '_', ''
    }

    abs_2 = {
        '<=': 'cmp',
        '!=': 'cmp',
        '<': 'cmp',
        '>': 'cmp',
        '=': 'cmp',
        '>=': 'cmp',
        'like': 'cmp',
        'not in': 'cmp',
        'not': 'cmp',
        'in': 'cmp',
        'between': 'cmp',
        '-': 'op',
        '+': 'op',
        'except': 'iue',
        'union': 'iue',
        'intersect': 'iue',
        'count': 'agg',
        'max': 'agg',
        'avg': 'agg',
        'sum': 'agg',
        'min': 'agg'
    }

    abs_3 = {
        ')': '',
        '(': '',
        'by': '',
        'distinct': '',
        'on': '',
        'cmp': '',
        'op': '',
        'agg': '',
        'join': '',
        'and': '',
        'or': '',
        'asc': '',
        'desc': ''
    }

    def __init__(self, demonstrations, levels=None):
        if levels is None:
            levels = [0, 1, 2, 3]
        self.demonstrations = demonstrations
        self.levels = levels
        self.trie = {
            0: Node("START_0", count=0),
            1: Node("START_1", count=0),
            2: Node("START_2", count=0),
            3: Node("START_3", count=0),
        }
        self.abs_process = {
            0: self._abs_0,
            1: self._abs_1,
            2: self._abs_2,
            3: self._abs_3,
        }
        for level in levels:
            self._build_trie(level)

    def _build_trie(self, level: int):
        for i, demo in tqdm(enumerate(self.demonstrations), desc=f"Trie build for abs_{level}"):
            skeleton = self.abs_process[level](demo['sql_skeleton'])  # type: ignore
            parent = self.trie[level]
            child = None
            parent.count += 1  # type: ignore
            for tok in skeleton.split() + ['END']:

                children = [child.name for child in parent.children]
                if tok in children:
                    idx = children.index(tok)
                    child = parent.children[idx]
                else:
                    child = Node(tok, parent=parent, count=0)
                    if child.name == "END":
                        child.ins = []
                parent = child
                parent.count += 1
            child.ins.append(i)
        # from anytree import RenderTree
        # print(RenderTree(self.trie[level]))

    def _abs_0(self, ori_skeleton):
        sql_skeleton = []

        for tok in ori_skeleton.split():
            if tok not in self.all_units:
                continue

            sql_skeleton.append(tok)
        return " ".join(sql_skeleton)

    def _abs_1(self, ori_skeleton):
        ori_skeleton = self._abs_0(ori_skeleton)
        sql_skeleton = []

        for tok in ori_skeleton.split():
            if tok in self.abs_1:
                continue
            sql_skeleton.append(tok)
        return " ".join(sql_skeleton)

    def _abs_2(self, ori_skeleton):
        ori_skeleton = self._abs_1(ori_skeleton)
        sql_skeleton = []

        for tok in ori_skeleton.split():
            if tok in self.abs_2:
                tok = self.abs_2[tok]
            sql_skeleton.append(tok)
        return " ".join(sql_skeleton)

    def _abs_3(self, ori_skeleton):
        ori_skeleton = self._abs_2(ori_skeleton)
        sql_skeleton = []

        for tok in ori_skeleton.split():
            if tok in self.abs_3:
                tok = self.abs_3[tok]
            sql_skeleton.append(tok)
        return " ".join(sql_skeleton)

    def trie_search(self, skeleton: str, level: int = 0):
        skeleton = self.abs_process[level](skeleton)  # type: ignore
        return self._sub_tree_search(self.trie[level], skeleton.split() + ['END'])

    def _sub_tree_search(self, sub_tree: Node, toks: List[str]):
        children = [child.name for child in sub_tree.children]
        tok = toks.pop(0)
        if tok in children:
            child = sub_tree.children[children.index(tok)]
        else:
            return None

        if child.name == 'END':
            return child
        else:
            return self._sub_tree_search(child, toks)

    def demo_rank(self, dev_ins, level=4):
        res = []
        ins_lists = []
        ins_sql_skeletons = [s['generated_text'] for s in dev_ins['sql_skeleton']][:4]

        for abs_level in range(level):
            for i, skeleton in enumerate(ins_sql_skeletons):
                end = self.trie_search(skeleton, abs_level)
                if end:
                    ins_lists.append(copy.deepcopy(end.ins))


        priority = 1
        while len(ins_lists) > 0 and len(res) <= 98:
            for candidates in ins_lists[:priority]:
                random.shuffle(candidates)
                while candidates:  # based on the count number? or mix together for random?
                    choice = candidates.pop(0)
                    if choice not in res and (choice + 1) not in res and (choice - 1) not in res:
                        res.append(choice)
                        break
            for i in range(len(ins_lists) - 1, -1, -1):
                if len(ins_lists[i]) == 0:
                    ins_lists.pop(i)
            priority += 1

        # Random for others
        tail = list(set(list(range(len(self.demonstrations)))) - set(res))
        random.shuffle(tail)
        res += tail
        return res


def main():
    dev = "./datasets/spider/dev_pruned.json"
    with open(dev, 'r') as f:
        dev = json.load(f)

    train = "./datasets/spider/train_spider_pruned.json"
    with open(train, 'r') as f:
        train = json.load(f)

    pred_raw = "./datasets/spider/dev_skeleton_8.json"
    with open(pred_raw, 'r') as f:
        pred_raw = json.load(f)
    pred = []
    for i in range(4):
        pred += [p[i]['generated_text'] for p in pred_raw]

    pr = PurpleRanker(train)
    # print(RenderTree(pr.trie[0]))

    # Trie search
    for i, p in tqdm(enumerate(pred)):
        end = pr.trie_search(p, 3)
        if end:
            # print(i, dev[i % 1034]['sql_skeleton'] == p)
            if dev[i % 1034]['sql_skeleton'] != p and len(dev[i % 1034]['sql_skeleton']) != len(p):
                print(i, i // 1034, False, len(end.ins))
                print(dev[i % 1034]['sql_skeleton'])
                print(p)
            pass
        else:
            print(i, i // 1034, None, p)


if __name__ == '__main__':
    main()
