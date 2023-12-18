# -*- coding: utf-8 -*-
# @Time    : 2023/6/4 22:07
# @Author  : Ray
# @Email   : httdty2@163.com
# @File    : structure_sim.py
# @Software: PyCharm
import collections
import copy
import itertools

import networkx as nx
import json
import matplotlib.pyplot as plt
from tqdm import tqdm


def amend_missing_foreign_keys(db_instance):
    c_dict = collections.defaultdict(list)
    for idx, c in enumerate(db_instance['column_names_original']):
        c_name = c[1].lower()
        c_dict[c_name].append(idx)
        primary_keys = db_instance['primary_keys']
        # print(primary_keys)
        foreign_keys = set([tuple(sorted(x)) for x in db_instance['foreign_keys']])
        for c_name in c_dict:
            if c_name in ['name', 'id', 'code']:
                continue
            if len(c_dict[c_name]) > 1:
                for p, q in itertools.combinations(c_dict[c_name], 2):
                    if p in primary_keys or q in primary_keys:
                        if not (p, q) in foreign_keys:
                            foreign_keys.add((p, q))
                            print('added: {}-{}, {}-{}'.format(p, db_instance['column_names_original'][p],
                                                               q, db_instance['column_names_original'][q]))
                            # if num_foreign_keys_added % 10 == 0:
                            #     import pdb
                            #     pdb.set_trace()
        foreign_keys = sorted(list(foreign_keys), key=lambda x: x[0])
        db_instance['foreign_keys'] = foreign_keys


with open("./datasets/spider/tables.json") as f:
    data = json.load(f)

connected = dict()
for ins in tqdm(data):
    if ins['db_id'] != "school_finance":
        continue
    amend_missing_foreign_keys(ins)
    g = nx.Graph()

    for i, tab in enumerate(ins['table_names']):
        g.add_node(i, cat='table', name=tab.lower())

    base = len(ins['table_names'])
    for i, col in enumerate(ins['column_names']):
        if col[1] == "*":
            continue
        g.add_node(i + base, cat='column', name=col[1].lower())
        g.add_edge(col[0], i + base, edge_type="col-belong-tab")

    for fk, pk in ins['foreign_keys']:
        g.add_edge(fk + base, pk + base, edge_type="fk-pk")
    
    nx.draw(g, with_labels=True, font_weight='bold')
    plt.title(ins['db_id'], fontsize='large', fontweight='bold')
    plt.draw()
    plt.savefig(f"./tmp/{ins['db_id']}.png")
    plt.show()
    connected[ins['db_id']] = nx.is_connected(g)


# print(connected)
# nx.draw_shell(G, nlist=[range(5, 10), range(5)], with_labels=True, font_weight='bold')
#
# g2 = nx.Graph()
#
# ins = data[2]
# print(ins)
# for i, tab in enumerate(ins['table_names']):
#     g2.add_node(i, cat='table', name=tab.lower())
#
# base = len(ins['table_names'])
# for i, col in enumerate(ins['column_names']):
#     if col[1] == "*":
#         continue
#     g2.add_node(i + base, cat='column', name=col[1].lower())
#     g2.add_edge(col[0], i + base, edge_type="col-belong-tab")
#
#
# for fk, pk in ins['foreign_keys']:
#     g2.add_edge(fk + base, pk + base, edge_type="fk-pk")
#
#
# print(g2)
#
# # subax1 = plt.subplot(121)
# # nx.draw(g2, with_labels=True, font_weight='bold', )
# # subax2 = plt.subplot(122)
# # nx.draw_shell(G, nlist=[range(5, 10), range(5)], with_labels=True, font_weight='bold')
#
# g3 = copy.deepcopy(g)
# g3.remove_node(9)
# nx.draw(g3, with_labels=True, font_weight='bold', )
# plt.show()
#
# dis = nx.optimize_graph_edit_distance(g3, g)
# for i in dis:
#     print(i)
