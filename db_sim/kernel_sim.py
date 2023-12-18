# -*- coding: utf-8 -*-
# @Time    : 2023/6/9 14:58
# @Author  : Ray
# @Email   : httdty2@163.com
# @File    : kernel_sim.py
# @Software: PyCharm
import json

import dill
from grakel.kernels import WeisfeilerLehman, VertexHistogram, RandomWalkLabeled, NeighborhoodSubgraphPairwiseDistance, NeighborhoodHash
from grakel.kernels import RandomWalk as GraphKernel
import numpy as np


with open("./datasets/spider/schema_graph.bin", 'rb') as f:
    schema_graph = dill.load(f)

train_dbs = set()
with open("./datasets/spider/train_spider.json", 'r') as f:
    train_data = json.load(f)
    for ins in train_data:
        train_dbs.add(ins['db_id'])

train_graph = []
train_schema = []
test_graph = []
test_schema = []
for ins in schema_graph:
    if ins['schema']['db_id'] in train_dbs:
        train_graph.append(ins['graph'])
        train_schema.append(ins['schema'])
    else:
        test_graph.append(ins['graph'])
        test_schema.append(ins['schema'])


wl_kernel = WeisfeilerLehman(n_iter=5, normalize=True, base_graph_kernel=GraphKernel)
K_train = wl_kernel.fit_transform(train_graph)
K_test = wl_kernel.transform(test_graph)

print(K_train)
print(K_test)

scores = np.dot(K_test, K_train)
for i in range(len(scores)):
    sim_id = scores[i].argsort()[::-1][:5]
    print(test_schema[i]['db_id'], "\t", " ".join([train_schema[s_id]["db_id"] for s_id in sim_id]))


# TODO: Here we get the bad sim score, but we want to check the real sim between them.
#  I'd like to measure them by the viz. But need to viz them in the above style~
