# -*- coding: utf-8 -*-
# @Time    : 2023/6/9 14:59
# @Author  : Ray
# @Email   : httdty2@163.com
# @File    : graph_build.py
# @Software: PyCharm

import json
import os.path
from typing import List

import dill
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from db_sim.utils import amend_missing_foreign_keys
import numpy as np
import random
import argparse
import torch

NODE_TYPE_DICT = {
    'db': 0,
    'table': 1,
    'column': 2,
    'pk': 3
}
EDGE_TYPE_DICT = {
    'db-has-tab': 0,
    'tab-belong-db': 1,
    'col-belong-tab': 2,
    'tab-has-col': 3,
    'pk-fk': 4,
    'fk-pk': 5,
}

model = None
while not model:
    try:
        if torch.cuda.is_available():
            model = SentenceTransformer('all-mpnet-base-v2', device="cuda:0")
        else: 
            print("Loading onto cpu")
            model = SentenceTransformer('all-mpnet-base-v2')
    except Exception as e:
        print(e)
        

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_files", type=str, required=True,
                        help="Train files, split with;")
    parser.add_argument("--viz_output_file", type=str, required=True,
                        help="Viz output file")
    parser.add_argument("--save_dir", type=str, required=True,
                        help="Dataset save dir")

    args_ = parser.parse_args()
    return args_


def build_graph(base, ins):
    # Parameters init
    edges = set()
    nodes_type = dict()
    nodes_embedding = dict()
    edges_type = dict()
    invert_index = dict()

    # Fix foreign keys
    ins['fk'] += amend_missing_foreign_keys(ins)  # TODO: fix

    # DB node
    nodes_type[base] = NODE_TYPE_DICT['db']
    nodes_embedding[base] = model.encode(ins["db_id"])
    offset = 0

    # Table nodes
    for tab in ins['db_schema']:
        offset += 1
        nodes_type[base + offset] = NODE_TYPE_DICT['table']
        nodes_embedding[base + offset] = model.encode(tab['table_name'])
        edges.add((base + offset, base))
        edges_type[(base + offset, base)] = EDGE_TYPE_DICT['tab-belong-db']
        edges.add((base, base + offset))
        edges_type[(base, base + offset)] = EDGE_TYPE_DICT['db-has-tab']
        invert_index[tab['table_name_original']] = base + offset
        tab_idx = base + offset
        # Column nodes
        for col_idx, col in enumerate(tab['column_names_original']):
            offset += 1
            nodes_type[base + offset] = NODE_TYPE_DICT['column']
            nodes_embedding[base + offset] = model.encode(tab['column_names'][col_idx])
            edges.add((base + offset, tab_idx))
            edges_type[(base + offset, tab_idx)] = EDGE_TYPE_DICT['col-belong-tab']
            edges.add((tab_idx, base + offset))
            edges_type[(tab_idx, base + offset)] = EDGE_TYPE_DICT['tab-has-col']
            invert_index[f"{tab['table_name_original']}.{col}"] = base + offset

    for fk in ins['fk']:
        l = invert_index[f"{fk['source_table_name_original']}.{fk['source_column_name_original']}"]
        r = invert_index[f"{fk['target_table_name_original']}.{fk['target_column_name_original']}"]
        edges.add((l, r))
        edges_type[(l, r)] = EDGE_TYPE_DICT['fk-pk']
        edges.add((r, l))
        edges_type[(r, l)] = EDGE_TYPE_DICT['pk-fk']

    # Primary keys
    for pk in ins['pk']:
        idx = invert_index[f"{pk['table_name_original']}.{pk['column_name_original']}"]
        nodes_type[idx] = NODE_TYPE_DICT['pk']
    return edges, nodes_type, edges_type, nodes_embedding


def main(db_files: List[str], output_file: str, save_dir: str):
    base = 1
    dbs = []
    for db_file in db_files:
        with open(db_file, 'r') as f:
            dbs.extend(json.load(f))
    schema_graphs = []
    idx = 1
    print("Preparing data...")
    nodes_type_list = []
    nodes_embedding_list = []
    graph_indicator = []
    graph_map = {}
    graph_label = []
    edges_type_list = []
    edges_list = []

    for db in tqdm(dbs):
        graph = build_graph(base, db)
        edges, nodes_type, edges_type, nodes_embedding = graph
        for key in sorted(nodes_type.keys()):
            nodes_type_list.append(str(nodes_type[key]))
            graph_indicator.append(str(idx))
            graph_label.append(str(random.randint(0, 9)))
            graph_map[idx] = db['db_id']

        for key in sorted(nodes_embedding.keys()):
            nodes_embedding_list.append(nodes_embedding[key])

        for key in edges_type.keys():
            edges_list.append(key)
            edges_type_list.append(str(edges_type[key]))
        # for ed in
        #     edges_type
        schema_graphs.append({
            "graph": graph,
            "schema": db
        })

        idx += 1
        base += len(graph[1])
    save_dir = os.path.join(save_dir, "DB", "raw")
    os.makedirs(save_dir, exist_ok=True)

    np.savetxt(os.path.join(save_dir, 'DB_node_attributes.txt'), nodes_embedding_list, delimiter=',')
    with open(os.path.join(save_dir, 'DB_node_labels.txt'), 'w') as f:
        for i in nodes_type_list:
            f.write(i + '\n')
    with open(os.path.join(save_dir, 'DB_graph_indicator.txt'), 'w') as f:
        for i in graph_indicator:
            f.write(i + '\n')
    with open(os.path.join(save_dir, 'DB_graph_name.txt'), 'w') as f:
        json.dump(graph_map, f)
    with open(os.path.join(save_dir, 'DB_graph_labels.txt'), 'w') as f:
        for i in graph_label:
            f.write(i + '\n')
    with open(os.path.join(save_dir, 'DB_edge_labels.txt'), 'w') as f:
        for i in edges_type_list:
            f.write(i + '\n')
    with open(os.path.join(save_dir, 'DB_A.txt'), 'w') as f:
        for test_tuple in edges_list:
            f.write(', '.join(str(t) for t in test_tuple) + '\n')
    print("Saving")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'wb') as f:
        dill.dump(schema_graphs, f)
    print(f"Processed data saved in {output_file}")


if __name__ == '__main__':
    args = parse_args()
    main(
        args.input_files.split(';'),
        args.viz_output_file,
        args.save_dir
    )
