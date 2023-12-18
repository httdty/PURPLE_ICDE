# -*- coding: utf-8 -*-
# @Time    : 2023/6/29 16:20
# @Author  : Ray
# @Email   : httdty2@163.com
# @File    : GCL_inference.py
# @Software: PyCharm
import json
import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
import argparse

from tqdm import tqdm

from db_sim.GCL_train import test, Encoder, GConv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--train_emb", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args_ = parser.parse_args()
    return args_


args = parse_args()
dev_dataset = TUDataset(args.input, name='DB', use_node_attr=True)
dev_dataloader = DataLoader(dev_dataset, batch_size=256)
encoder_model = torch.load(args.model, map_location=torch.device('cpu'))
train_emb = torch.load(args.train_emb, map_location=torch.device('cpu'))
dev_emb = test(encoder_model, dev_dataloader)

result_mapping = []
for dev_idx, emb in enumerate(tqdm(dev_emb)):
    sim = torch.mul(torch.cosine_similarity(train_emb, emb, dim=-1), -1)
    rank = sim.argsort()
    res = []
    scores = []
    for idx in rank[:len(rank) // 5]:
        res.append(idx.item())
        scores.append(sim[idx] * -1)
    result_mapping.append(res)

with open(args.output, 'w') as f:
    json.dump(result_mapping, f, indent=2)
