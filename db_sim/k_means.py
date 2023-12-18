# -*- coding: utf-8 -*-
# @Time    : 2023/5/26 09:25
# @Author  : Ray
# @Email   : httdty2@163.com
# @File    : k_means.py
# @Software: PyCharm
from models.utils import load_schema
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans


def db_info(schema):
    """ Generate description of database.

    Args:
        schema: description of database schema

    Returns:
        list of description lines
    """
    lines = []
    tables = schema['table_names_original']
    all_columns = schema['column_names_original']
    nr_tables = len(tables)
    for tbl_idx in range(nr_tables):
        tbl_name = tables[tbl_idx]
        tbl_name = tbl_name.lower()

        table_columns = [c[1] for c in all_columns if c[0] == tbl_idx]
        table_columns = [c.lower() for c in table_columns]
        quoted_columns = ["'" + c + "'" for c in table_columns]
        col_list = ', '.join(quoted_columns)
        line = f"Table '{tbl_name}' with columns {col_list}"
        lines.append(line)
    return lines


schema = load_schema("./datasets/spider/tables.json", "./datasets/spider/database")
model = SentenceTransformer('all-mpnet-base-v2')

corpus = []

for i in schema.keys():
    corpus.append(i + ":" + schema[i]['description'])

sentence_embeddings = model.encode(corpus)

num_clusters = 10
clustering_model = KMeans(n_clusters=num_clusters)
clustering_model.fit(sentence_embeddings)
cluster_assignment = clustering_model.labels_

clustered_sentences = [[] for i in range(num_clusters)]
for sentence_id, cluster_id in enumerate(cluster_assignment):
    clustered_sentences[cluster_id].append(corpus[sentence_id])

res = []

for i in range(len(clustered_sentences)):
    res.append([])
    for j in clustered_sentences[i]:
        res[i].append(j.split(":")[0])

for i, cluster in enumerate(res):
    print("Cluster ", i+1)
    print(cluster)
    print("")