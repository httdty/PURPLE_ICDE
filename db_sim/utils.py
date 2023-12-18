# -*- coding: utf-8 -*-
# @Time    : 2023/6/10 11:48
# @Author  : Ray
# @Email   : httdty2@163.com
# @File    : utils.py
# @Software: PyCharm
import collections
import itertools
import json


def amend_missing_foreign_keys(instance, verbose=False):
    schema = instance['db_schema']
    c_dict = collections.defaultdict(list)
    pks = [f"{pk['table_name_original']}.{pk['column_name_original']}" for pk in instance['pk']]
    fks = [
        (
            f"{fk['source_table_name_original']}.{fk['source_column_name_original']}",
            f"{fk['target_table_name_original']}.{fk['target_column_name_original']}"
        ) for fk in instance['fk']
    ]

    for idx, table in enumerate(schema):
        for col in table['column_names_original']:
            c_dict[col].append(idx)
    added_fks = []
    for col in c_dict:
        if col in {'name', 'id', 'code'}:
            continue
        if len(c_dict[col]) > 1:
            for p, q in itertools.combinations(c_dict[col], 2):
                p_str = f"{schema[p]['table_name_original']}.{col}"
                q_str = f"{schema[q]['table_name_original']}.{col}"
                if p_str in pks or q_str in pks:
                    if (p_str, q_str) in fks or (q_str, p_str) in fks:
                        continue
                    else:
                        if q_str in pks:
                            added_fks.append(
                                {
                                    "source_table_name_original": schema[p]['table_name_original'],
                                    "source_column_name_original": col,
                                    "target_table_name_original": schema[q]['table_name_original'],
                                    "target_column_name_original": col,
                                }
                            )
                            fks.append((p_str, q_str))
                        else:
                            added_fks.append(
                                {
                                    "source_table_name_original": schema[q]['table_name_original'],
                                    "source_column_name_original": col,
                                    "target_table_name_original": schema[p]['table_name_original'],
                                    "target_column_name_original": col,
                                }
                            )
                            fks.append((q_str, p_str))
    if added_fks:
        print("==================================================")
        print(json.dumps(added_fks, indent=2))
    return added_fks

