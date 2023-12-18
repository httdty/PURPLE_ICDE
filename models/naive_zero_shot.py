# -*- coding: utf-8 -*-
# @Time    : 2023/4/10 09:12
# @Author  : Ray
# @Email   : httdty2@163.com
# @File    : naive_zero_shot.py
# @Software: PyCharm


def get_zero_shot(instance, schema, p_type):
    if p_type == "question":
        prompt = get_zero_shot_prompt_question(instance)
    elif p_type == "api":
        prompt = get_zero_shot_prompt_api(instance, schema)
    elif p_type == "3row":
        prompt = get_zero_shot_prompt_select(instance, schema)
    elif p_type == "description":
        prompt = get_zero_shot_prompt(instance, schema)
    else:
        prompt = get_zero_shot_prompt(instance, schema)
    return prompt


def get_zero_shot_prompt(instance, schema):
    task_desc = "Text2SQL task: Give you database schema and NL question, " \
                       "generate an executable SQL query for me."
    db_info = f"The database has {schema[instance['db_id']]['description']};"
    nl_info = f"The question is '{instance['question']}';"
    task_prefix = "The SQL query is: "
    prompt = f"{task_desc} {db_info} {nl_info} {task_prefix}"
    # prompt = f"{task_desc} {nl_info} {task_prefix}"
    return prompt


def get_zero_shot_prompt_question(instance):
    prompt = f"# Using valid SQLite, answer the following questions.\n" \
             f"# {instance['question']}\n" \
             f"SELECT "
    return prompt


def get_zero_shot_prompt_api(instance, schema):
    prompt = f"### SQLite SQL tables, with their properties:\n" \
             f"#\n" \
             f"# {schema[instance['db_id']]['description_table']}\n" \
             f"#\n" \
             f"### {instance['question']}\n" \
             f"SELECT "
    return prompt


def get_zero_shot_prompt_select(instance, schema):
    prompt = f'"""'
    prompt = f"{prompt}\n{schema[instance['db_id']]['description_content']}"
    prompt = f'{prompt}\n"""'
    prompt = f"{prompt}\n" \
             f"# Using valid SQLite, answer the following questions for the tables provided above.\n" \
             f"# {instance['question']}\n" \
             f"SELECT "
    return prompt
