# -*- coding: utf-8 -*-
# @Time    : 2023/4/10 09:13
# @Author  : Ray
# @Email   : httdty2@163.com
# @File    : __init__.py.py
# @Software: PyCharm
import torch

from llms.openai import Openai


def model_init(model_name: str, **kwargs):
    if "llama" in model_name.lower():
        pass
        # with torch.no_grad():
        #     return LLAMA(model_name, **kwargs)
    elif model_name in Openai.models:
        return Openai(model_name, **kwargs)
    elif model_name == "moss":
        pass
        # return Moss(model_name, **kwargs)
