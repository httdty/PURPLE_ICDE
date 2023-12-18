# -*- coding: utf-8 -*-
# @Time    : 2023/4/11 21:55
# @Author  : Ray
# @Email   : httdty2@163.com
# @File    : llm.py
# @Software: PyCharm
from typing import List


class LLM:
    def __init__(self, name):
        self.name = name
        self.count = 0
        self.prompt_length = 0
        self.completion_length = 0


    def infer(self, prompt: List[str]):
        pass
