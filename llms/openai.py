# -*- coding: utf-8 -*-
# @Time    : 2023/4/11 21:48
# @Author  : Ray
# @Email   : httdty2@163.com
# @File    : openai.py
# @Software: PyCharm
import time
from concurrent.futures import ThreadPoolExecutor
from itertools import repeat

from llms.llm import LLM
from llms.config import OPENAI_KEY
import openai

openai.api_key = OPENAI_KEY


class Openai(LLM):
    models = {'gpt-4', 'gpt-3.5-turbo', 'text-davinci-003',
              'gpt-3.5-turbo-0613', 'gpt-3.5-turbo-1106', 'gpt-3.5-turbo-0301',
              'gpt-4-0613', 'gpt-4-0314'}

    def __init__(self, name, interval=0, **kwargs):
        super().__init__(name)
        if 'api_key' in kwargs:
            if kwargs['api_key']:
                openai.api_key = kwargs['api_key']
        if self.name in {'gpt-4', 'gpt-3.5-turbo', 'gpt-3.5-turbo-0613', 'gpt-3.5-turbo-1106',
                         'gpt-3.5-turbo-0301', 'gpt-4-0613', 'gpt-4-0314'}:
            self.api = openai.ChatCompletion.create
        elif self.name in {'text-davinci-003'}:
            self.api = openai.Completion.create
        else:
            raise LookupError("Please use valid model name for Openai model")
        self.interval = interval

    def __infer_one(self, prompt, kwargs):
        response = None
        while not response:
            try:
                if self.name in {'gpt-4', 'gpt-3.5-turbo',
                                 'gpt-3.5-turbo-0613', 'gpt-3.5-turbo-1106', 'gpt-3.5-turbo-0301',
                                 'gpt-4-0613', 'gpt-4-0314'}:
                    response = self.api(
                        model=self.name,
                        messages=[
                            {"role": "user", "content": prompt}
                        ],
                        # temperature=0,
                        max_tokens=128,
                        # top_p=1,
                        frequency_penalty=0,
                        presence_penalty=0,
                        request_timeout=30,
                        **kwargs
                    )
                elif self.name in {'text-davinci-003'}:
                    response = self.api(
                        model=self.name,
                        prompt=prompt,
                        # temperature=0,
                        max_tokens=128,
                        # top_p=1,
                        frequency_penalty=0,
                        presence_penalty=0,
                        request_timeout=30,
                        **kwargs
                    )
            except Exception as e:
                print(e)
                time.sleep(12)
                response = None

        return response

    def infer(self, prompt_list, **kwargs):
        assert isinstance(prompt_list, list), "Please make sure the input is a list of str"
        res = []
        with ThreadPoolExecutor(max_workers=min(len(prompt_list), 32)) as pool:
            response_list = pool.map(self.__infer_one, prompt_list, repeat(kwargs))
            for response in response_list:
                self.count += 1
                if self.name in {'gpt-4', 'gpt-3.5-turbo',
                                 'gpt-3.5-turbo-0613', 'gpt-3.5-turbo-1106', 'gpt-3.5-turbo-0301',
                                 'gpt-4-0613', 'gpt-4-0314'}:
                    res.append([choice.message.content for choice in response.choices])
                    self.prompt_length += response.usage.prompt_tokens
                    self.completion_length += response.usage.completion_tokens
                elif self.name in {'text-davinci-003'}:
                    # res.append([choice.message.content for choice in response.choices])
                    res.append([choice.text.strip() for choice in response.choices])
                    self.prompt_length += response.usage.prompt_tokens
                    self.completion_length += response.usage.completion_tokens
                else:
                    res.append(response.choices)
            time.sleep(self.interval)
        return res
