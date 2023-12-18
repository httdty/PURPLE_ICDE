# -*- coding: utf-8 -*-
# @Time    : 2023/7/21 15:22
# @Author  : Ray
# @Email   : httdty2@163.com
# @File    : infer.py
# @Software: PyCharm
# Set up logging
import random
import sys
import os
import time

import datasets
from torch.utils.data import dataset
from tqdm import tqdm

from skeleton.utils.pipeline import SkeletonGenerationPipeline, SkeletonInput

sys.path.append(os.path.abspath(os.getcwd()))
if sys.path[0] == os.path.join(os.path.abspath(os.getcwd()), "skeleton"):
    sys.path.pop(0)

import json
import torch.distributed as dist
import torch
from transformers.hf_argparser import HfArgumentParser
from transformers.models.auto import AutoConfig, AutoTokenizer, AutoModelForSeq2SeqLM
from transformers.models.t5.modeling_t5 import T5ForConditionalGeneration
from transformers.models.t5.tokenization_t5_fast import T5TokenizerFast
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from tokenizers import AddedToken

from skeleton.utils.args import ModelArguments
from skeleton.utils.dataset import DataTrainingArguments, InferDataArguments


def main():
    # See all possible arguments by passing the --help flag to this script.
    parser = HfArgumentParser(
        (ModelArguments, InferDataArguments, DataTrainingArguments)
    )
    model_args: ModelArguments
    data_args: InferDataArguments
    data_training_args: DataTrainingArguments
    model_args, data_args, data_training_args = parser.parse_args_into_dataclasses()

    print("Initialize config")
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        max_length=data_training_args.max_target_length,
        num_beams=data_training_args.num_beams,
        num_beam_groups=data_training_args.num_beam_groups,
        diversity_penalty=data_training_args.diversity_penalty,
        num_return_sequences=data_training_args.num_return_sequences,
    )

    print("Initialize tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    assert isinstance(tokenizer, PreTrainedTokenizerFast), "Only fast tokenizers are currently supported"
    if isinstance(tokenizer, T5TokenizerFast):
        # In T5 `<` is OOV, see https://github.com/google-research/language/blob/master/language/nqg/tasks/spider/restore_oov.py
        tokenizer.add_tokens([AddedToken(" <="), AddedToken(" <")])

    print("Initialize model")
    model = None
    if torch.cuda.is_available() and not dist.is_initialized():
        print("Manually initialized")
        dist.init_process_group(
            backend='nccl',
            init_method=f'tcp://localhost:{random.randint(10000, 30000)}',
            rank=0,
            world_size=1
        )
    elif not torch.cuda.is_available() and not dist.is_initialized():
        print("Manually initialized")
        dist.init_process_group(
            backend='gloo',
            init_method=f'tcp://localhost:{random.randint(10000, 30000)}',
            rank=0,
            world_size=1
        )

    while not model:
        try:
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
            )
        except ConnectionError as e:
            print(e)
            model = None
    if isinstance(model, T5ForConditionalGeneration):
        model.resize_token_embeddings(len(tokenizer))

    # Pipeline ready
    if torch.cuda.is_available():
        pipe = SkeletonGenerationPipeline(
            model=model,
            tokenizer=tokenizer,
            device=0
        )
    else:
        pipe = SkeletonGenerationPipeline(
            model=model,
            tokenizer=tokenizer,
            device=-1
        )
    gen_kwargs = {
        "max_length": data_training_args.val_max_target_length,
        "num_beams": data_training_args.num_beams,
        "synced_gpus": True,
        "output_scores": True,
        "return_dict_in_generate": True,
        "num_return_sequences": data_training_args.num_return_sequences,
    }

    # Load data
    ins_list = []
    with open(data_args.input_file, 'r') as f:
        raw_data = json.load(f)
    for ins in raw_data:
        db_schema = {
            "table_name_original": [table['table_name_original'] for table in ins['db_schema']],
            "table_name": [table['table_name'] for table in ins['db_schema']],
            "column_names_original": [table['column_names_original'] for table in ins['db_schema']],
            "column_names": [table['column_names'] for table in ins['db_schema']],
            "db_contents": [table['db_contents'] for table in ins['db_schema']],

        }
        pk = {
            "table_name_original": [p['table_name_original'] for p in ins['pk']],
            "column_name_original": [p['column_name_original'] for p in ins['pk']],
        }
        fk = {
            "source_table_name_original": [f['source_table_name_original'] for f in ins['fk']],
            "source_column_name_original": [f['source_column_name_original'] for f in ins['fk']],
            "target_table_name_original": [f['target_table_name_original'] for f in ins['fk']],
            "target_column_name_original": [f['target_column_name_original'] for f in ins['fk']],
        }
        ins_list.append(
            SkeletonInput(
                question=ins['question'],
                db_id=ins['db_id'],
                db_schema=db_schema,
                pk=pk,
                fk=fk
            )
        )

    # Inference
    res_list = []
    start = time.time()
    if data_args.batch_size == 1:
        for ins in tqdm(ins_list):
            res = pipe(ins, **gen_kwargs)
            res_list.append(res)
    else:
        res_list = pipe(ins_list, batch_size=data_args.batch_size, **gen_kwargs)
    print("Inference time cost:", time.time() - start)

    # Output file
    with open(data_args.output_file, 'w') as f:
        json.dump(res_list, f, indent=2)



if __name__ == "__main__":
    main()


