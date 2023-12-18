# coding=utf-8
# Copyright 2021 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Spider: A Large-Scale Human-Labeled Dataset for Text-to-SQL Tasks"""

import json
import os
from typing import List, Generator, Any, Dict, Tuple
import datasets

logger = datasets.logging.get_logger(__name__)

_CITATION = """\
@article{yu2018spider,
  title={Spider: A large-scale human-labeled dataset for complex and cross-domain semantic parsing and text-to-sql task},
  author={Yu, Tao and Zhang, Rui and Yang, Kai and Yasunaga, Michihiro and Wang, Dongxu and Li, Zifan and Ma, James and Li, Irene and Yao, Qingning and Roman, Shanelle and others},
  journal={arXiv preprint arXiv:1809.08887},
  year={2018}
}
"""

_DESCRIPTION = """\
Spider is a large-scale complex and cross-domain semantic parsing and text-toSQL dataset annotated by 11 college students
"""

_HOMEPAGE = "https://yale-lily.github.io/spider"

_LICENSE = "CC BY-SA 4.0"


class Spider(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.0.0")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="spider",
            version=VERSION,
            description="Spider: A Large-Scale Human-Labeled Dataset for Text-to-SQL Tasks",
        ),
    ]

    def __init__(self, *args, writer_batch_size=None, **kwargs) -> None:
        super().__init__(*args, writer_batch_size=writer_batch_size, **kwargs)
        self.schema_cache = dict()
        self.include_train_others: bool = kwargs.pop("include_train_others", True)

    def _info(self) -> datasets.DatasetInfo:
        features = datasets.Features(
            {
                "question": datasets.Value("string"),
                "db_id": datasets.Value("string"),
                "sql": datasets.Value("string"),
                "norm_sql": datasets.Value("string"),
                "sql_skeleton": datasets.Value("string"),
                "db_schema": datasets.features.Sequence(
                    {
                        "table_name_original": datasets.Value("string"),
                        "table_name": datasets.Value("string"),
                        "column_names_original": datasets.features.Sequence(datasets.Value("string")),
                        "column_names": datasets.features.Sequence(datasets.Value("string")),
                        "column_types": datasets.features.Sequence(datasets.Value("string")),
                        "db_contents": datasets.features.Sequence(
                            datasets.features.Sequence(
                                datasets.Value("string")
                            )
                        ),
                    }
                ),
                "fk": datasets.features.Sequence(
                    {
                        "source_table_name_original": datasets.features.Value("string"),
                        "source_column_name_original": datasets.features.Value("string"),
                        "target_table_name_original": datasets.features.Value("string"),
                        "target_column_name_original": datasets.features.Value("string"),
                    }
                ),
                "pk": datasets.features.Sequence(
                    {
                        "table_name_original": datasets.features.Value("string"),
                        "column_name_original": datasets.features.Value("string"),
                    }
                ),
                # "db_schema_pruned": datasets.features.Sequence(
                #     {
                #         "table_name_original": datasets.Value("string"),
                #         "table_name": datasets.Value("string"),
                #         "column_names_original": datasets.features.Sequence(datasets.Value("string")),
                #         "column_names": datasets.features.Sequence(datasets.Value("string")),
                #         "column_types": datasets.features.Sequence(datasets.Value("string")),
                #         "db_contents": datasets.features.Sequence(
                #             datasets.features.Sequence(
                #                 datasets.Value("string")
                #             )
                #         ),
                #     }
                # )
            }
        )
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            supervised_keys=None,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager) -> List[datasets.SplitGenerator]:

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "data_filepaths": [
                        "./datasets/spider/train_spider_pruned.json",
                        "./datasets/spider/train_others_pruned.json",
                    ]
                    if self.include_train_others
                    else ["./datasets/spider/train_spider_pruned.json"],
                    "db_path": "./datasets/spider/database",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "data_filepaths": [
                        os.path.join("./datasets/spider/dev_pruned.json"),
                    ],
                    "db_path": "./datasets/spider/database",
                },
            ),
        ]

    def _generate_examples(
            self, data_filepaths: List[str], db_path: str
    ) -> Generator[Tuple[int, Dict[str, Any]], None, None]:
        """This function returns the examples in the raw (text) form."""
        for data_filepath in data_filepaths:
            logger.info("generating examples from = %s", data_filepath)
            with open(data_filepath, encoding="utf-8") as f:
                spider = json.load(f)
                print(f)
                for idx, sample in enumerate(spider):
                    # sample["query"] = sample["query"][:-1] if sample["query"][-1] == ';' else sample["query"]
                    yield idx, {
                        "question": sample["question"],
                        "db_id": sample["db_id"],
                        "sql": sample["sql"],
                        "norm_sql": sample["norm_sql"],
                        "sql_skeleton": sample["sql_skeleton"],
                        "db_schema": sample["db_schema"],
                        "pk": sample["pk"],
                        "fk": sample['fk']
                    }
