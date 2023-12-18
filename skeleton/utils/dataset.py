from typing import Optional, List, Dict, Callable
from dataclasses import dataclass, field
from datasets.dataset_dict import DatasetDict
from datasets.arrow_dataset import Dataset
from transformers.training_args import TrainingArguments
import re


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_source_length: Optional[int] = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_target_length: Optional[int] = field(
        default=512,
        metadata={
            "help": "The maximum total sequence length for target text after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    val_max_target_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "The maximum total sequence length for validation target text after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded. Will default to `max_target_length`."
                    "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
                    "during ``evaluate`` and ``predict``."
        },
    )
    val_max_time: Optional[int] = field(
        default=None,
        metadata={
            "help": "The maximum allowed time in seconds for generation of one example. This setting can be used to stop "
                    "generation whenever the full generation exceeds the specified amount of time."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
                    "value if set."
        },
    )
    max_val_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of validation or test examples to this "
                    "value if set."
        },
    )
    num_beams: int = field(
        default=1,
        metadata={
            "help": "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
                    "which is used during ``evaluate`` and ``predict``."
        },
    )
    num_beam_groups: int = field(
        default=1,
        metadata={
            "help": "Number of beam groups to use for evaluation. This argument will be passed to ``model.generate``, "
                    "which is used during ``evaluate`` and ``predict``."
        },
    )
    diversity_penalty: Optional[float] = field(
        default=None,
        metadata={
            "help": "Diversity penalty to use for evaluation. This argument will be passed to ``model.generate``, "
                    "which is used during ``evaluate`` and ``predict``."
        },
    )
    num_return_sequences: Optional[int] = field(
        default=None,
        metadata={
            "help": "The number of sequences to generate during evaluation. This argument will be passed to "
                    "``model.generate``, which is used during ``evaluate`` and ``predict``."
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether or not to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    source_prefix: Optional[str] = field(
        default=None,
        metadata={"help": "A prefix to add before every source text (useful for T5 models)."},
    )
    is_pruned_schema: bool = field(
        default=False,
        metadata={"help": "Whether or not to use the pruned schema."},
    )
    training_method: str = field(
        default="PT",
        metadata={"help": "Choose between ``PT`` and ``FT``"},
    )
    prompt_path: str = field(
        default="",
        metadata={"help": "The path to the soft prompts."},
    )
    schema_serialization_randomized: bool = field(
        default=False,
        metadata={"help": "Whether or not to randomize the order of tables."},
    )
    schema_serialization_with_db_id: bool = field(
        default=False,
        metadata={"help": "Whether or not to add the database id to the context. Needed for Picard."},
    )
    schema_serialization_with_db_content: bool = field(
        default=True,
        metadata={"help": "Whether or not to use the database content to resolve field matches."},
    )
    normalize_query: bool = field(default=True, metadata={"help": "Whether to normalize the SQL queries."})
    target_with_db_id: bool = field(
        default=False,
        metadata={"help": "Whether or not to add the database id to the target. Needed for Picard."},
    )

    def __post_init__(self):
        if self.val_max_target_length is None:
            self.val_max_target_length = self.max_target_length


@dataclass
class DataArguments:
    dataset: str = field(
        metadata={
            "help": "The dataset to be used. Choose between ``spider``, ``cosql``, or ``cosql+spider``, or ``spider_realistic``, or ``spider_syn``, or ``spider_dk``."},
    )
    dataset_paths: Dict[str, str] = field(
        default_factory=lambda: {
            "spider": "./skeleton/datasets/spider",
            # "kaggledbqa": "./datasets/kaggledbqa",
            # "drspider": "./datasets/drspider",
        },
        metadata={"help": "Paths of the dataset modules."},
    )
    metric_config: str = field(
        default="both",
        metadata={"help": "Choose between ``exact_match``, ``test_suite``, or ``both``."},
    )
    # we are referencing spider_realistic to spider metrics only as both use the main spider dataset as base.
    metric_paths: Dict[str, str] = field(
        default_factory=lambda: {
            "spider": "./skeleton/metrics/spider",
        },
        metadata={"help": "Paths of the metric modules."},
    )
    test_suite_db_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the test-suite databases."})
    data_config_file: Optional[str] = field(
        default=None,
        metadata={"help": "Path to data configuration file (specifying the database splits)"}
    )
    test_sections: Optional[List[str]] = field(
        default_factory=lambda: ["validation"],
        metadata={"help": "Sections from the data config to use for testing"}
    )
    toy: bool = field(
        default=False,
        metadata={"help": "Weather enable tpy setting for small dataset"}
    )


@dataclass
class InferDataArguments:
    input_file: str = field(
        metadata={
            "help": "pruned data for skeleton inference"},
    )
    output_file: str = field(
        metadata={
            "help": "skeleton inference result"},
    )
    batch_size: int = field(
        default=1,
        metadata={
            "help": "pipeline inference batch size"
        }
    )


@dataclass
class TrainSplit(object):
    dataset: Dataset
    # schemas: Dict[str, dict]


@dataclass
class EvalSplit(object):
    dataset: Dataset
    examples: Dataset
    # schemas: Dict[str, dict]


@dataclass
class DatasetSplits(object):
    train_split: Optional[TrainSplit]
    eval_split: Optional[EvalSplit]
    test_splits: Optional[Dict[str, EvalSplit]]
    # schemas: Dict[str, dict]


def _get_schemas(examples: Dataset) -> Dict[str, dict]:
    schemas: Dict[str, dict] = dict()
    for ex in examples:
        if ex["db_id"] not in schemas:
            schemas[ex["db_id"]] = {
                "db_table_names": ex["db_table_names"],
                "db_column_names": ex["db_column_names"],
                "db_column_types": ex["db_column_types"],
                "db_primary_keys": ex["db_primary_keys"],
                "db_foreign_keys": ex["db_foreign_keys"],
            }
    return schemas


def _prepare_train_split(
        dataset: Dataset,
        data_training_args: DataTrainingArguments,
        add_serialized_schema: Callable[[dict], dict],
        pre_process_function: Callable[[dict, Optional[int], Optional[int]], dict],
) -> TrainSplit:
    dataset = dataset.map(
        lambda ex: add_serialized_schema(ex=ex),
        batched=False,
        num_proc=data_training_args.preprocessing_num_workers,
        load_from_cache_file=False,
    )
    if data_training_args.max_train_samples is not None:
        dataset = dataset.select(range(data_training_args.max_train_samples))
    column_names = dataset.column_names
    dataset = dataset.map(
        lambda batch: pre_process_function(
            batch=batch,
            max_source_length=data_training_args.max_source_length,
            max_target_length=data_training_args.max_target_length,
        ),
        batched=True,
        num_proc=data_training_args.preprocessing_num_workers,
        remove_columns=column_names,
        load_from_cache_file=False,
    )
    return TrainSplit(dataset=dataset)


def _prepare_eval_split(
        dataset: Dataset,
        data_training_args: DataTrainingArguments,
        add_serialized_schema: Callable[[dict], dict],
        pre_process_function: Callable[[dict, Optional[int], Optional[int]], dict],
) -> EvalSplit:
    eval_examples = dataset
    eval_dataset = eval_examples.map(
        lambda ex: add_serialized_schema(ex=ex),
        batched=False,
        num_proc=data_training_args.preprocessing_num_workers,
        load_from_cache_file=False,
    )
    column_names = eval_dataset.column_names
    eval_dataset = eval_dataset.map(
        lambda batch: pre_process_function(
            batch=batch,
            max_source_length=data_training_args.max_source_length,
            max_target_length=data_training_args.val_max_target_length,
        ),
        batched=True,
        num_proc=data_training_args.preprocessing_num_workers,
        remove_columns=column_names,
        load_from_cache_file=False,
    )
    return EvalSplit(dataset=eval_dataset, examples=eval_examples)


def prepare_splits(
        dataset_dict: DatasetDict,
        data_args: DataArguments,
        training_args: TrainingArguments,
        data_training_args: DataTrainingArguments,
        add_serialized_schema: Callable[[dict], dict],
        pre_process_function: Callable[[dict, Optional[int], Optional[int]], dict],
) -> DatasetSplits:
    train_split, eval_split, test_splits = None, None, None

    if training_args.do_train:
        train_split = _prepare_train_split(
            dataset_dict["train"],
            data_training_args=data_training_args,
            add_serialized_schema=add_serialized_schema,
            pre_process_function=pre_process_function,
        )

    if training_args.do_eval:
        eval_split = _prepare_eval_split(
            dataset_dict["validation"],
            data_training_args=data_training_args,
            add_serialized_schema=add_serialized_schema,
            pre_process_function=pre_process_function,
        )

    if training_args.do_predict:
        test_splits = {
            section: _prepare_eval_split(
                dataset_dict[section],
                data_training_args=data_training_args,
                add_serialized_schema=add_serialized_schema,
                pre_process_function=pre_process_function,
            )
            for section in data_args.test_sections
        }
    if data_args.toy:
        if train_split:
            train_split.dataset = train_split.dataset.select(range(100))
        if eval_split:
            eval_split.dataset = eval_split.dataset.select(range(100))

    return DatasetSplits(
        train_split=train_split,
        eval_split=eval_split,
        test_splits=test_splits,
    )


def normalize(query: str) -> str:
    def comma_fix(s):
        # Remove spaces in front of commas
        return s.replace(" , ", ", ")

    def white_space_fix(s):
        # Remove double and triple spaces
        return " ".join(s.split())

    def lower(s):
        # Convert everything except text between (single or double) quotation marks to lower case
        return re.sub(r"\b(?<!['\"])(\w+)(?!['\"])\b", lambda match: match.group(1).lower(), s)

    return comma_fix(white_space_fix(lower(query)))


def serialize_schema(
        # question: str,
        db_id: str,
        # sql: str,
        # norm_sql: str,
        # sql_skeleton: str,
        db_schema: Dict[str, List],
        pk: Dict[str, List],
        fk: Dict[str, List],
) -> str:
    db_id_str = " {db_id}"
    table_sep = ""
    table_str = " | {table} : {columns}"
    column_sep = " , "
    column_str_with_values = "{column} ( {values} )"
    column_str_without_values = "{column}"
    value_sep = " , "

    def get_column_str(column_name: str, db_contents: List[str]) -> str:
        column_name_str = column_name.lower()
        if db_contents:
            return column_str_with_values.format(column=column_name_str, values=value_sep.join(db_contents[:1]))
        else:
            return column_str_without_values.format(column=column_name_str)

    tables = []
    for idx, schema_table in enumerate(db_schema['table_name_original']):
        columns = []
        for col, val in zip(db_schema['column_names_original'][idx], db_schema['db_contents'][idx]):
            columns.append(get_column_str(col, val))
        table = table_str.format(
            table=schema_table,
            columns=column_sep.join(columns)
        )
        tables.append(table)

    serialized_schema = db_id_str.format(db_id=db_id) + table_sep.join(tables)
    return serialized_schema


def form_input(question, serialized_schema):
    return f"Translate the question into a SQL skeleton according to the database. " \
           f"Question: {question} | " \
           f"Schema : {serialized_schema} | " \
           f"Skeleton : "
