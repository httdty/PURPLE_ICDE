# -*- coding: utf-8 -*-
# @Time    : 2023/7/21 10:57
# @Author  : Ray
# @Email   : httdty2@163.com
# @File    : pipeline.py
# @Software: PyCharm
from dataclasses import dataclass
from typing import Union, List, Dict

import torch
from transformers.pipelines.text2text_generation import ReturnType, Text2TextGenerationPipeline
from transformers.tokenization_utils_base import BatchEncoding

from skeleton.utils.dataset import form_input, serialize_schema


@dataclass
class SkeletonInput(object):
    question: str
    db_id: str
    db_schema: Dict[str, List]
    pk: Dict[str, List]
    fk: Dict[str, List]


class SkeletonGenerationPipeline(Text2TextGenerationPipeline):
    """
    Pipeline for text-to-SQL generation using seq2seq models.

    The models that this pipeline can use are models that have been fine-tuned on the Spider text-to-SQL task.

    Usage::

        model = AutoModelForSeq2SeqLM.from_pretrained(...)
        tokenizer = AutoTokenizer.from_pretrained(...)
        db_path = ... path to "concert_singer" parent folder
        text2sql_generator = Text2SQLGenerationPipeline(
            model=model,
            tokenizer=tokenizer,
            db_path=db_path,
        )
        text2sql_generator(inputs=Text2SQLInput(utterance="How many singers do we have?", db_id="concert_singer"))
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, inputs: Union[SkeletonInput, List[SkeletonInput]], *args, **kwargs):
        r"""
        Generate the output SQL expression(s) using text(s) given as inputs.

        Args:
            inputs (:obj:`Text2SQLInput` or :obj:`List[Text2SQLInput]`):
                Input text(s) for the encoder.
            return_tensors (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to include the tensors of predictions (as token indices) in the outputs.
            return_text (:obj:`bool`, `optional`, defaults to :obj:`True`):
                Whether or not to include the decoded texts in the outputs.
            clean_up_tokenization_spaces (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to clean up the potential extra spaces in the text output.
            truncation (:obj:`TruncationStrategy`, `optional`, defaults to :obj:`TruncationStrategy.DO_NOT_TRUNCATE`):
                The truncation strategy for the tokenization within the pipeline.
                :obj:`TruncationStrategy.DO_NOT_TRUNCATE` (default) will never truncate, but it is sometimes desirable
                to truncate the input to fit the model's max_length instead of throwing an error down the line.
            generate_kwargs:
                Additional keyword arguments to pass along to the generate method of the model (see the generate method
                corresponding to your framework `here <./model.html#generative-models>`__).

        Return:
            A list or a list of list of :obj:`dict`: Each result comes as a dictionary with the following keys:

            - **generated_sql** (:obj:`str`, present when ``return_text=True``) -- The generated SQL.
            - **generated_token_ids** (:obj:`torch.Tensor` or :obj:`tf.Tensor`, present when ``return_tensors=True``)
              -- The token ids of the generated SQL.
        """
        result = super().__call__(inputs, *args, **kwargs)
        if (
            isinstance(inputs, list)
            and all(isinstance(el, SkeletonInput) for el in inputs)
            and all(len(res) == 1 for res in result)
        ):
            return [res[0] for res in result]
        return result

    def _parse_and_tokenize(
        self, *args, truncation
    ) -> BatchEncoding:
        inputs = args[0]
        if isinstance(inputs, list):
            if self.tokenizer.pad_token_id is None:
                raise ValueError("Please make sure that the tokenizer has a pad_token_id when using a batch input")
            inputs = [self._pre_process(t2s_input=i) for i in inputs]
            padding = True
        elif isinstance(inputs, SkeletonInput):
            inputs = self._pre_process(t2s_input=inputs)
            padding = False
        else:
            raise ValueError(
                f" `inputs`: {inputs} have the wrong format. The should be either of type `Text2SQLInput` or type `List[Text2SQLInput]`"
            )
        encodings = self.tokenizer(inputs, padding=padding, truncation=truncation, return_tensors=self.framework)
        # This is produced by tokenizers but is an invalid generate kwargs
        if "token_type_ids" in encodings:
            del encodings["token_type_ids"]
        return encodings

    @staticmethod
    def _pre_process(t2s_input: SkeletonInput) -> str:
        s = serialize_schema(t2s_input.db_id, t2s_input.db_schema, t2s_input.pk, t2s_input.fk)
        input_str = form_input(t2s_input.question, s)
        return input_str

    def _forward(self, model_inputs, **generate_kwargs):
        in_b, input_length = model_inputs["input_ids"].shape

        generate_kwargs["min_length"] = generate_kwargs.get("min_length", self.model.config.min_length)
        generate_kwargs["max_length"] = generate_kwargs.get("max_length", self.model.config.max_length)
        self.check_inputs(input_length, generate_kwargs["min_length"], generate_kwargs["max_length"])
        output = self.model.generate(**model_inputs, **generate_kwargs)
        output_ids = output.sequences
        sequences_scores = output.sequences_scores
        out_b = output_ids.shape[0]
        output_ids = output_ids.reshape(in_b, out_b // in_b, *output_ids.shape[1:])
        sequences_scores = sequences_scores.reshape(in_b, out_b // in_b)
        return {
            "output_ids": output_ids,
            "sequences_scores": torch.nn.functional.softmax(sequences_scores, dim=-1)
        }

    def postprocess(self, model_outputs, return_type=ReturnType.TEXT, clean_up_tokenization_spaces=False):
        records = []
        for idx, output_ids in enumerate(model_outputs["output_ids"][0]):
            if return_type == ReturnType.TENSORS:
                record = {f"{self.return_name}_token_ids": output_ids}
            elif return_type == ReturnType.TEXT:
                if 'sequences_scores' in model_outputs:
                    record = {
                        f"{self.return_name}_text": self.tokenizer.decode(
                            output_ids,
                            skip_special_tokens=True,
                            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
                        ),
                        "scores": model_outputs['sequences_scores'][0][idx].item()
                    }
                else:
                    record = {
                        f"{self.return_name}_text": self.tokenizer.decode(
                            output_ids,
                            skip_special_tokens=True,
                            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
                        ),
                    }
            records.append(record)
        print(records)
        return records
