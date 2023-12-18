set -e

source activate purple

# Pre-process dev
# PASSED on 167
if [ -f "./datasets/spider/dev_preprocessed.json" ];then
  echo "dev spider preprocess: have done"
  else
    echo "======================== dev spider preprocess: start ========================"
    python -m schema_prune.preprocessing --mode=test \
      --table_path=./datasets/spider/tables.json \
      --input_file=./datasets/spider/dev.json \
      --db_path=./datasets/spider/database \
      --output_file=./datasets/spider/dev_preprocessed.json
    echo "======================== dev spider preprocess: finished ========================"
fi


# inference on dev set
# PASSED on 167
if [ -f "./datasets/spider/dev_with_probs.json" ];then
  echo "======================== schema pruning inference for dev: have done ========================"
  else
    echo "======================== schema pruning inference for dev: start ========================"
    python -m schema_prune.classifier \
    --batch_size 12 \
    --seed 42 \
    --save_path "./saved_models/resd_classifier" \
    --dev_filepath "./datasets/spider/dev_preprocessed.json" \
    --output_filepath "./datasets/spider/dev_with_probs.json" \
    --mode "test"
    echo "======================== schema pruning inference for dev: finished ========================"
fi

# postprocess on dev set
# PASSED on 167
if [ -f "./datasets/spider/dev_pruned.json" ];
  then
    echo "======================== schema pruning postprocessing for dev: have done ========================"
  else
    echo "======================== schema pruning postprocessing for dev: start ========================"
    python -m schema_prune.postprocessing \
      --input_file=./datasets/spider/dev_with_probs.json \
      --db_dir=./datasets/spider/database \
      --output_file=./datasets/spider/dev_pruned.json
    echo "======================== schema pruning postprocessing for dev: finished ========================"
fi


# Skeleton inference
if [ -f "./datasets/spider/dev_skeleton.json" ];
  then
    echo "======================== dev Skeleton inference: have done ========================"
  else
    echo "======================== dev Skeleton inference: start ========================"
    python -m skeleton.infer \
      --model_name_or_path ./saved_models/train/tg_3b/BEST_MODEL \
      --source_prefix  "" \
      --normalize_query \
      --cache_dir transformers_cache \
      --num_beams 3 \
      --num_return_sequences 3 \
      --num_beam_groups 1 \
      --overwrite_cache \
      --input_file ./datasets/spider/dev_pruned.json \
      --output_file ./datasets/spider/dev_skeleton.json \
      --batch_size 1
    echo "======================== dev Skeleton inference: done ========================"
fi

# # LLM inference run !!!
echo "======================== LLMs inference: start ========================"
python -m models.run \
  --model_name=gpt-3.5-turbo-0613 \
  --shot=few \
  --prompt=purple \
  --enable_skeleton \
  --enable_distinct \
  --train_file=./datasets/spider/train_spider_pruned.json \
  --dev_file=./datasets/spider/dev_pruned.json \
  --ori_dev_file=./datasets/spider/dev.json \
  --pred_skeleton=./datasets/spider/dev_skeleton.json \
  --db_dir=./datasets/spider/database/ \
  --bug_fix \
  --batch_size=5 \
  --prompt_length=3072 \
  --consistency_num=30 \
  --exp_name=result \
  --stage=test


python -m models.run \
  --model_name=gpt-4 \
  --shot=few \
  --prompt=purple \
  --enable_skeleton \
  --enable_distinct \
  --train_file=./datasets/spider/train_spider_pruned.json \
  --dev_file=./datasets/spider/dev_pruned.json \
  --ori_dev_file=./datasets/spider/dev.json \
  --pred_skeleton=./datasets/spider/dev_skeleton.json \
  --db_dir=./datasets/spider/database/ \
  --bug_fix \
  --batch_size=5 \
  --prompt_length=3072 \
  --consistency_num=30 \
  --exp_name=result \
  --stage=test
  
echo "======================== LLMs inference: done ========================"

