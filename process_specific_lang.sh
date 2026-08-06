#!/bin/sh


source ../ultrafineweb_exps/mega_env.sh

export CUDA_VISIBLE_DEVICES="3"
#model=./models/FacebookAI/xlm-roberta-base
#model=/home/jeeves/xlm_roberta_base_classifier/results/checkpoint-9280
#model=/home/jeeves/xlm_roberta_base_classifier/results/checkpoint-13920
#model=results/checkpoint-202254/
#test=/data/multilingual_projects/datasets/文本数据/zh.jsonl
#output_file=/data/multilingual_projects/trainning_data/zh/all/zh_xlm_res.jsonl



#python3 inference.py --jsonl_files $test --max_samples 200 --output_file $output_file --model_path $model

#LANGUAGES=('ar' 'es' 'en'  'de')
#LANGUAGES=('en')
LANGUAGES=('en')

for LANG in ${LANGUAGES[@]}
do
  #model=output/results_${LANG}
  model=output/results_en/checkpoint-1362
#  input_file=/data/multilingual_projects/datasets/文本数据/${LANG}.jsonl
#  input_file="/data/multilingual_projects/datasets/文本数据/en_part_aa.jsonl"
  input_file="/data/multilingual_projects/datasets/文本数据/en_part_ab.jsonl"
#  input_file="/data/multilingual_projects/datasets/文本数据/en_part_aa_ab.jsonl"
  output_file=/data/multilingual_projects/trainning_data/${LANG}/all/${LANG}_xlm_allpred_ab.jsonl

  if [ "$lang" = 'en']; then
      batch_size=32
  else
      batch_size=32
  fi
#
  nohup \
    python3 -u scripts/inference.py --jsonl_files $input_file --batch_size $batch_size  --output_file $output_file --model_path $model \
     >>logs/process_all_${LANG}_ab.log&

done
#         >>logs/process_all_${LANG}_aa_ab.log&
