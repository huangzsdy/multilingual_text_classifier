#!/bin/bash

pos_list=(
  '/data/multilingual_projects/datasets/qwen_labeled/zh/classify_result/zh_pos.jsonl'
#'/data/multilingual_projects/datasets/multi_langs/fineweb2_hq_part/zh/000_00075.jsonl'
#'/data/multilingual_projects/datasets/multi_langs/fineweb2_hq_part/zh/000_00348.jsonl'
#'/data/multilingual_projects/datasets/multi_langs/fineweb2_hq_part/zh/000_00370.jsonl'
#'/data/multilingual_projects/datasets/multi_langs/fineweb2_hq_part/zh/000_00371.jsonl'
#'/data/multilingual_projects/datasets/multi_langs/fineweb2_hq_part/zh/000_00372.jsonl'
#'/data/multilingual_projects/datasets/multi_langs/fineweb2_hq_part/zh/000_00373.jsonl'
#'/data/multilingual_projects/datasets/multi_langs/fineweb2_hq_part/zh/000_00374.jsonl'
#'/data/multilingual_projects/datasets/multi_langs/fineweb2_hq_part/zh/cmn_Hani/_jsonl/000_00000.jsonl'
#'/data/multilingual_projects/datasets/multi_langs/fineweb2_hq_part/zh/cmn_Hani/_jsonl/000_00001.jsonl'
#'/data/multilingual_projects/datasets/multi_langs/fineweb2_hq_part/zh/cmn_Hani/_jsonl/000_00002.jsonl'
)
neg_list=(
  '/data/multilingual_projects/datasets/qwen_labeled/zh/classify_result/zh_neg.jsonl'
)


LANGUAGES=('es' 'en'  'de' 'ar')



for LANG in ${LANGUAGES[@]}
do
pos_file="/data/multilingual_projects/datasets/qwen_labeled/${LANG}/classify_result/${LANG}_pos.jsonl"
neg_file="/data/multilingual_projects/datasets/qwen_labeled/${LANG}/classify_result/${LANG}_neg.jsonl"

target_dir=/data/multilingual_projects/trainning_data/${LANG}
#if [ ! -f "$MODEL_PATH" ]; then
#    echo "错误: 模型文件 $MODEL_PATH 不存在"
#    exit 1
#fi

#target_dir=/data/multilingual_projects/trainning_data
  python -u filter_testset.py \
        --pos "$pos_file" \
        --neg "$neg_file" \
        --test-size  100 \
        --key-fields content \
        --output-dir $target_dir \
        --seed 123
done

#            --key-fields content text \
#      --pos "${pos_list[@]}" \
#      --neg "${neg_list[@]}" \