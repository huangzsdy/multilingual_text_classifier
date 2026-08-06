#set -x

:<<comment
用于同时跑微调后模型预测
comment

export CUDA_VISIBLE_DEVICES="0"
model_name="xlm-roberta-base"
#model=/home/jeeves/xlm_roberta_base_classifier/results/checkpoint-9280
#model=/home/jeeves/xlm_roberta_base_classifier/results/checkpoint-13920
#model=results  #66w
#test=/data/multilingual_projects/trainning_data/zh/filtered_result/test.jsonl
#output_file=/data/multilingual_projects/trainning_data/zh/filtered_result/tf_test.jsonl

#LANGUAGES=('es' 'en'  'de' 'ar' 'zh')
#LANGUAGES=('ar')
#LANGUAGES=('es' 'en'  'de' )
LANGUAGES=('ru' 'fr' 'de' 'ja')
LANGUAGES=('fr' 'de' 'ja')
#LANGUAGES=('ru' 'fr' 'de' 'zh' 'ja')
batch_size=200
for LANG in ${LANGUAGES[@]}
do
  model=output/${model_name}/results_${LANG}
  checkpoint_dir=output/${model_name}/results_${LANG}/checkpoint/
  input_file="/data/multilingual_projects/trainning_data/${LANG}/filtered_result/test.jsonl"
  output_file="/data/multilingual_projects/trainning_data/${LANG}/xlm_output/${model_name}/test_pred.jsonl"
#  python3 scripts/inference_stream_v5.py --jsonl_files $input_file --max_samples 200 --output_file $output_file --model_path $model

  python3 -u scripts/inference_stream_v5.py --checkpoint_dir $checkpoint_dir \
    --gpu_ids $CUDA_VISIBLE_DEVICES --use_amp --batch_size $batch_size \
    --text_field "content" --jsonl_files $input_file --output_file $output_file\
    --model_path $model \

done