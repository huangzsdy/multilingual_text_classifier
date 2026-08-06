set -x
#!/bin/sh

source ./xlm_env.sh

export CUDA_VISIBLE_DEVICES="0"
#model=./models/FacebookAI/xlm-roberta-base
#model=/home/jeeves/xlm_roberta_base_classifier/results/checkpoint-9280
#model=/home/jeeves/xlm_roberta_base_classifier/results/checkpoint-13920
#model=results/checkpoint-202254/
#test=/data/multilingual_projects/datasets/文本数据/zh.jsonl
#output_file=/data/multilingual_projects/trainning_data/zh/all/zh_xlm_res.jsonl



#LANGUAGES=('ar' 'es' 'ja' 'de' 'fr')
#LANGUAGES=('ar' 'es' 'ja')
#LANGUAGES=('hi' 'it')
#LANGUAGES=('zh')
#LANGUAGES=('en')
#LANGUAGES=('zh' 'en')
LANGUAGES=('zh')
#LANGUAGES=('fr' 'ja' 'de' 'ru')
#LANGUAGES=('fr' 'he' 'hi' 'it' 'ja')
#LANGUAGES=('ko' 'nl')
#LANGUAGES=('de' 'zh')
#LANGUAGES=('pt' 'ru' )
#LANGUAGES=('en' 'zh' 'it' 'fr')


model_name="xlm-roberta-large"

for LANG in ${LANGUAGES[@]}
do
  model=output/$model_name/results_${LANG}
  export LANG
  input_file=/data/multilingual_projects/datasets/文本数据/${LANG}.jsonl
#  output_file=/data/multilingual_projects/trainning_data/${LANG}/all/${LANG}_xlm_allpred_keepon.jsonl # note:es, he, ar, de 已跑完
#  process_file=/data/multilingual_projects/trainning_data/${LANG}/all/${LANG}_xlm_allpred.jsonl # note:es, he, ar, de 已跑完
  output_file=/data/multilingual_projects/trainning_data/${LANG}/all_v5/$model_name/${LANG}_xlm_allpred.jsonl # note:es, he, ar, de 已跑完
  checkpoint_dir=/data/multilingual_projects/trainning_data/${LANG}/all_v5/$model_name/checkpoint

  # zh
#  output_file=/data/multilingual_projects/trainning_data/${LANG}/all_v5/${LANG}_xlm_allpred.jsonl # note:es, he, ar, de 已跑完
##  output_file=/data/multilingual_projects/trainning_data/${LANG}/all_v3/${LANG}_xlm_allpred.jsonl # note:es, he, ar, de 已跑完
##  output_file=/data/multilingual_projects/trainning_data/${LANG}/all_v2/${LANG}_xlm_allpred.jsonl # note:es, he, ar, de 已跑完
#  process_file=/data/multilingual_projects/trainning_data/${LANG}/all_v2/${LANG}_xlm_allpred.jsonl # note:es, he, ar, de 已跑完
#  checkpoint_dir=/data/multilingual_projects/trainning_data/${LANG}/all_v2/

  if [ $LANG = 'all']; then
      batch_size=5000
  else
      batch_size=1000
  fi
#  nohup \
#    python3 -u scripts/inference_stream_v3.py --read_batch_size 102400 --text_field "content"  --jsonl_files $input_file --batch_size $batch_size  --output_file $output_file --model_path $model \
#         >>logs/process_all_${LANG}_v2.log&
#  nohup \
#    python3 -u scripts/inference_stream_v3.py --read_batch_size 5000 --text_field "content" --processed_file_lasttime $process_file --jsonl_files $input_file --batch_size $batch_size  --output_file $output_file --model_path $model \
#         >>logs/process_all_${LANG}.log&

#  nohup \
#    python3 -u scripts/inference_stream_v4.py --batch_size $batch_size --text_field "content" --processed_file_lasttime $process_file --jsonl_files $input_file --batch_size $batch_size  --output_file $output_file --model_path $model \
#         >>logs/process_all_${LANG}_v3.log&
#  nohup \
#    python3 -u scripts/inference_stream_v4_nocompiling.py --use_amp --device cuda --batch_size $batch_size --text_field "content" --processed_file_lasttime $process_file --jsonl_files $input_file --batch_size $batch_size  --output_file $output_file --model_path $model \
#         >logs/process_all_${LANG}_v4_nocompiling.log&

  nohup \
    python3 -u scripts/inference_stream_v5.py --checkpoint_dir $checkpoint_dir --gpu_ids $CUDA_VISIBLE_DEVICES --use_amp --batch_size $batch_size --text_field "content" --jsonl_files $input_file --batch_size $batch_size  --output_file $output_file --model_path $model \
         >>logs/process_all_${LANG}_v5.log&
done

