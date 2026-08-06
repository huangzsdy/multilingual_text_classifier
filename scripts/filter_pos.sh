set -x

#LANGUAGES=('es' 'en'  'de' 'ar' 'zh')
LANGUAGES=('ja')
#LANGUAGES=('en' 'zh')
#LANGUAGES=('fr' 'ja' 'de' 'ru' 'zh')



threshold=0.9
model_name="xlm-roberta-large"

for LANG in ${LANGUAGES[@]}
do
  if [ $LANG = 'zh']; then
    threshold=0.75
  else
    threshold=0.75
  fi
#  model=output/results_${LANG}
  model=output/$model_name/results_${LANG}

  input_file=/data/multilingual_projects/trainning_data/${LANG}/all_v5/$model_name/${LANG}_xlm_allpred.jsonl # note:es, he, ar, de 已跑完

#  input_file=/data/multilingual_projects/trainning_data/${LANG}/all_v5/${LANG}_xlm_allpred.jsonl
#  output_file=/data/multilingual_projects/trainning_data/${LANG}/all_v5/${LANG}_xlm_allpred_filterpos.jsonl
  output_file=/data/multilingual_projects/trainning_data/${LANG}/all_v5/$model_name/${LANG}_xlm_allpred_filterpos_${threshold}.jsonl

#  nohup \
    python3 -u scripts/filter_pos_jsonl.py $input_file $output_file $threshold \
#  >> logs/filter_pos_${model_name}_{LANG}.log&
done