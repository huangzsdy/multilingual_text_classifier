#!/usr/bin/env bash


#这个文件是专门用来划分训练集和验证集的pos/neg

SCRIPT_DIR=./
#cd $SCRIPT_DIR

#LANGUAGES=('fr' 'es')
#LANGUAGES=('fr')
#LANGUAGES=('ar'  'de'   'he'  'hi'  'it'  'ja'  'ko'   'nl'  'pl'  'pt'  'ru')
#LANGUAGES=('de' 'he'  'hi' 'ko' 'pt'  'ru')
#LANGUAGES=('de' 'he'  'hi' 'ko' 'pt'  'ru' 'zh' 'en')

#LANGUAGES=('zh')
#LANGUAGES=('de' 'he'  'hi' 'ko')
#LANGUAGES=('es' 'en'  'de' 'ar')
LANGUAGES=('ru' 'pl')
#LANGUAGES=('fr' 'zh')



for LANG in ${LANGUAGES[@]}
do
#  TO_CLASSIFICATION=output/minimal_batch_output/${LANG}/final_pred/${LANG}.jsonl
  TO_CLASSIFICATION=/data/multilingual_projects/datasets/qwen_labeled/${LANG}.jsonl
  CLASSIFICATION_DIR=/data/multilingual_projects/datasets/qwen_labeled/${LANG}/classify_result/

  LOG_FILE=logs/classify_$LANG.log

#  touch $LOG_FILE &&
  echo "start processing $LANG ----------------------"
  nohup \
    python -u $SCRIPT_DIR/jsonl_classify.py $TO_CLASSIFICATION -o $CLASSIFICATION_DIR \
      2>&1 >> ${LOG_FILE} &
#  echo "finished processing $LANG -- output: ${CLASSIFICATION_DIR} --------------------"
done

cd -