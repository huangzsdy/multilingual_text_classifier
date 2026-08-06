#!/bin/bash

# 使用最小化配置的批量评测验证集脚本

echo "=== classifier 验证集评测 ==="
#source ./tc_env.sh
formatted_time=$(date "+%Y%m%d_%H%M%S")

#LANG=fr
#LANG=es
#LANG=it
#LANG=nl
#LANG=pl
#LANG=ar
#LANG=de
#LANG=ja
#LANG=pt
#LANG=ru
#LANG=ko
#LANG=he
#LANG=hi
#LANG=en
#LANG=zh

#LANGUAGES=('es' 'en'  'de' 'ar')
#LANGUAGES=('zh')
#LANGUAGES=('ar')
#LANGUAGES=('es' 'en'  'de')
#LANGUAGES=('ru' 'fr' 'de' 'zh' 'ja')
LANGUAGES=('ru' 'fr' 'de' 'ja')



for LANG in ${LANGUAGES[@]}
do
  GT_FILE="/data/multilingual_projects/trainning_data/${LANG}/filtered_result/test.jsonl"
  PRED_FILE="/data/multilingual_projects/trainning_data/${LANG}/xlm_output/test_pred.jsonl"
  SAVE_RESULTS_PATH="/data/multilingual_projects/trainning_data/${LANG}/xlm_output/calculate_metrics"

  # 创建输出目录
  mkdir -p $SAVE_RESULTS_PATH

  echo "pred文件: $PRED_FILE"
  echo "gt路径: $GT_FILE"
  echo "输出目录: $SAVE_RESULTS_PATH"
  echo ""

  # 检查文件是否存在
  if [ ! -f "$PRED_FILE" ]; then
      echo "错误: pred文件 PRED_FILE 不存在"
      exit 1
  fi

  if [ ! -f "$GT_FILE" ]; then
      echo "错误: gt文件 GT_FILE 不存在"
      exit 1
  fi

  echo "开始批量计算metrics..."
  echo "结果保存在: $SAVE_RESULTS_PATH"

  export PYTHONPATH=`pwd`/tools:$PYTHONPATH



  # 运行批量推理命令
  #nohup \
   python -u commands/calculate_fasttext_prediction.py \
      -gt $GT_FILE \
      -pred $PRED_FILE \
      -srp $SAVE_RESULTS_PATH \
  #    >> logs/fasttext_calculate_evaluation_${formatted_time}_${LANG}.log&

  echo ""
  #echo "批量推理完成！"



done