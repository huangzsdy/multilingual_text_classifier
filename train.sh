#export CUDA_VISIBLE_DEVICES="1,2"
export CUDA_VISIBLE_DEVICES="3"
#config_file=configs/config.json
#config_file=configs/config_test.json
config_file=configs/config_test.toml

source /data/multilingual_projects/codes/ultrafineweb_exps/mega_env.sh

#LANGUAGES=('ru' 'fr' 'de' 'zh' 'ja')
#LANGUAGES=('es' 'ar' 'en'  'de')
LANGUAGES=('zh' 'ru')
#
#MODEL_NAME="models/FacebookAI/xlm-mlm-17-1280"
#MODEL_NAME="models/FacebookAI/xlm-roberta-large"
#MODEL_NAME="models/FacebookAI/xlm-roberta-base"
MODEL_NAME="models/FacebookAI/xlm-mlm-100-1280"


export MODEL_NAME
for lang in ${LANGUAGES[@]}
do
  export LANG=${lang}
  export OUTPUT_DIR=./output/${MODEL_NAME##*/}/results_${LANG}
#  export MODEL_NAME
  nohup \
    python -m scripts.train \
        --config $config_file \
        --procTitle ${lang} \
        >> logs/train_${lang}.log&
done
