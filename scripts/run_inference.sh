set -x
export CUDA_VISIBLE_DEVICES="5"
model=./models/FacebookAI/xlm-roberta-base
model=/home/jeeves/xlm_roberta_base_classifier/results/checkpoint-9280
#model=/home/jeeves/xlm_roberta_base_classifier/results/checkpoint-13920
model=./results
test=/data/multilingual_projects/trainning_data/zh/filtered_result/test.jsonl
output_file=/data/multilingual_projects/trainning_data/zh/xlm_output/test_pred.jsonl

python3 inference.py --jsonl_files $test --max_samples 200 --output_file $output_file --model_path $model
