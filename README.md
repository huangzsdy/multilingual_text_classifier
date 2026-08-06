## 具体步骤

### 待处理语种
fr, ko, ja, ...共15种
ko, nl, pl, pt, ru
### 标注数据分批
1. scripts/classify.sh
将qwen标注的jsonl文件根据pos/neg 进行划分
2. scripts/filter.sh
为了避免重复，每次都需要重新随机划分出200条测试集， pos, neg_target 

### 标注数据Bert微调
train.sh
pos 和 neg_target 用于bert微调


### 验证微调是否带来增益
multilingual_testset.sh
calculate_metrics.sh
跑分F1

### 微调后模型预测全量数据
process_all_lang.sh
微调后选择对应语种进行预测
wip:
 en, zh

### 分流预测后样本，选择超过阈值的正样本用于minicpm-1b跑评测

bash scripts/filter_pos.sh 将预测结果中pos部分捞出来

wip:



done:





### minicpm-1b 跑评测

sstable:



微调 wip:
 'ko' 'nl' 'pl' 'pt' 'ru' hi es de fr it

微调 done:

convert model:

跑评测done:
ja, ar

跑不了评测：
he

### 可用小语种dataset

