# AI-LI:智能判决书分类与打标平台

### 网站
网址：http://law.seutools.com/
使用说明：



### 项目介绍

关键词：自然语言处理，分类任务，多标签，多分类，法律任务。

项目来源：CAIL2021年多案件标签分类比赛，增加了部分数据集&增加标签语义。

原始比赛链接：http://cail.cipsc.org.cn/task8.html

个人基础方案：单句预测标签最终结果求并集作为案件的标签。

个人改进方案：滑动窗多标签并集模型及负样本采样的训练策略。


### 运行

训练模型：
```
cd bert-bseline
```

base实验运行：
```
python train_wandb.py --debug --debug_train_num=200 --debug_valid_num=40 --train_batch=16 --valid_batch=64 --model_path=bert-base-chinese --learning_rate=5e-5 --train_rate=0.8 --content_size=100 --epoch_number=10 --freeze --pn_rate=1 --time_limit=20
```

pn_rate 10：
```
python train_wandb.py --debug --debug_train_num=200 --debug_valid_num=40 --train_batch=16 --valid_batch=64 --model_path=bert-base-chinese --learning_rate=5e-5 --train_rate=0.8 --content_size=100 --epoch_number=10 --freeze --pn_rate=10 --time_limit=20
```

pn_rate 0.1
```
python train_wandb.py --debug --debug_train_num=200 --debug_valid_num=40 --train_batch=16 --valid_batch=64 --model_path=bert-base-chinese --learning_rate=5e-5 --train_rate=0.8 --content_size=100 --epoch_number=10 --freeze --pn_rate=0.1 --time_limit=20
```

freeze = false
```
python train_wandb.py --debug --debug_train_num=200 --debug_valid_num=40 --train_batch=16 --valid_batch=64 --model_path=bert-base-chinese --learning_rate=5e-5 --train_rate=0.8 --content_size=100 --epoch_number=10 --pn_rate=1 --time_limit=20
```

content_size = 500
```
python train_wandb.py --debug --debug_train_num=200 --debug_valid_num=40 --train_batch=16 --valid_batch=64 --model_path=bert-base-chinese --learning_rate=5e-5 --train_rate=0.8 --content_size=500 --epoch_number=10 --freeze --pn_rate=1 --time_limit=20
```

train
```
python train_wandb.py --train_batch=16 --valid_batch=64 --model_path=bert-base-chinese --learning_rate=5e-5 --train_rate=0.8 --content_size=100 --epoch_number=20 --freeze --pn_rate=1 --time_limit=100
```



