
采用什么模型需要提前先定好，模型保存在saved文件夹下。
当前采用的模型文件为：9b1519a4f2.pth。

运行：
1. 进入项目目录。
2. 在input项目目录中，添加：code_tree.json,train.json文件
3. 创建output文件目录。
4. 将训练号的模型添加至saved路径下。
5. 运行程序：
```
python main.py --train_batch 32 --valid_batch 64 --model_path hfl/chinese-roberta-wwm-ext --learning_rate 2e-5 --train_rate 0.9 --content_size 300 --epoch_number 20 --freeze --pn_rate 0 --time_limit 500 
```