
采用什么模型需要提前先定好，模型保存在saved文件夹下。
当前采用的模型文件为：9b1519a4f2.pth。

运行：
```
python main.py --train_batch 32 --valid_batch 64 --model_path hfl/chinese-roberta-wwm-ext --learning_rate 2e-5 --train_rate 0.9 --content_size 300 --epoch_number 20 --freeze --pn_rate 0 --time_limit 500 
```