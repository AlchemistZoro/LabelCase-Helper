from torch.utils.data import DataLoader,random_split
from torch.utils.data import Dataset
from transformers import BertTokenizer, AdamW,AutoModel, AutoTokenizer
import numpy as np 
import random
import json
from tqdm import tqdm
import pandas as pd
import time
import datetime
from torch.utils.data import Dataset
import torch
from transformers import BertTokenizer, AdamW
from torch.utils.data import DataLoader,random_split
import hashlib
import torch.nn as nn
import os
import argparse
from data_process import get_dataset


class CaseClassification(nn.Module):
    def __init__(self, class_num,model_path):
        super(CaseClassification, self).__init__()
        self.bert = AutoModel.from_pretrained(model_path)
        self.linear = nn.Linear(in_features=768, out_features=class_num)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, label=None):
        outputs = self.bert(input_ids, attention_mask, token_type_ids)
        pooler_output = outputs['pooler_output']

        logits = self.linear(pooler_output)
        # logits = torch.sigmoid(logits)
        if label is not None:
            loss_fn = nn.BCELoss()
            # loss_fn = nn.BCEWithLogitsLoss()
            # loss = loss_fn(logits, label)
            loss = multilabel_cross_entropy(logits, label)
            return loss, logits

        return logits

class CaseData(Dataset):
    def __init__(self, data,label,class_num):
        self.data = data
        self.label = label
        self.class_num = class_num
 

    def __getitem__(self, idx):
        fact = self.data.iloc[idx,-1]
        id = int(self.data.iloc[idx,-2])
        l = torch.tensor(self.data.iloc[idx,0:class_num], dtype=float)
        # print(fact,id,l)

        return id,fact, l

    def __len__(self):
        return len(self.data)

def out_time_limit(start,limit_minutes):
    now = datetime.datetime.now()  
    result = (now - start).total_seconds()
    if result>60*limit_minutes:
        return True
    return False

def multilabel_cross_entropy(y_pred, y_true):
    """
    https://kexue.fm/archives/7359
    """
    y_pred = (1 - 2 * y_true) * y_pred  # -1 -> pos classes, 1 -> neg classes
    y_pred_neg = y_pred - y_true * 1e12  # mask the pred outputs of pos classes
    y_pred_pos = (y_pred - (1 - y_true) * 1e12)  # mask the pred outputs of neg classes
    zeros = torch.zeros_like(y_pred[..., :1])
    y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
    y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
    neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
    pos_loss = torch.logsumexp(y_pred_pos, dim=-1)

    return (neg_loss + pos_loss).mean()

def cal_metrics(pred_choice,target):
    TP,TN,FN,FP = 0,0,0,0
    # TP predict 和 label 同时为1
    TP += ((pred_choice == 1) & (target == 1)).sum()
    # TN predict 和 label 同时为0
    TN += ((pred_choice == 0) & (target == 0)).sum()
    # FN predict 0 label 1
    FN += ((pred_choice == 0) & (target == 1)).sum()
    # FP predict 1 label 0
    FP += ((pred_choice == 1) & (target == 0)).sum()
    p = TP / (TP + FP+0.001)
    r = TP / (TP + FN+0.001)
    F1 = 2 * r * p / (r + p+0.0001)
    acc = (TP + TN) / (TP + TN + FP + FN)
    return acc,p,r,F1


        
# 分类转标签
def get_level3labels(tree):
    level3labels = []
    for t in tree:
        level1 = t['value']
        children1 = t['children']
        for child1 in children1:
            level2 = child1['value']
            children2 = child1['children']
            for child2 in children2:
                level3 = child2['value']
                level3labels.append('/'.join([level1, level2, level3]))
    return level3labels

def seed_everything(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


parser = argparse.ArgumentParser(description='Data process')
parser.add_argument('--debug', action="store_true")
parser.add_argument('--debug_train_num', default=100,type=int)
parser.add_argument('--debug_valid_num', default=20,type=int)
parser.add_argument('--train_batch', default=16,type=int)
parser.add_argument('--valid_batch', default=64,type=int)

parser.add_argument('--model_path', default='bert-base-chinese')
parser.add_argument('--learning_rate', default=0.00001,type=float)
parser.add_argument('--train_rate', default=0.8,type=float)
parser.add_argument('--content_size', default=100,type=int)
parser.add_argument('--token_size', default=512,type=int)

parser.add_argument('--epoch_number', default=10,type=int)

parser.add_argument('--freeze', action="store_true")

parser.add_argument('--pn_rate', default=1,type=float)
parser.add_argument('--class_num', default=234,type=int)

parser.add_argument('--time_limit', default=30,type=int)
parser.add_argument('--f1_limit', default=0.8,type=float)
parser.add_argument('--diff_limit', default=2,type=int)
parser.add_argument('--f1_save_limit', default=0.3,type=float)

parser.add_argument('--loss', default='MCE')
parser.add_argument('--optm', default='Adam')

parser.add_argument('--input_dir', default='./input/train.json')
parser.add_argument('--output_dir', default='./output/answer.json')
args = parser.parse_args()


if __name__ == "__main__":
    input_path = args.input_dir
    output_path = args.output_dir

    debug = args.debug
    debug_train_num = args.debug_train_num
    debug_valid_num = args.debug_valid_num
    train_batch = args.train_batch
    valid_batch = args.valid_batch
    model_path = args.model_path   #'chinese-bert-wwm',
    learning_rate = args.learning_rate
    train_rate = args.train_rate
    content_size = args.content_size

    epoch_number = args.epoch_number
    freeze = args.freeze
    loss= args.loss # BCE、BCEWG
    optm = args.optm 

    pn_rate = args.pn_rate

    class_num = args.class_num

    time_limit =args.time_limit
    f1_limit = args.f1_limit
    diff_limit =args.diff_limit
    f1_save_limit = args.f1_save_limit 
    token_size = args.token_size

    dic = vars(args)
    model_name=hashlib.md5(str(int(time.time())).encode("utf-8")).hexdigest()[0:10]
    print(dic)



    process_data_path = '../processeddata/tr-%s-%s/' %(str(train_rate),str(content_size))

    level3labels = get_level3labels(json.load(open('./input/code_tree.json')))





    seed_everything()

    start = datetime.datetime.now()  
    # model download from hugging-face


    # tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
    # model_path = "thunlp/Lawformer"
    if model_path== "thunlp/Lawformer":
        tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
    else:
        tokenizer = BertTokenizer.from_pretrained(model_path)


    device = 'cuda' if torch.cuda.is_available() else 'cpu'




    valid_label,valid_data=get_dataset(content_size)

    valid_data=valid_data[valid_data.iloc[:,-1].notnull()]

    if debug :
        valid_label=valid_label.head(debug_valid_num)
        valid_data=valid_data[valid_data["id"].isin(valid_label.iloc[:,0])]

    valid_dataset = CaseData(valid_data,valid_label,class_num=class_num)


    valid_dataloader = DataLoader(valid_dataset,batch_size=valid_batch,shuffle=False)




    model_name = "9b1519a4f2"
    model_load_path = "./saved/%s.pth" % (model_name)
    model = torch.load(model_load_path).to(device)

    valid_index_list=valid_label.values[:,0]
    label = valid_label.values[:,1:]
    valid_size = valid_data.shape[0]
    valid_case_size = len(valid_label)
    valid_index_dict = dict(zip(valid_index_list,range(len(valid_index_list))))


    @torch.no_grad()
    def valid_fn(valid_dataloader,epoch,level3labels):
        predict=np.zeros((len(valid_index_list),class_num))

        model.eval()
        total_precision = 0
        total_recall = 0
        total_f1 = 0
        total_acc = 0     

        case_precision = 0
        case_recall = 0
        case_f1 = 0
        case_acc = 0  
        running_loss =0
        n=0
        with tqdm(valid_dataloader) as t:
            for i, data in enumerate(t):
                id,fact, c_label= data

                # tokenize the data text
                inputs = tokenizer(list(fact), max_length=512, padding=True, truncation=True, return_tensors='pt')
                # move data to device
                input_ids = inputs['input_ids'].to(device)
                token_type_ids = inputs['token_type_ids'].to(device)
                attention_mask = inputs['attention_mask'].to(device)
                c_label = c_label.to(device)



                # forward and backward propagations
                loss, logits = model(input_ids, attention_mask, token_type_ids, c_label)
                n+=len(id)
                # print statistics
                running_loss += loss.item()
                t.set_postfix(loss=loss.item())

                threshold = 0
                logits[logits>threshold] = 1
                logits[logits<=threshold] = 0



                logits=logits.cpu().numpy()

                # 单句标签f1
                c_label = c_label.cpu().numpy()
                for i in range(len(id)):
                    idx = int(id[i])
                    row_idx = valid_index_dict[idx]
                    predict[row_idx] += logits[i]
                

            # 案件标签f1
            predict[predict>1] = 1
            for i in range(predict.shape[0]):

                acc,p,r,F1 = cal_metrics(predict[i],label[i]) 
                case_precision+= p
                case_recall+= r
                case_f1+= F1
                case_acc+= acc
            print('Valid Loss:',running_loss/len(valid_dataloader))

            print('Epoch %d Case Valid   acc: %.4f, pre: %.4f, rec: %.4f, f1: %.4f' % (epoch+1,case_acc/valid_case_size,case_precision/valid_case_size,case_recall/valid_case_size,case_f1/valid_case_size))

            
            return predict
    predict=valid_fn(valid_dataloader,0,level3labels)



    results = []
    for p in range(predict.shape[0]):
        result = []
        for i in range(predict.shape[1]):
            if predict[p][i]==1:
                # print(level3labels[i])
                result.append(level3labels[i])
        results.append(result)

    json.dump(results, open(output_path, "w", encoding="utf8"), indent=2, ensure_ascii=False)


