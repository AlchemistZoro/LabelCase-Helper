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
import wandb


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

parser = argparse.ArgumentParser(description='Data process')
parser.add_argument('--debug', default=True,type=bool)
parser.add_argument('--debug_train_num', default=100,type=int)
parser.add_argument('--debug_valid_num', default=20,type=int)
parser.add_argument('--train_batch', default=16,type=int)
parser.add_argument('--valid_batch', default=64,type=int)

parser.add_argument('--model_path', default='bert-base-chinese')
parser.add_argument('--learning_rate', default=0.00001,type=float)
parser.add_argument('--train_rate', default=0.8,type=float)
parser.add_argument('--content_size', default=100,type=int)

parser.add_argument('--epoch_number', default=10,type=int)

parser.add_argument('--freeze', default=True,type=bool)
parser.add_argument('--loss', default='MCE')
parser.add_argument('--optm', default='Adam')
parser.add_argument('--pn_rate', default=1,type=int)
parser.add_argument('--class_num', default=234,type=int)

parser.add_argument('--time_limit', default=30,type=int)
parser.add_argument('--f1_limit', default=0.55,type=float)
parser.add_argument('--diff_limit', default=2,type=int)
parser.add_argument('--f1_save_limit', default=0.3,type=float)
args = parser.parse_args()


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

dic = vars(args)

wandb.login()
wandb.init(project='case-label', 
                     job_type="train",
                     config = dic
                     )

process_data_path = '../processeddata/tr-%s-%s/' %(str(train_rate),str(content_size))
# debug = True
# debug_train_num = 100
# debug_valid_num = 20
# train_batch = 16
# valid_batch = 64
# model_path = 'bert-base-chinese'   #'chinese-bert-wwm',
# learning_rate = 5e-5
# train_rate = 0.8
# content_size = 100
# process_data_path = '../processeddata/tr-%s-%s/' %(str(train_rate),str(content_size))
# epoch_number = 10
# freeze = True
# loss= 'MCE' # BCE、BCEWG
# optm = 'Adam'
# pn_rate = 1

# class_num = 234

# time_limit = 30
# f1_limit = 0.55
# diff_limit =2
# f1_save_limit = 0.3


RANDOM_SEED = 42

def seed_everything(seed=RANDOM_SEED):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything()

start = datetime.datetime.now()  
# model download from hugging-face
model_path = 'bert-base-chinese'
tokenizer = BertTokenizer.from_pretrained(model_path)
# tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
# model_path = "thunlp/Lawformer"

device = 'cuda' if torch.cuda.is_available() else 'cpu'


'''
data=[onehot-vec,id,content]
label = [0,onehot-vec]
'''

process_data_path = '../processeddata/tr-0.8-100/'
train_data = pd.read_csv(process_data_path+'train_data.csv')
valid_data = pd.read_csv(process_data_path+'valid_data.csv')
train_label = pd.read_csv(process_data_path+'train_label.csv')
valid_label = pd.read_csv(process_data_path+'valid_label.csv')

print(train_data.shape)
print(valid_data.shape)
print(train_label.shape)
print(valid_label.shape)


train_data=train_data[train_data.iloc[:,-1].notnull()]
valid_data=valid_data[valid_data.iloc[:,-1].notnull()]
print(train_data.shape)
print(valid_data.shape)
print(train_label.shape)
print(valid_label.shape)

if debug :
    train_label=train_label.head(debug_train_num)
    valid_label=valid_label.head(debug_valid_num)
    train_data=train_data[train_data["id"].isin(train_label.iloc[:,0])]
    valid_data=valid_data[valid_data["id"].isin(valid_label.iloc[:,0])]

print(train_data.shape)
print(valid_data.shape)
print(train_label.shape)
print(valid_label.shape)

num = train_data.shape[0]
train_data_sum = []
for i in tqdm(range(num)):
    train_data_sum.append(train_data.iloc[i,0:234].sum())
train_data['sum']=train_data_sum

p_num = train_data[train_data["sum"]!= 0].shape[0]
n_num = train_data[train_data["sum"]== 0].shape[0]
print('样本数:',num)
print('正样本数:',p_num)
print('负样本数:',n_num)
print('正负样本比例:',p_num/n_num)

if pn_rate<p_num/n_num:
    n_change_num = n_num
else:
    n_change_num = int(p_num/pn_rate)

n_train_data=train_data[train_data["sum"]==0].sample(n=n_change_num,replace=True)
p_train_data=train_data[train_data["sum"]!=0]
train_data_now = pd.concat([n_train_data,p_train_data],axis=0)

print('训练样本数：',train_data_now.shape)
del train_data_now['sum']


train_dataset = CaseData(train_data_now,train_label,class_num=class_num)
valid_dataset = CaseData(valid_data,valid_label,class_num=class_num)

# print(len(full_data))
train_dataloader = DataLoader(train_dataset, batch_size=train_batch, shuffle=True)
valid_dataloader = DataLoader(valid_dataset,batch_size=valid_batch,shuffle=False)



    



# load the model and tokenizer
model = CaseClassification(class_num=class_num,model_path=model_path).to(device)
# tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
# tokenizer = BertTokenizer.from_pretrained(model_path)

# prepare the optimizer and corresponding hyper-parameters

no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
{'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
{'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]

if freeze:
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
else:
    optimizer = AdamW(model.parameters(), lr=learning_rate)

valid_index_list=valid_label.values[:,0]
label = valid_label.values[:,1:]
valid_size = valid_data.shape[0]
valid_case_size = len(valid_label)
valid_index_dict = dict(zip(valid_index_list,range(len(valid_index_list))))



# def get_predict_label(logits,threshold):
#     for i in range(len(logits)):
#         for j in range(len(logits[0])):
#             if logits[i][j]>threshold: logits[i][j] = 1
#             else: logits[i][j] = 0
#     return logits


def train_fn(train_dataloader,optimizer,epoch):

    print_diff = 50
    model.train()
    running_loss = 0.0
    total_precision = 0
    total_recall = 0
    total_f1 = 0
    total_acc = 0  

    with tqdm(train_dataloader) as t:
        for i, data in enumerate(t):
            id,fact, label= data

            # tokenize the data text
            inputs = tokenizer(list(fact), max_length=512, padding=True, truncation=True, return_tensors='pt')
        
            # move data to device
            input_ids = inputs['input_ids'].to(device)
            token_type_ids = inputs['token_type_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)
            label = label.to(device)
            # forward and backward propagations
            loss, logits = model(input_ids, attention_mask, token_type_ids, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.item()
            logits = logits.cpu()
            # t.set_postfix(loss=loss.item(),min=torch.min(logits),max=torch.max(logits),mean = torch.mean(logits))
            t.set_postfix(loss=loss.item())
            wandb.log({'Batch Train Loss': loss.item()})
        print('Train Loss:',running_loss/len(train_dataloader))
        wandb.log({'Epoch Train Loss': running_loss/len(train_dataloader)})



    


@torch.no_grad()
def valid_fn(valid_dataloader,epoch):
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
            # for i in range(logits.shape[0]):
            #     c_predict=np.zeros(class_num)
            #     c_predict[np.argmax(logits[i])]=1
            #     logits[i]=c_predict
            
            # 单句标签f1
            c_label = c_label.cpu().numpy()
            for i in range(len(id)):
                idx = int(id[i])
                row_idx = valid_index_dict[idx]
                predict[row_idx] += logits[i]
            


            for i in range(len(logits)):
                acc,p,r,F1 = cal_metrics(logits[i],c_label[i])        
                # print(logits[0],label[1])       
                total_precision+= p
                total_recall+= r
                total_f1+= F1
                total_acc+= acc

        # 案件标签f1
        predict[predict>1] = 1
        for i in range(predict.shape[0]):

            acc,p,r,F1 = cal_metrics(predict[i],label[i]) 
            case_precision+= p
            case_recall+= r
            case_f1+= F1
            case_acc+= acc
        print('Valid Loss:',running_loss/len(valid_dataloader))
            
        print('Epoch %d Sen Valid   acc: %.4f, pre: %.4f, rec: %.4f, f1: %.4f' % (epoch+1,total_acc/valid_size,total_precision/valid_size,total_recall/valid_size,total_f1/valid_size,))
        print('Epoch %d Case Valid   acc: %.4f, pre: %.4f, rec: %.4f, f1: %.4f' % (epoch+1,case_acc/valid_case_size,case_precision/valid_case_size,case_recall/valid_case_size,case_f1/valid_case_size))
        return case_f1/valid_case_size

def model_save(model,model_name):
    ## 保存模型
    torch.save(model, './saved/%s.pth' % (model_name)) 
    print('save')

best_f1 = 0
model_name=hashlib.md5("123456".encode("utf-8")).hexdigest()
for epoch in range(epoch_number):
    train_fn(train_dataloader,optimizer,epoch)
    now_f1=valid_fn(valid_dataloader,epoch)
    if now_f1>best_f1:
        diff = 0
        best_f1 = now_f1
        if now_f1>f1_save_limit:
            model_save(model,model_name)
        if now_f1>f1_limit:          
            break
    else:
        diff+=1
        if diff>=diff_limit :
            break
        if out_time_limit(start,time_limit):
            break