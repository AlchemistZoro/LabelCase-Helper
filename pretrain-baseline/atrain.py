# %%
import torch
from torch.utils.data import DataLoader,random_split
from torch.utils.data import Dataset
from model import CaseClassification
from transformers import BertTokenizer, AdamW
import numpy as np 
import random
import json
from tqdm import tqdm

# %%
is_debug = False

# %%
model_root_path = 'E:/pre-train-model' # model root path on windows
epochs = 5
learning_rate = 1e-5
model_list = [
    'bert-base-chinese',
    'chinese-bert-wwm',
    
]
model_idx =0
model_path = model_root_path + '/'+model_list[model_idx]+'/'
model_path = model_list[model_idx]
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# %%
import pandas as pd
process_data_path = '../preprocessdata/'
data = pd.read_csv(process_data_path+'train.csv')

# 大约有100多个content是空值，后面会留下错误
data=data[data["content"].notnull()]

vec_frame = pd.read_csv(process_data_path+'vec_frame.csv')
debug = True
if debug:
    vec_frame = vec_frame.head(200)
train_rate = 0.80
case_size = vec_frame.shape[0]
# print(data.shape)
# data.dropna(how='any',axis=0)
# print(data.shape)

train_index_list = vec_frame.iloc[0:int(train_rate*case_size),0]
valid_index_list = vec_frame.iloc[int(train_rate*case_size):,0]


train_label = vec_frame[vec_frame["0"].isin(train_index_list)]
valid_label = vec_frame[vec_frame["0"].isin(valid_index_list)]
train_data = data[data["id"].isin(train_index_list)]
valid_data = data[data["id"].isin(valid_index_list)]

num = train_data.shape[0]
p_num = train_data[train_data["label"]!= '234'].shape[0]
n_num = train_data[train_data["label"]== '234'].shape[0]
print('样本数:',num)
print('正样本数:',p_num)
print('负样本数:',n_num)
print('正负样本比例:',p_num/n_num)
pn_rate = 100
if pn_rate<p_num/n_num:
    n_change_num = n_num
else:
    n_change_num = int(p_num/pn_rate)

n_train_data=train_data[train_data["label"]=='234'].sample(n=n_change_num,replace=True)
p_train_data=train_data[train_data["label"]!='234']
train_data_now = pd.concat([n_train_data,p_train_data],axis=0)

print(train_data_now.shape)

# %%
from torch.utils.data import Dataset
import torch
from model import CaseClassification
from transformers import BertTokenizer, AdamW
from torch.utils.data import DataLoader,random_split







# %%

class CaseData(Dataset):
    def __init__(self, data,index_list,class_num):
        self.data = data
        self.index_list = index_list
        self.class_num = class_num
 

    def __getitem__(self, idx):
        fact = self.data.iloc[idx,2]
        id = int(self.data.iloc[idx,0])
        label_list = self.data.iloc[idx,1]
        label = torch.zeros(self.class_num)
        for i in label_list.split("#"):
            label[int(i)] = 1
        
        return id,fact, label

    def __len__(self):
        return len(self.data)

class_num = 235
train_dataset = CaseData(train_data_now,train_index_list,class_num=class_num)
valid_dataset = CaseData(valid_data,valid_index_list,class_num=class_num)

# print(len(full_data))
train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
valid_dataloader = DataLoader(valid_dataset,batch_size=16,shuffle=False)






# %%
# print(train_data_now["id"].unique())
# print(train_index_list)
a,b,c=train_dataset.__getitem__(1)
c

# %%
# 空文本会带来问题,用这个方法筛出来了
def check_dataset(data_set):
    for i in range(train_dataset.__len__()):
        a,b,c=train_dataset.__getitem__(i)  
        if not (isinstance (a,int) and isinstance(b,str) ):
            print(type(a),type(b),type(c),b)
check_dataset(valid_dataset)
print(valid_dataset.__len__())

# %%
class_num = 235
# load the model and tokenizer
model = CaseClassification(class_num=class_num,model_path=model_path).to(device)
# tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
tokenizer = BertTokenizer.from_pretrained(model_path)

# prepare the optimizer and corresponding hyper-parameters

optimizer = AdamW(model.parameters(), lr=learning_rate)


# %%


label = valid_label.values[:,1:]
valid_size = valid_data.shape[0]
valid_case_size = len(valid_label)
valid_index_dict = dict(zip(valid_index_list,range(len(valid_index_list))))

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
    p = TP / (TP + FP+1)
    r = TP / (TP + FN+1)
    F1 = 2 * r * p / (r + p+1)
    acc = (TP + TN) / (TP + TN + FP + FN+1)
    return acc,p,r,F1

def get_predict_label(logits,threshold):
    for i in range(len(logits)):
        for j in range(len(logits[0])):
            if logits[i][j]>threshold: logits[i][j] = 1
            else: logits[i][j] = 0
    return logits


def train_fn(train_dataloader,optimizer,epoch):

    print_diff = 50
    model.train()
    running_loss = 0.0
    total_precision = 0
    total_recall = 0
    total_f1 = 0
    total_acc = 0  

    for i, data in enumerate(train_dataloader):
        id,fact, label= data

        # tokenize the data text
        inputs = tokenizer(fact, max_length=512, padding=True, truncation=True, return_tensors='pt')
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

        if i % print_diff == print_diff -1 :
            print('Epoch %d Train, step: %2d, train_loss: %.3f' % (epoch + 1, i + 1, running_loss / print_diff))
            # wandb.log({'train_loss': running_loss / 50})
            running_loss = 0.0

    


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

    
    for i, data in enumerate(valid_dataloader):
        id,fact, c_label= data

        # tokenize the data text
        inputs = tokenizer(fact, max_length=512, padding=True, truncation=True, return_tensors='pt')
        # move data to device
        input_ids = inputs['input_ids'].to(device)
        token_type_ids = inputs['token_type_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        c_label = c_label.to(device)



        # forward and backward propagations
        loss, logits = model(input_ids, attention_mask, token_type_ids, c_label)

        threshold = 0.5
        logits[logits>threshold] = 1
        logits[logits<=threshold] = 0



        logits=logits.cpu().numpy()
        # for i in range(logits.shape[0]):
        #     c_predict=np.zeros(class_num)
        #     c_predict[np.argmax(logits[i])]=1
        #     logits[i]=c_predict
            
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


    predict[predict>1] = 1

    for i in range(predict.shape[0]):

        acc,p,r,F1 = cal_metrics(predict[i][0:-1],label[i]) 
        case_precision+= p
        case_recall+= r
        case_f1+= F1
        case_acc+= acc

        
    print('Epoch %d Sen Valid   acc: %.4f, pre: %.4f, rec: %.4f, f1: %.4f' % (epoch,total_acc/valid_size,total_precision/valid_size,total_recall/valid_size,total_f1/valid_size,))
    print('Epoch %d Case Valid   acc: %.4f, pre: %.4f, rec: %.4f, f1: %.4f' % (epoch,case_acc/valid_case_size,case_precision/valid_case_size,case_recall/valid_case_size,case_f1/valid_case_size,))
  
for epoch in range(epochs):
    train_fn(train_dataloader,optimizer,epoch)
    valid_fn(valid_dataloader,epoch)
    


# %%



