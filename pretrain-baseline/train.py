
import torch
from torch.utils.data import DataLoader,random_split
from dataset import CaseData
from model import CaseClassification
from transformers import BertTokenizer, AdamW

import wandb
# wandb.login(key='abe8f1e0a080952d6bfe4c3325c9679f92c77d06')


import numpy as np 
import random

import warnings
warnings.filterwarnings("ignore")
import os
save_path='./saved/'
if not os.path.exists(save_path):
    os.mkdir(save_path)

# Random Seed Initialize
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

model_root_path = 'E:/pre-train-model' # model root path on windows
# model_root_path = ''  # model root path on linux
model_list = [
    'bert-base-chinese',
    'chinese-bert-wwm',
]
model_idx =0
model_path = model_root_path + '/'+model_list[model_idx]+'/'
model_path = 'bert-base-chinese'
epochs = 10
learning_rate = 1e-5

config = {
        "learning_rate": 1e-5,
        "architecture": model_list[model_idx],
        "dataset": "0.85",
        "epochs": epochs,
}

# wandb.init(project='case-label', 
#                      name=model_list[model_idx],
#                      job_type="train",
#                      )
print(model_path)
# check the device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

# prepare training  data
is_debug = False
full_data = CaseData('../rawdata/train.json', class_num=252,is_debug=is_debug)             # 252 is the number of level3 labels


train_rate = 0.85
train_size = int(train_rate*len(full_data))
valid_size = len(full_data)-train_size


print(len(full_data),train_size,valid_size)
train_dataset,valid_dataset = random_split(dataset=full_data,lengths=[train_size,valid_size])



# print(len(full_data))
train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
valid_dataloader = DataLoader(valid_dataset,batch_size=16,shuffle=True)

# load the model and tokenizer
model = CaseClassification(class_num=252,model_path=model_path).to(device)
# tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
tokenizer = BertTokenizer.from_pretrained(model_path)

# prepare the optimizer and corresponding hyper-parameters

optimizer = AdamW(model.parameters(), lr=learning_rate)

from tqdm import tqdm

def cal_metrics(pred_choice,target):
    TP,TN,FN,FP = 0,0,0,0
    # TP predict 和 label 同时为1
    TP += ((pred_choice == 1) & (target == 1)).cpu().sum()
    # TN predict 和 label 同时为0
    TN += ((pred_choice == 0) & (target == 0)).cpu().sum()
    # FN predict 0 label 1
    FN += ((pred_choice == 0) & (target == 1)).cpu().sum()
    # FP predict 1 label 0
    FP += ((pred_choice == 1) & (target == 0)).cpu().sum()
    p = TP / (TP + FP)
    r = TP / (TP + FN)
    F1 = 2 * r * p / (r + p)
    acc = (TP + TN) / (TP + TN + FP + FN)
    return acc,p,r,F1

# from torchmetrics.functional import precision_recall
# logits = torch.tensor([[1,-1,0.2][0.4,-0.3,0.5]])
# logits = logits[logits>0]
# label = torch.tensor([[1,0,0],[1,1,0]])


def get_predict_label(logits,threshold):
    for i in range(len(logits)):
        for j in range(len(logits[0])):
            if logits[i][j]>threshold: logits[i][j] = 1
            else: logits[i][j] = 0
    return logits

best_micro_F1 = 0
# start training process
def train_fn(train_dataloader,model,optimizer,epoch):

    print_diff = 50
    model.train()
    running_loss = 0.0
    total_precision = 0
    total_recall = 0
    total_f1 = 0
    total_acc = 0  

    for i, data in tqdm(enumerate(train_dataloader)):
        fact, label = data

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
            
        logits=get_predict_label(logits,0.5)
        for i in range(len(logits)):
            acc,p,r,F1 = cal_metrics(logits[i],label[i])        
            # print(logits[0],label[1])       
            total_precision+= p
            total_recall+= r
            total_f1+= F1
            total_acc+= acc

    return total_acc/train_size,total_precision/train_size,total_recall/train_size,total_f1/train_size

# class MetricMonitor():
#     def __init__(self,data_size):
#         self.precision = 0
#         self.recall = 0
#         self.f1 = 0
#         self.acc = 0   
#         self.data_size = data_size
#     def cal_metric(self,logits,label):
#         for i in range(len(logits)):
#             (pre,rec)=precision_recall(logits[i], label[i], average='micro')
#             self.acc += 
#             self.precision+=
#             self.recall+=
#             self.f1+= accuracy(preds, target)
#         return self.acc,self.f1



@torch.no_grad()
def valid_fn(valid_dataloader,model):
    model.eval()
    total_precision = 0
    total_recall = 0
    total_f1 = 0
    total_acc = 0       
    
    for i, data in enumerate(valid_dataloader):
        fact, label = data

        # tokenize the data text
        inputs = tokenizer(fact, max_length=512, padding=True, truncation=True, return_tensors='pt')
        # move data to device
        input_ids = inputs['input_ids'].to(device)
        token_type_ids = inputs['token_type_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        label = label.to(device)

        

        # forward and backward propagations
        loss, logits = model(input_ids, attention_mask, token_type_ids, label)
        logits=get_predict_label(logits,0.5)

        for i in range(len(logits)):
            acc,p,r,F1 = cal_metrics(logits[i],label[i])        
            # print(logits[0],label[1])       
            total_precision+= p
            total_recall+= r
            total_f1+= F1
            total_acc+= acc
    
    return total_acc/valid_size,total_precision/valid_size,total_recall/valid_size,total_f1/valid_size

        

for epoch in range(epochs):
    acc,precision,recall,f1=train_fn(train_dataloader,model,optimizer,epoch)
    print('Epoch %d Train   acc: %.4f, pre: %.4f, rec: %.4f, f1: %.4f' % (epoch,acc,precision,recall,f1))
    acc,precision,recall,f1=valid_fn(valid_dataloader,model)
    
    print('Epoch %d Valid   acc: %.4f, pre: %.4f, rec: %.4f, f1: %.4f' % (epoch,acc,precision,recall,f1))
    # wandb.log({'loss': running_loss / 50})

torch.save(model.state_dict(), './saved/model'+str(epochs)+'.pth')


# if __name__ == "__main__":