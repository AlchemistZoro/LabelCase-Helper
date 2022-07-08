import pandas as pd
import numpy as np
import os
import json
from tqdm import tqdm
import random
import argparse


def seed_everything(seed=42):
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        random.seed(seed)


def tree_to_table(tree):
    table= []
    for i in tree:
        a = i["value"]
        for j in i["children"]:
            b = j["value"]
            for z in j["children"]:
                c=z["value"]
                table.append([a,b,c])
    return table

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

def labeltree_dataset(textual_tree,code_tree,process_data_path):
    textual_table = []
    code_table = []


    textual_table = tree_to_table(textual_tree)
    code_table = tree_to_table(code_tree)
    textual_frame = pd.DataFrame(textual_table,columns=["A","B","C"])
    code_frame = pd.DataFrame(code_table,columns=["A","B","C"])

    text_code_frame = pd.concat([code_frame["C"],textual_frame["C"]],axis=1)
    text_code_frame.columns = ["code_label","textual_label"]

    code_dict = dict(zip(code_frame["C"],textual_frame["C"]))
    # print("code_dict size:",len(code_dict))
    # print("code_frame size",code_frame.shape[0])
    # print("text_frame size",textual_frame.shape[0])



    return textual_frame,code_frame,text_code_frame


def create_train_dataset(data,process_data_path):
    columns = ['id','label','content']
    label_size=234
    process_data = []
    y=0
    label_set = set()
    label_c_set = set()
    for i in tqdm(data):
        content_list = []
        for j in i["content"]:
            content_list.append([i["id"],label_size,j])
    

        if not process_data:
            process_data = content_list
        else:
            process_data=process_data+content_list
    dataset = pd.DataFrame(process_data,columns=columns)

    return dataset


def create_vectorlabel_dataset(data,process_data_path):
    vectorlabel_list = []    
    case_label_num = []
    label_num=np.zeros(234,'int64')
    for i in data:
        vectorlabel = np.zeros(235,'int64')
        vectorlabel[0]=int(i["id"])
        case_label_num.append(len(i["result"]))    
        vectorlabel_list.append(vectorlabel)

    vec_frame = pd.DataFrame(np.array(vectorlabel_list))
    # vec_frame.to_csv(process_data_path+'vec_frame.csv',index = False)
    return vec_frame,case_label_num,label_num

# def create_vectorlabel_dataset(data,process_data_path):
#     vectorlabel_list = []    
#     case_label_num = []
#     label_num=np.zeros(234,'int64')
#     for i in data:
#         vectorlabel = np.zeros(235,'int64')
#         vectorlabel[0]=int(i["id"])
#         for j in i["result"]: 
#             idx=int(j.split('/')[-1][1:])
#             vectorlabel[idx+1] = 1
#             label_num[idx]+=1
#         case_label_num.append(len(i["result"]))    
#         vectorlabel_list.append(vectorlabel)

#     vec_frame = pd.DataFrame(np.array(vectorlabel_list))
#     # vec_frame.to_csv(process_data_path+'vec_frame.csv',index = False)
#     return vec_frame,case_label_num,label_num


def get_dataset(content_size=300):
    train_rate=1
    print(train_rate,content_size)
    RANDOM_SEED = 42
    seed_everything(RANDOM_SEED)
    process_data_path = './input/'
    if not os.path.exists(process_data_path):
        os.mkdir(process_data_path)

    raw_data=json.load(open('./input/train.json', encoding='utf-8'))

    '''
    train_frame:[id,label,text]
    '''
    data=create_train_dataset(raw_data,process_data_path)

    '''
    vec_frame:[id,[label_vec]]
    '''
    case_data,case_label_num,label_num=create_vectorlabel_dataset(raw_data,process_data_path)

    # 删除没有正类的样本集
    section_set = set()
    delete_sections = ['当事人信息','再审被申请人辩称','被上诉人答辩','审判人员','裁判结果','开始','']
    pre_id = data.loc[0][0]
    pre_key = '开始'
    now_key = '开始'
    section_labels=[]
    for i,row in tqdm(data.iterrows()):
        now_id=row["id"]
        if pre_id !=now_id:
            now_key = "开始"
        if row['content'] != "" and row["content"][0] =='【':
            now_key = row['content'][1:-1]
            section_set.add(row['content'])
        row['section_label'] = now_key
        section_labels.append(now_key)
        pre_key=now_key 
        pre_id=now_id 
    data['section_label']=section_labels
    data=data[~data["section_label"].isin(delete_sections)]
    print('删除全负section样本后的个数：',data.shape[0])
    data=data[~data["content"].isin(section_set)]
    print('删除头样本后的个数：',data.shape[0])


    onehot_labels = []
    class_num = 234
    for i,row in tqdm(data.iterrows()):
        labels = np.zeros(class_num)
        if row["label"] == class_num:
            onehot_labels.append(labels)
            continue    
        for idx in row["label"].split("#"):
            labels[int(idx)] = 1
        onehot_labels.append(labels)
    onehot_labels = np.array(onehot_labels)



    onehot_labels=pd.DataFrame(onehot_labels,columns=[i for i in range(class_num)])
    data=data.reset_index(drop=True)
    data_onehot=pd.concat([data,onehot_labels],axis=1)

    data_windows = []
    # content_label = np.zeros(234)
    for i,group in tqdm(data_onehot.groupby('id')):
        content_label = np.zeros(234)
        id = group.iloc[0]['id']
        content_windows = ''
        start_index = 0
        end_index = 0
        while len(content_windows)+len(group.iloc[end_index]['content'])<content_size:
            content_label=content_label+group.iloc[end_index].values[4:]
            content_windows+=group.iloc[end_index]['content']
            end_index+=1
        # for idx,row in group.iterrows():
        current_label = content_label.copy()
        current_label[current_label>1] =1
        current_label = list(current_label)
        current_label.append(id)
        current_label.append(content_windows)
        data_windows.append(current_label)
        while end_index<group.shape[0]:
            
            content_label=content_label+group.iloc[end_index].values[4:]
            content_windows+=group.iloc[end_index]['content']
            end_index+=1
            while len(content_windows)>content_size:
                content_label=content_label-group.iloc[start_index].values[4:]
                start_size = len(group.iloc[start_index]['content'])
                content_windows= content_windows[start_size:]
                start_index+=1
            current_label = content_label.copy()
            current_label[current_label>1] =1
            current_label = list(current_label)
            current_label.append(id)
            current_label.append(content_windows)
            data_windows.append(current_label)


    columns_dw = [str(i) for i in range(class_num) ]
    columns_dw.append('id')
    columns_dw.append('content')
    data_dw_frame=pd.DataFrame( data_windows,columns=columns_dw)

    train_data_path = process_data_path+"tr-%s-%s/"%(str(train_rate),str(content_size))
    if not os.path.exists(train_data_path):
        os.mkdir(train_data_path)

    case_data.to_csv(train_data_path+'valid_label.csv',index = False)
    data_dw_frame.to_csv(train_data_path+'valid_data.csv',index = False)
    return case_data,data_dw_frame





if __name__ == "__main__":
    _,_=get_dataset(300)