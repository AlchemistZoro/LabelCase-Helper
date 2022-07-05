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


def create_train_dataset(textual_frame,code_frame,data,process_data_path):
    columns = ['id','label','content']
    label_size=textual_frame.shape[0]
    process_data = []
    y=0
    label_set = set()
    label_c_set = set()
    for i in tqdm(data):
        content_list = []
        for j in i["content"]:
            content_list.append([i["id"],label_size,j])
    
        for j in i["evidence"]:
            idx = j["index"]
            value_list = j["value"]
            
            pro_value_list = []
            for z in value_list:
                label_set.add(z)
                code_label = z.split('/')[-1]
                label_c_set.add(code_label)
                num_label = code_frame[code_frame["C"]==code_label].index.values[0]
                pro_value_list.append(str(num_label))
                
            content_list[idx][1] = '#'.join(pro_value_list)
        

        if not process_data:
            process_data = content_list
        else:
            process_data=process_data+content_list
    # 234
    # print(len(label_set))
    # print(len(label_c_set))
    dataset = pd.DataFrame(process_data,columns=columns)
    # 
    return dataset


def create_vectorlabel_dataset(data,process_data_path):
    vectorlabel_list = []    
    case_label_num = []
    label_num=np.zeros(234,'int64')
    for i in data:
        vectorlabel = np.zeros(235,'int64')
        vectorlabel[0]=int(i["id"])
        for j in i["result"]: 
            idx=int(j.split('/')[-1][1:])
            vectorlabel[idx+1] = 1
            label_num[idx]+=1
        case_label_num.append(len(i["result"]))    
        vectorlabel_list.append(vectorlabel)

    vec_frame = pd.DataFrame(np.array(vectorlabel_list))
    # vec_frame.to_csv(process_data_path+'vec_frame.csv',index = False)
    return vec_frame,case_label_num,label_num

parser = argparse.ArgumentParser(description='Data process')
parser.add_argument('--train_rate', default=0.8,type=float)
parser.add_argument('--content_size', default=500,type=int)
args = parser.parse_args()

if __name__ == "__main__":
    train_rate=args.train_rate
    content_size=args.content_size
    print(train_rate,content_size)
    RANDOM_SEED = 42
    seed_everything(RANDOM_SEED)
    process_data_path = './processeddata/'
    if not os.path.exists(process_data_path):
        os.mkdir(process_data_path)

    code_tree=json.load(open('./rawdata/code_tree.json', encoding='utf-8'))
    textual_tree = json.load(open('./rawdata/textual_tree.json', encoding='utf-8'))
    code_list = get_level3labels(code_tree)
    textual_list = get_level3labels(textual_tree)
    # 216
    # print(len(code_list))
    # print(len(textual_list))
    raw_data=json.load(open('./rawdata/train.json', encoding='utf-8'))

    '''
    textual_frame: [A,B,C] 三级标签-中文
    code_frame: [A,B,C] 三级标签-代码
    text_code_frame: [C1,C2] 三级标签映射
    '''
    textual_frame,code_frame,text_code_frame=labeltree_dataset(textual_tree,code_tree,process_data_path)
    map_path='map/'
    if not os.path.exists(process_data_path+map_path):
        os.mkdir(process_data_path+map_path)
    textual_frame.to_csv(process_data_path+map_path+'textual_frame.csv',index=False)
    code_frame.to_csv(process_data_path+map_path+'code_frame.csv',index=False)
    text_code_frame.to_csv(process_data_path+map_path+'text_code_frame.csv',index=False)


    '''
    train_frame:[id,label,text]
    '''
    data=create_train_dataset(textual_frame,code_frame,raw_data,process_data_path)
    data.to_csv(process_data_path+'train.csv',index = False)
    '''
    vec_frame:[id,[label_vec]]
    '''
    case_data,case_label_num,label_num=create_vectorlabel_dataset(raw_data,process_data_path)
    case_data.to_csv(process_data_path+'case.csv',index = False)

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


    '''
    r=[0.8,0.9,1]，r=1：全量训练模型 
    '''

    train_data_path = process_data_path+"tr-%s-%s/"%(str(train_rate),str(content_size))
    if not os.path.exists(train_data_path):
        os.mkdir(train_data_path)

    train_label=case_data.sample(frac=train_rate,replace=False) #抽取20%的数据
    train_label.to_csv(train_data_path+'train_label.csv',index = False)
    train_data = data_dw_frame[data_dw_frame["id"].isin(train_label[0])]
    train_data.to_csv(train_data_path+'train_data.csv',index = False)

    print('train label shape:',train_label.shape)

    if not train_rate==1:
        valid_label=case_data[~case_data.index.isin(train_label.index)]
        valid_label.to_csv(train_data_path+'valid_label.csv',index = False)
        valid_data = data_dw_frame[data_dw_frame["id"].isin(valid_label[0])]
        valid_data.to_csv(train_data_path+'valid_data.csv',index = False)
        print('valid label shape:',valid_label.shape)

