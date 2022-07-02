# EXPLORATION DATA ANALYSIS & DATASET CREATE
import pandas as pd
import numpy as np
import os
import json
from torch import int16
from tqdm import tqdm

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



    textual_frame.to_csv(process_data_path+'textual_frame.csv',index=False)
    code_frame.to_csv(process_data_path+'code_frame.csv',index=False)
    text_code_frame.to_csv(process_data_path+'text_code_frame.csv',index=False)

    return textual_frame,code_frame,text_code_frame

def print_list(l):
    for i in l: print(i,'\n')

def create_train_dataset(textual_frame,code_frame,data,process_data_path):
    columns = ['id','label','content']
    label_size=textual_frame.shape[0]
    process_data = []
    y=0
    for i in tqdm(data):
        content_list = []
        for j in i["content"]:
            content_list.append([i["id"],label_size,j])
    
        for j in i["evidence"]:
            idx = j["index"]
            value_list = j["value"]
            pro_value_list = []
            for z in value_list:
                code_label = z.split('/')[-1]
                num_label = code_frame[code_frame["C"]==code_label].index.values[0]
                pro_value_list.append(str(num_label))
                
            content_list[idx][1] = '#'.join(pro_value_list)
        

        if not process_data:
            process_data = content_list
        else:
            process_data=process_data+content_list
        
    dataset = pd.DataFrame(process_data,columns=columns)
    dataset.to_csv(process_data_path+'train.csv',index = False)

def create_vectorlabel_dataset(data,process_data_path):
    vectorlabel_list = []    
    for i in data:
        vectorlabel = np.zeros(235,'int64')
        vectorlabel[0]=int(i["id"])
        for j in i["result"]: 
            idx=int(j.split('/')[-1][1:])
            vectorlabel[idx+1] = 1
        vectorlabel_list.append(vectorlabel)

    vec_frame = pd.DataFrame(np.array(vectorlabel_list))
    vec_frame.to_csv(process_data_path+'vec_frame.csv',index = False)

if __name__ == "__main__":

    process_data_path = '../preprocessdata/'
    if not os.path.exists(process_data_path):
        os.mkdir(process_data_path)

    code_tree=json.load(open('../rawdata/code_tree.json', encoding='utf-8'))
    textual_tree = json.load(open('../rawdata/textual_tree.json', encoding='utf-8'))
    data=json.load(open('../rawdata/train.json', encoding='utf-8'))

    
    textual_frame,code_frame,text_code_frame=labeltree_dataset(textual_tree,code_tree,process_data_path)

    # create_train_dataset(textual_frame,code_frame)
    create_vectorlabel_dataset(data,process_data_path)