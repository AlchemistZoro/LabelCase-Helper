import torch
from transformers import BertTokenizer, AutoModel
import torch.nn as nn
import json
import time


# 模型分类器
class CaseClassification(nn.Module):
    def __init__(self, class_num, model_path):
        super(CaseClassification, self).__init__()
        self.bert = AutoModel.from_pretrained(model_path)
        self.linear = nn.Linear(in_features=768, out_features=class_num)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, label=None):
        outputs = self.bert(input_ids, attention_mask, token_type_ids)
        pooler_output = outputs['pooler_output']

        logits = self.linear(pooler_output)

        return logits


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


class_num = 234
model_name = 'bert-base-chinese'
model_saved = '../model/model_name.pth'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = CaseClassification(class_num=class_num, model_path=model_name).to(device)
state_dict = torch.load(model_saved)
model.load_state_dict(state_dict['model'])
level3labels = get_level3labels(json.load(open('../../../Sites/LCH/backend/input/textual_tree.json', encoding="utf-8")))
tokenizer = BertTokenizer.from_pretrained(model_name)


def refer(content):
    res = {}
    for k, fact in enumerate(content, 0):
        inputs = tokenizer(fact, max_length=512, padding=True, truncation=True, return_tensors='pt')
        input_ids = inputs['input_ids'].to(device)
        token_type_ids = inputs['token_type_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)

        logits = model(input_ids, attention_mask, token_type_ids, torch.zeros(class_num).to(device))

        threshold = 0.5
        probs = logits[0]
        logits = logits[0]

        logits[logits > threshold] = 1
        logits[logits <= threshold] = 0

        for p in range(len(logits)):
            if logits[p] == 1:
                if not res.get(level3labels[p]):
                    res[level3labels[p]] = [k]
                else:
                    res[level3labels[p]].append(k)

    return res


if __name__ == '__main__':
    a = time.time()
    content = json.load(open("../../../Sites/LCH/backend/input/train.json", encoding='utf-8'))[0]["content"]
    for i in refer(content):
        print(i)
