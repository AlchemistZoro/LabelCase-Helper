import torch
import json
from tqdm import tqdm
from transformers import BertTokenizer
from model import CaseClassification
from utils import get_fact, get_level3labels
import argparse


parser = argparse.ArgumentParser(description='Evaluate for aqbq.')
parser.add_argument('--input_dir', default='./input/test.json')
parser.add_argument('--output_dir', default='./output/answer.json')
args = parser.parse_args()


if __name__ == "__main__":
    input_path = args.input_dir
    output_path = args.output_dir

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = CaseClassification(class_num=252).to(device)
    model.load_state_dict(torch.load('./saved/model10.pth', map_location=device))

    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    level3labels = get_level3labels(json.load(open('./data/tree.json')))

    records = json.load(open(input_path, encoding='utf-8'))
    results = []

    for record in tqdm(records):
        fact = get_fact(record['content'])
        inputs = tokenizer(fact, max_length=512, padding=True, truncation=True, return_tensors='pt')

        # move data to device
        input_ids = inputs['input_ids'].to(device)
        token_type_ids = inputs['token_type_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)

        logits = model(input_ids, attention_mask, token_type_ids)
        prediction = torch.nonzero((torch.sigmoid(logits) > 0.15).view(-1)).view(-1)

        result = []
        for p in prediction:
            result.append(level3labels[p])

        results.append(result)

    json.dump(results, open(output_path, "w", encoding="utf8"), indent=2, ensure_ascii=False)
