#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2023/5/16 下午8:13
# @Author  : liu yuhan
# @FileName: predict.py
# @Software: PyCharm
import json

from model import MyNER
from utils import *
from parser import parameter_parser
from collections import Counter


class DataProcessPredict:
    def __init__(self, max_len=128):
        self.tokenizer = BertTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext')
        self.categories = set()
        self.max_len = max_len

    def load_data(self, filename):
        D = []
        with open(filename, encoding='utf-8') as f:
            for l in f:
                s_list = l.split('。')
                D.extend(s_list)
        return D

    def get_token(self, text, do_lower_case=True):
        token_ = []
        for c in text:
            if do_lower_case:
                c = c.lower()
            if c in self.tokenizer.vocab:
                token_.append(self.tokenizer.convert_tokens_to_ids(c))
            else:
                token_.append(self.tokenizer.unk_token_id)
        return token_

    def data_process(self, data):
        token_ids = []
        attention_ids = []
        for text in tqdm(data, desc='tokenizing'):
            if len(text) > self.max_len:
                text = text[:self.max_len]
            token = self.get_token(text, do_lower_case=self.tokenizer.do_lower_case)
            attention_mask = [1] * len(token)
            token_ids.append(token)
            attention_ids.append(attention_mask)
        # padding
        for i in tqdm(range(len(token_ids)), desc='padding'):
            pad_length = self.max_len - len(token_ids[i])
            token_ids[i] = token_ids[i] + [self.tokenizer.pad_token_id] * pad_length
            attention_ids[i] = attention_ids[i] + [0] * pad_length
        return token_ids, attention_ids


def get_word(token_ids, i, j, tokenizer):
    entity = token_ids[i:j + 1]
    entity = tokenizer.decode(entity)
    return ''.join(entity.split())


def predict(args):
    """
    这里在predict里面加一个file_inf参数，方便文件的检索
    """
    device = gpu_setup(args)
    # data
    max_len = 128
    data_process = DataProcessPredict(max_len=max_len)
    data = data_process.load_data(args.env_path + 'predict/input_{}.txt'.format(args.task_inf))
    token_ids, attention_ids = data_process.data_process(data)
    with open(args.env_path + 'data/categories.json', 'r', encoding='utf-8') as f:
        categories = json.load(f)
    # key: value->category: index->index: category
    categories = {v: k for k, v in categories.items()}

    # model
    print('model')
    ner_head_size = 64  # 这是另外一个关键参数
    model = MyNER(ner_num_heads=len(categories),
                  ner_head_size=ner_head_size,
                  device=device,
                  if_rope=True,
                  if_efficientnet=False
                  )
    model_state_dict = torch.load(args.env_path + 'model/GP_20230803-171814_2.pt')
    model.load_state_dict(model_state_dict, strict=False)
    model.to(device)

    entity_dict = dict()

    # predict
    for token, attention_mask in tqdm(zip(token_ids, attention_ids), total=len(token_ids)):
        token = torch.tensor([token]).to(device)
        attention_mask = torch.tensor([attention_mask]).to(device)
        score = model(token, attention_mask)[0]
        score = torch.where(score > 0, 1, 0)
        score = score.cpu().numpy()

        print('score', score.shape)
        for c in range(len(score)):
            for i in range(0, max_len - 1):
                for j in range(max_len):
                    if score[c][i][j] == 1:
                        entity = get_word(token[0], i, j, data_process.tokenizer)
                        if categories[c] not in entity_dict:
                            entity_dict[categories[c]] = [entity]
                        else:
                            entity_dict[categories[c]].append(entity)

    for k, e_list in entity_dict.items():
        e_list = [e for e in e_list if len(e) < 10]
        entity_dict[k] = list(set(e_list))

    test2entity_list = []
    for d in data:
        test2entity = dict()
        test2entity['text'] = d
        for c, e_list in entity_dict.items():
            position = []
            for e in e_list:
                if e in d:
                    start = d.find(e)
                    for i in range(d.count(e)):
                        position.append([start, start + len(e)])
                        start = d.find(e, start + 1)
            test2entity[c] = position
        test2entity_list.append(test2entity)

    # save
    with open(args.env_path + 'predict/output_{}.json'.format(args.task_inf), 'w', encoding='utf-8') as f:
        json.dump(test2entity_list, f, ensure_ascii=False)


if __name__ == '__main__':
    args = parameter_parser()
    predict(args)
