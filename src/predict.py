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
from collections import Counter, defaultdict
import torch
import numpy as np


def data_process():
    with open('../../data_0617/processed_file/text4ner.txt', 'r', encoding='utf-8') as f:
        text_list = f.readlines()
    # strip
    text_list = [t.strip() for t in text_list]
    text_list = [''.join(t.split()) for t in text_list]

    # tokenizer
    tokenizer = BertTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext-large')
    batch_size = 32
    max_len = 128

    token_ids_list = []
    attention_mask_list = []

    for i in tqdm(range(0, len(text_list), batch_size)):
        text_batch = text_list[i:i + batch_size]
        outputs = tokenizer(text_batch, truncation=True, max_length=max_len, padding='max_length')
        token_ids_list.extend(outputs['input_ids'])
        attention_mask_list.extend(outputs['attention_mask'])

    # save token_ids_list and attention_mask_list
    with open('../../data_0617/processed_file/token_ids_list4ner.json', 'w', encoding='utf-8') as f:
        json.dump(token_ids_list, f)
    with open('../../data_0617/processed_file/attention_mask_list4ner.json', 'w', encoding='utf-8') as f:
        json.dump(attention_mask_list, f)


def get_word(token_ids, i, j, tokenizer):
    entity = token_ids[i:j + 1]
    entity = tokenizer.decode(entity)
    return ''.join(entity.split())


def predict():
    """
    这里在predict里面加一个file_inf参数，方便文件的检索
    """
    # device
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    # data
    with open('../../data_0617/processed_file/text_id_list4ner.json', 'r', encoding='utf-8') as f:
        text_id_list = json.load(f)
    with open('../../data_0617/processed_file/token_ids_list4ner.json', 'r', encoding='utf-8') as f:
        token_ids_list = json.load(f)
    with open('../../data_0617/processed_file/attention_mask_list4ner.json', 'r', encoding='utf-8') as f:
        attention_mask_list = json.load(f)
    with open('../data/categories.json', 'r', encoding='utf-8') as f:
        categories = json.load(f)
    # key: value->category: index->index: category
    categories = {v: k for k, v in categories.items()}

    # model
    print('model')
    ner_head_size = 128  # 这是另外一个关键参数
    base_model = 'hfl/chinese-roberta-wwm-ext-large'

    model = MyNER(base_model=base_model,
                  ner_num_heads=len(categories),
                  ner_head_size=ner_head_size,
                  device=device,
                  if_rope=True,
                  if_efficientnet=False
                  )
    model_state_dict = torch.load('../model/GP_20240716-121846_8.pt')
    model.load_state_dict(model_state_dict, strict=False)
    model.to(device)
    # tokenizer
    tokenizer = BertTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext-large')

    text_id2categories2word2freq = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

    # predict
    batch_size = 48

    for i in tqdm(range(0, len(token_ids_list), batch_size)):
        token_ids_batch = token_ids_list[i:i + batch_size]
        attention_mask_batch = attention_mask_list[i:i + batch_size]
        token_ids_batch = torch.tensor(token_ids_batch).to(device)
        attention_mask_batch = torch.tensor(attention_mask_batch).to(device)
        score = model(token_ids_batch, attention_mask_batch)
        score = torch.where(score > 0, 1, 0)
        score = score.cpu().numpy()

        for text_id, token_ids, s in zip(text_id_list[i:i + batch_size], token_ids_list[i:i + batch_size], score):
            for c in range(len(s)):
                ones_indices = np.where(s[c] == 1)
                end_index = token_ids.index(102)
                for m, k in zip(ones_indices[0], ones_indices[1]):
                    if k > end_index:
                        continue
                    if m <= k:
                        word = get_word(token_ids, m, k, tokenizer)
                        text_id2categories2word2freq[text_id][categories[c]][word] += 1

    # save
    with open('../../data_0617/processed_file/text_id2categories2word2freq4ner_add_action.json', 'w', encoding='utf-8') as f:
        json.dump(text_id2categories2word2freq, f, ensure_ascii=False)


if __name__ == '__main__':
    # data_process()
    predict()
