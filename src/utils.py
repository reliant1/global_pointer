#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2023/5/12 上午10:13
# @Author  : liu yuhan
# @FileName: utils.py
# @Software: PyCharm
import json
import os

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import BertTokenizer


def gpu_setup(args):
    """
    set gpu
    """
    use_gpu, gpu_id = args.use_gpu, args.gpu_id
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    if torch.cuda.is_available() and use_gpu:
        print('cuda available with GPU:', torch.cuda.get_device_name(0))
        device = torch.device("cuda")
    else:
        print('cuda not available')
        device = torch.device("cpu")
    return device


def get_categories(filename):
    category2index = dict()

    with open(filename, encoding='utf-8') as f:
        for l in f:
            l = json.loads(l)
            for k in l['label']:
                if k not in category2index:
                    category2index[k] = len(category2index)
    # sorted
    category2index = dict(sorted(category2index.items(), key=lambda x: x[1]))
    # # save
    with open('../data/categories.json', 'w', encoding='utf-8') as f:
        json.dump(category2index, f)


class DataProcess:
    def __init__(self, max_len=256, base_model='hfl/chinese-roberta-wwm-ext'):
        self.tokenizer = BertTokenizer.from_pretrained(base_model)
        self.max_len = max_len
        # load category2index
        with open('../data/categories.json', encoding='utf-8') as f:
            self.categories = json.load(f)

    def load_data(self, filename):
        """加载数据
        单条格式：[text, (start, end, label), (start, end, label), ...]，
                  意味着text[start:end + 1]是类型为label的实体。
        """
        text_list = []
        label_list = []
        with open(filename, encoding='utf-8') as f:
            for l in f:
                l = json.loads(l)
                label_list_temp = []
                for k, v in l['label'].items():
                    for spans in v.values():
                        for start, end in spans:
                            label_list_temp.append((start, end, self.categories[k]))
                text_list.append(l['text'])
                label_list.append(label_list_temp)
        return text_list, label_list

    def data_process(self, text_list, label_list):
        token_ids = []
        attention_ids = []
        label_trans_list = []
        for text, label in tqdm(zip(text_list, label_list), desc='tokenizing', total=len(text_list)):
            # Tokenize sentences
            encoded_input = self.tokenizer(text, truncation=True, max_length=self.max_len)
            token = encoded_input['input_ids']
            attention_mask = encoded_input['attention_mask']

            label_matrix = torch.zeros((len(self.categories), self.max_len, self.max_len))
            for start, end, l in label:
                word = text[start:end + 1]
                word_input_ids = self.tokenizer.encode(word, add_special_tokens=False)
                word_length = len(word_input_ids)
                for i in range(self.max_len - word_length):
                    if token[i:i + word_length] == word_input_ids:
                        start, end = i, i + word_length - 1
                        label_matrix[l, start, end] = 1
            token_ids.append(token)
            attention_ids.append(attention_mask)
            label_trans_list.append(label_matrix)
        # padding to max_len
        token_ids = [token + [self.tokenizer.pad_token_id] * (self.max_len - len(token))
                     for token in token_ids]
        attention_ids = [attention_mask + [0] * (self.max_len - len(attention_mask))
                         for attention_mask in attention_ids]

        return token_ids, attention_ids, label_trans_list

    def make_dataloader(self, text_list, label_list, batch_size):
        token_ids, attention_ids, label_list = self.data_process(text_list, label_list)
        dataset = MyDataSet(token_ids, attention_ids, label_list)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        return dataloader


class MyDataSet(Dataset):
    """
    data load
    """

    def __init__(self, token_ids_list, attention_mask_list, label_list):
        self.sequence_list = torch.tensor(token_ids_list, dtype=torch.long)
        self.attention_mask_list = torch.tensor(attention_mask_list, dtype=torch.long)
        self.label_list = label_list

    def __len__(self):
        return len(self.sequence_list)

    def __getitem__(self, idx):
        return self.sequence_list[idx], self.attention_mask_list[idx], self.label_list[idx]
