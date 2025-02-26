#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2023/5/10 下午4:37
# @Author  : liu yuhan
# @FileName: ner_main.py
# @Software: PyCharm

# 中文的命名实体识别
from utils import *
from model import MyNER
from parser import parameter_parser

import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler

from tqdm import tqdm
import numpy as np
import time


def train(args):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    base_model = 'hfl/chinese-roberta-wwm-ext-large'
    print(device)
    # data process
    print('data process')
    get_categories('../data/trainData.json')
    data_process = DataProcess(max_len=128, base_model=base_model)
    ner_num_heads = len(data_process.categories)
    train_text_list, train_label_list = data_process.load_data('../data/trainData.json')
    dev_text_list, dev_label_list = data_process.load_data('../data/devData.json')
    # dataloader
    print('dataloader')
    print('  train data')
    train_dataloader = data_process.make_dataloader(train_text_list, train_label_list, batch_size=args.batch_size)
    print('  dev data')
    dev_dataloader = data_process.make_dataloader(dev_text_list, dev_label_list, batch_size=args.batch_size)

    # model
    print('model')
    ner_head_size = 128  # 这是另外一个关键参数
    model = MyNER(base_model=base_model,
                  ner_num_heads=ner_num_heads,
                  ner_head_size=ner_head_size,
                  device=device,
                  if_rope=True,
                  if_efficientnet=False
                  )
    # model_state_dict = torch.load('../model/rope_model.pt')
    # model.load_state_dict(model_state_dict)
    model.to(device)

    # optimizer
    print('optimizer')
    optimizer = optim.Adam(model.parameters(), lr=args.init_lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max',
                                                     factor=args.lr_reduce_factor,
                                                     patience=args.lr_schedule_patience,
                                                     min_lr=args.lr_min,
                                                     verbose=True)
    print('model initialized')
    print('model parameters:', sum(p.numel() for p in model.parameters() if p.requires_grad))
    # train
    print('training...')
    print('batch_size:', args.batch_size, 'epochs:', args.epochs)
    scaler = GradScaler()

    best_f1 = 0

    for epoch in range(args.epochs):
        # train
        model.train()
        loss_collect = []
        with tqdm(total=len(train_dataloader), desc='train---epoch:{}'.format(epoch)) as bar:
            for step, (token_ids, attention_mask, labels) in enumerate(train_dataloader):
                token_ids = token_ids.to(device)
                attention_mask = attention_mask.to(device)
                labels = labels.to(device)

                with autocast():
                    score = model(token_ids, attention_mask)
                    loss = model.fun_loss(score, labels, attention_mask)
                    # loss = model.fun_loss2(score, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                loss_collect.append(loss.item())
                scheduler.step(loss.item())
                bar.update(1)
                bar.set_postfix(loss=np.mean(loss_collect).item(), lr=optimizer.param_groups[0]['lr'])

        # dev
        model.eval()
        f1_collect = []
        with tqdm(total=len(dev_dataloader), desc='dev') as bar:
            for step, (token_ids, attention_mask, labels) in enumerate(dev_dataloader):
                token_ids = token_ids.to(device)
                attention_mask = attention_mask.to(device)
                labels = labels.to(device)
                with torch.no_grad():
                    score = model(token_ids, attention_mask)
                    f1 = model.evaluate(score, labels, attention_mask)
                    f1_collect.append(f1.item())
                bar.update(1)
                bar.set_postfix(f1=np.mean(f1_collect))
        if np.mean(f1_collect) > best_f1:
            best_f1 = np.mean(f1_collect)
            local_time = time.strftime("%Y%m%d-%H%M%S", time.localtime())
            model_path = '../model/GP_{}_{}.pt'.format(local_time, epoch)
            torch.save(model.state_dict(), model_path)


if __name__ == '__main__':
    args = parameter_parser()
    train(args)
