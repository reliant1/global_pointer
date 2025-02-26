#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2023/5/10 下午4:55
# @Author  : liu yuhan
# @FileName: model.py
# @Software: PyCharm


import torch
import torch.nn as nn
import numpy as np
from transformers import BertModel, BertConfig


class GlobalPointer(nn.Module):
    def __init__(self, hidden_size, num_heads, head_size, device, if_rope):
        super().__init__()
        """
        这里参考原文的实现，使用了两个全连接层，用于构建qi和kj
        在苏神的代码中这两个全连接层被组合了
        参数：
        head: 实体种类
        head_size: 每个index的映射长度
        """
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_size = head_size
        self.device = device
        self.dense = nn.Linear(hidden_size, num_heads * head_size * 2, bias=True)
        # LN
        self.ln = nn.LayerNorm(num_heads * head_size * 2)
        # dropout
        self.dropout = nn.Dropout(0.1)
        if if_rope:
            # PoRE
            self.if_rope = if_rope
            indices = torch.arange(0, head_size // 2, dtype=torch.float)
            indices = torch.pow(torch.tensor(10000, dtype=torch.float), -2 * indices / head_size)
            emb_cos = torch.cos(indices)
            self.emb_cos = torch.repeat_interleave(emb_cos, 2, dim=-1).to(device)
            emb_sin = torch.sin(indices)
            self.emb_sin = torch.repeat_interleave(emb_sin, 2, dim=-1).to(device)
            # [1, 1, 1, 1]->[-1, 1, -1, 1]
            self.trans4tensor = torch.Tensor([-1, 1] * (self.head_size // 2)).to(device)

    def transpose_for_scores(self, x):
        # x:[batch_size, seq_len, head * head_size * 2]
        new_x_shape = x.size()[:-1] + (self.num_heads, self.head_size * 2)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)  # [batch_size, num_heads, seq_len, head_size]

    def scores_mask(self, attention_mask, seq_len):
        # Expand attention_mask for broadcasting
        extended_attention_mask = attention_mask[:, None, None, :]  # [batch_size, 1, 1, seq_len]
        # Create lower triangular mask
        upper_tri_mask = torch.triu(torch.ones((seq_len, seq_len), device=self.device))  # [seq_len, seq_len]
        # Combine masks by multiplying them, since both masks are in a binary form (0 or 1)
        combined_mask = extended_attention_mask * upper_tri_mask  # [batch_size, 1, seq_len, seq_len]
        return combined_mask

    def get_rope(self, tenser):
        return tenser * self.emb_cos + tenser * self.trans4tensor * self.emb_sin

    def forward(self, inputs):
        # inputs: [batch_size, seq_len, hidden_size]
        inputs = self.dense(inputs)  # [batch_size, seq_len, head * head_size * 2]
        # LN
        inputs = self.ln(inputs)
        # dropout
        inputs = self.dropout(inputs)
        # reshape
        inputs = self.transpose_for_scores(inputs)  # [batch_size, head, seq_len, head_size * 2]

        q, v = inputs[:, :, :, :self.head_size], inputs[:, :, :, self.head_size:]
        # PoRE
        if self.if_rope:
            q = self.get_rope(q)
            v = self.get_rope(v)

        # attention_scores
        scores = torch.matmul(q, v.transpose(-1, -2))  # [batch_size, num_heads, seq_len, seq_len]
        scores = scores / np.sqrt(self.head_size)
        return scores
        # # mask
        # return self.scores_mask(scores, attention_mask)


class EfficientGlobalPointer(nn.Module):
    def __init__(self, hidden_size, num_heads, head_size, device, if_rope):
        super().__init__()
        """
        这里参考原文的实现，使用了两个全连接层，用于构建qi和kj
        在苏神的代码中这两个全连接层被组合了
        参数：
        head: 实体种类
        head_size: 每个index的映射长度
        """
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_size = head_size
        self.device = device
        self.dense = nn.Linear(hidden_size, head_size * 2, bias=True)
        self.dense4head = nn.Linear(head_size * 2, num_heads * 2, bias=True)
        self.if_rope = if_rope
        if self.if_rope:
            # PoRE
            indices = torch.arange(0, head_size // 2, dtype=torch.float)
            indices = torch.pow(torch.tensor(10000, dtype=torch.float), -2 * indices / head_size)
            emb_cos = torch.cos(indices)
            self.emb_cos = torch.repeat_interleave(emb_cos, 2, dim=-1).to(device)
            emb_sin = torch.sin(indices)
            self.emb_sin = torch.repeat_interleave(emb_sin, 2, dim=-1).to(device)
            # [1, 1, 1, 1]->[-1, 1, -1, 1]
            self.trans4tensor = torch.Tensor([-1, 1] * (self.head_size // 2)).to(device)

    def scores_mask(self, attention_mask, seq_len):
        # Expand attention_mask for broadcasting
        extended_attention_mask = attention_mask[:, None, None, :]  # [batch_size, 1, 1, seq_len]
        # Create lower triangular mask
        upper_tri_mask = torch.triu(torch.ones((seq_len, seq_len), device=self.device))  # [seq_len, seq_len]
        # Combine masks by multiplying them, since both masks are in a binary form (0 or 1)
        combined_mask = extended_attention_mask * upper_tri_mask  # [batch_size, 1, seq_len, seq_len]
        return combined_mask

    def get_rope(self, tenser):
        return tenser * self.emb_cos + tenser * self.trans4tensor * self.emb_sin

    def forward(self, inputs, attention_mask):
        # inputs: [batch_size, seq_len, hidden_size]
        inputs = self.dense(inputs)  # [batch_size, seq_len, head_size * 2]
        q, v = inputs[:, :, :self.head_size], inputs[:, :, self.head_size:]  # [batch_size, seq_len, head_size]
        # PoRE
        if self.if_rope:
            q = self.get_rope(q)
            v = self.get_rope(v)
        # attention_scores
        scores = torch.matmul(q, v.transpose(-1, -2))  # [batch_size, num_heads, seq_len, seq_len]
        scores = scores / np.sqrt(self.head_size)
        # 以上应该没有问题

        bias = self.dense4head(inputs).permute(0, 2, 1) / 2  # [batch_size, num_heads * 2, seq_len]
        scores = scores[:, None] + bias[:, ::2, None] + bias[:, 1::2, :, None]
        # mask
        return self.scores_mask(scores, attention_mask)


class MyNER(nn.Module):
    def __init__(self, base_model, ner_num_heads, ner_head_size, device, if_rope=False, if_efficientnet=False):
        super().__init__()
        self.config = BertConfig.from_pretrained(base_model)
        self.bert = BertModel.from_pretrained(base_model)
        if if_efficientnet:
            self.ner_score = EfficientGlobalPointer(hidden_size=self.config.hidden_size,
                                                    num_heads=ner_num_heads,
                                                    head_size=ner_head_size,
                                                    device=device,
                                                    if_rope=if_rope)

        else:
            self.ner_score = GlobalPointer(hidden_size=self.config.hidden_size,
                                           num_heads=ner_num_heads,
                                           head_size=ner_head_size,
                                           device=device,
                                           if_rope=if_rope)

    def forward(self, input, attention_mask):
        hidden = self.bert(input, attention_mask=attention_mask).last_hidden_state
        scores = self.ner_score(hidden)
        return scores

    def fun_loss(self, score, label, attention_mask):
        # score: [batch_size, num_heads, seq_len, seq_len]
        # label: [batch_size, num_heads, seq_len, seq_len]
        # [batch_size, num_heads, seq_len, seq_len] -> [batch_size * num_heads, seq_len, seq_len]
        batch_size, num_heads, seq_len, _ = score.size()
        mask = self.ner_score.scores_mask(attention_mask, seq_len)
        mask = mask.repeat(1, num_heads, 1, 1)
        score = score.view(batch_size * num_heads, seq_len * seq_len)
        label = label.view(batch_size * num_heads, seq_len * seq_len)
        mask = mask.view(batch_size * num_heads, seq_len * seq_len)

        score = (1 - 2 * label) * score
        score_neg = score - label * 1e12
        score_pos = score - (1 - label) * 1e12
        # add mask
        score_neg = torch.where(mask == 0, -1e12, score_neg)
        score_pos = torch.where(mask == 0, -1e12, score_pos)
        # loss
        zeros = torch.zeros_like(score[..., :1])
        score_neg = torch.cat([score_neg, zeros], dim=-1)
        score_pos = torch.cat([score_pos, zeros], dim=-1)
        neg_loss = torch.logsumexp(score_neg, dim=-1)
        pos_loss = torch.logsumexp(score_pos, dim=-1)
        return torch.mean(neg_loss + pos_loss)

    def fun_loss2(self, score, label):
        """
        这是另外一种loss的计算方式
        :param score:
        :param label:
        :return:
        """
        batch_size, num_heads, seq_len, _ = score.size()
        score = score.view(batch_size * num_heads, seq_len * seq_len)
        label = label.view(batch_size * num_heads, seq_len * seq_len)
        pos_mask = torch.eq(label, 1)
        score_pos = torch.where(pos_mask, -score, -1e12)
        neg_mask = torch.eq(label, 0)
        score_neg = torch.where(neg_mask, score, -1e12)
        zeros = torch.zeros_like(score[..., :1])
        score_neg = torch.cat([score_neg, zeros], dim=-1)
        score_pos = torch.cat([score_pos, zeros], dim=-1)
        neg_loss = torch.logsumexp(score_neg, dim=-1)
        pos_loss = torch.logsumexp(score_pos, dim=-1)
        return torch.mean(neg_loss + pos_loss)

    def evaluate(self, score, labels, attention_mask):
        # score: [batch_size, num_heads, seq_len, seq_len]
        # label: [batch_size, num_heads, seq_len, seq_len]
        batch_size, num_heads, seq_len, _ = score.size()
        mask = self.ner_score.scores_mask(attention_mask, seq_len)
        mask = mask.repeat(1, num_heads, 1, 1)

        score = torch.where(score > 0, 1, 0)
        score = score * mask
        f1 = 2 * torch.sum(score * labels) / torch.sum(score + labels)
        return f1
