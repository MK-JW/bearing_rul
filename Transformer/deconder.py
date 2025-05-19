__author__ = 'minjinwu'

import math
import torch
import torch.nn as nn
from embedding import Embedding
from multi_head_selfattention import MultiHeadAttention
from feed_forward import FeedForward


# 解码器实现


    ## 解码器层
class DecoderLayer(nn.Module):

    def __init__(self, d_model, num_heads, d_ff, dropout=0.15):
        super(DecoderLayer, self).__init__()

        # 自注意力（带mask）
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)

        # 编码器-解码器注意力
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.norm2 = nn.LayerNorm(d_model)

        # 前馈层
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)


    def forward(self, x, enc_output, tgt_mask=True, memory_mask=None):
        # 自注意力（目标语言）
        _x = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(_x))

        # 编码器-解码器注意力（源-目标交互）
        _x = self.cross_attn(x, enc_output, enc_output, memory_mask)
        x = self.norm2(x + self.dropout(_x))

        # 前馈网络
        _x = self.feed_forward(x)
        x = self.norm3(x + self.dropout(_x))

        return x


    ## 解码器
class Decoder(nn.Module):


    def __init__(self, vocab_size, embedding_dim, num_layers, num_heads, d_ff, max_len=512, dropout=0.1):
        super(Decoder, self).__init__()
        self.embedding = Embedding(vocab_size, embedding_dim, max_len)
        self.layers = nn.ModuleList([
            DecoderLayer(embedding_dim, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.fc_out = nn.Linear(embedding_dim, vocab_size)  # 输出层
        self.dropout = nn.Dropout(dropout)


    def forward(self, x, enc_output, tgt_mask=True, memory_mask=None):
        x = self.embedding(x)  # 词嵌入 + 位置编码
        for layer in self.layers:
            x = layer(x, enc_output, tgt_mask, memory_mask)  # 通过每一层解码器
        x = self.fc_out(x)  # 输出层生成词汇概率分布
        return x

