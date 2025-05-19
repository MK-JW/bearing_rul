__author__ = 'minjinwu'

import math
import torch
import torch.nn as nn
import torch.functional as F
from embedding import Embedding
from multi_head_selfattention import MultiHeadAttention
from feed_forward import FeedForward


# 编码器类实现

## 编码器层
class EncoderLayer(nn.Module):


    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        """
        一个 Transformer 编码器层，包含多头注意力和前馈神经网络。
        
        参数:
        - d_model: 词向量的维度
        - num_heads: 多头注意力的头数
        - d_ff: 前馈网络的隐藏层维度
        - dropout: dropout 率
        """
        super(EncoderLayer, self).__init__()

        self.multihead_attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.layer_norm1 = nn.LayerNorm(d_model)  # 用于多头注意力后的层归一化
        self.layer_norm2 = nn.LayerNorm(d_model)  # 用于前馈网络后的层归一化
        self.dropout = nn.Dropout(dropout)
        

    def forward(self, x, mask=None):

        """
        前向传播：将输入经过多头注意力和前馈网络。
        
        输入:
        - x: (batch_size, seq_len, d_model)
        - mask: (batch_size, 1, 1, seq_len)，可选
        
        输出:
        - 编码器层的输出 (batch_size, seq_len, d_model)
        """
        
        # 多头注意力层
        attn_output = self.multihead_attention(x, x, x, mask) # 这里面x,x,x代表自注意力机制，即QKV都是通过x线性变换得到

        x = self.layer_norm1(attn_output + x)  # 残差连接 + 层归一化

        # 前馈网络
        ff_output = self.feed_forward(x)
        
        x = self.layer_norm2(ff_output + x)  # 残差连接 + 层归一化

        return x
    


## 多层编码器叠加
class Encoder(nn.Module):

    
    def __init__(self, vocab_size, embedding_dim, num_layers, num_heads, d_ff, max_len=512, dropout=0.1):
        """
        Transformer 编码器。
        
        参数:
        - vocab_size: 词汇表大小
        - embedding_dim: 词嵌入维度
        - num_layers: 编码器的层数（有多少个编码器堆叠）
        - num_heads: 多头注意力的头数
        - d_ff: 前馈网络的隐藏层维度
        - max_len: 句子的最大长度
        - dropout: dropout 率
        """
        super(Encoder, self).__init__()

        # 词嵌入和位置编码
        self.embedding = Embedding(vocab_size, embedding_dim, max_len)

        # 堆叠多个编码器层
        self.layers = nn.ModuleList([
            EncoderLayer(embedding_dim, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])


    def forward(self, x, mask=None):
        """
        前向传播：将输入通过多个编码器层。
        
        输入:
        - x: (batch_size, seq_len) 词索引
        - mask: (batch_size, 1, 1, seq_len)，可选
        
        输出:
        - 编码器的输出 (batch_size, seq_len, d_model)
        """
        x = self.embedding(x)  # 词嵌入 + 位置编码
        
        for layer in self.layers:
            x = layer(x, mask)  # 通过每一层编码器
        
        return x