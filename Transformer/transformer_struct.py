__author__ = 'minjinwu'

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import pandas as pd 

# print(torch.__version__)              # 应该输出 2.x.x
# print(torch.cuda.is_available())      # True
# print(torch.cuda.get_device_name(0))  # NVIDIA GeForce RTX 4060 Laptop

# 编码器类实现


    ## 词嵌入与位置编码

"对于多维时间序列的输入情况，这里面是可以不需要进行词嵌入的"
"可以对输入的原始的数据(batch_size,sql_len,features_num)进行线性变换nn.linear(features_num, model_dim)"
"然后再进行位置编码positional_encoding"

"下面的代码是需要对词表进行词嵌入与位置编码的,这里面的词嵌入与上面的线性变换很像，就像是提取特征一样"


class Embedding(nn.Module):


    def __init__(self, features_num, embedding_dim, max_len=512, is_enconder = True):
        """
        词嵌入类，支持共享词嵌入和位置编码。
        
        参数:
        - features_num: 特征维度大小   
        - embedding_dim: 维度转换
        - max_len: 句子最大长度（用于位置编码） 一次识别最多多少个词,少会padding,多会忽略
        - shared_weight: 可选，是否共享已有的嵌入层 (nn.Embedding)
        """
        super(Embedding, self).__init__()

        # 如果传入 shared_weight，则使用共享的嵌入层（就是encod和decoding进行权重共享）
        # if shared_weight is not None:
        #     self.embedding = shared_weight  # 共享权重
        # else:
        #     self.embedding = nn.Embedding(vocab_size, embedding_dim)
        if is_enconder is True:
            self.embedding = nn.Linear(features_num, embedding_dim)
        else:
            self.embedding = nn.Linear(1, embedding_dim)


        # 位置编码
        self.positional_encoding = self.create_positional_encoding(max_len, embedding_dim)


    def create_positional_encoding(self, max_len, embedding_dim):
        """
        生成位置编码，采用 Transformer 的正弦余弦位置编码方法。
        """
        pos_enc = torch.zeros(max_len, embedding_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-math.log(10000.0) / embedding_dim))

        pos_enc[:, 0::2] = torch.sin(position * div_term)
        pos_enc[:, 1::2] = torch.cos(position * div_term)

        return pos_enc.unsqueeze(0)  # (1, max_len, embedding_dim) 方便 batch 处理


    def forward(self, input_ids):
        """
        前向传播：
        - 词索引 -> 词嵌入
        - 词嵌入 + 位置编码
        输入:
        - input_ids: (batch_size, seq_len) 形状的张量，表示词索引
        这里面的seq_len是你数据实际输入时候的长度,它小于等于max_len
        
        输出:
        - 嵌入后的张量，形状为 (batch_size, seq_len, embedding_dim)
        """
        embedded = self.embedding(input_ids)  # 词嵌入
        seq_len = input_ids.size(1)
        positional_enc = self.positional_encoding[:, :seq_len, :].to(embedded.device)  # 取前 seq_len 个位置编码

        return embedded + positional_enc  # 词嵌入 + 位置编码
    



#     ## 多头注意力机制
# class MultiHeadAttention(nn.Module):

#     def __init__(self, d_model, num_heads, dropout=0.1):
#         super(MultiHeadAttention, self).__init__()
#         assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"

#         self.d_model = d_model
#         self.num_heads = num_heads
#         self.depth = d_model // num_heads  # 每个头的维度

#         # Q, K, V 的线性层
#         self.W_q = nn.Linear(d_model, d_model)
#         self.W_k = nn.Linear(d_model, d_model)
#         self.W_v = nn.Linear(d_model, d_model)

#         # 输出线性层
#         self.W_o = nn.Linear(d_model, d_model)

#         # 层归一化
#         self.layer_norm = nn.LayerNorm(d_model)
#         self.dropout = nn.Dropout(dropout)


#     def split_heads(self, x, batch_size):
#         # 将最后一维 d_model 拆成 num_heads * depth，并调整维度为 [batch, heads, seq_len, depth]
#         x = x.view(batch_size, -1, self.num_heads, self.depth)
#         return x.permute(0, 2, 1, 3)  # [batch, heads, seq_len, depth]


#     def scaled_dot_product_attention(self, Q, K, V, mask=None):
#         # Q, K, V: [batch, heads, seq_len, depth]
#         #attention_output = softmax(Q*K/sqrt(d_k))*V
#         d_k = Q.size(-1)
#         scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32, device=Q.device))

#         if mask is not None:
#             scores = scores.masked_fill(mask == 0, -1e9)
#             # print("score.shape:",scores.shape)

#         attn_weights = F.softmax(scores, dim=-1)
#         attn_weights = self.dropout(attn_weights)
#         output = torch.matmul(attn_weights, V)
#         return output


#     def forward(self, Q, K=None, V=None, mask=None):
#         """
#         通用前向传播接口，支持：
#         - 自注意力：仅传入 Q (即x)
#         - 编码器-解码器交叉注意力：传入 Q, K, V

#         Q, K, V: [batch_size, seq_len, d_model]
#         mask: [batch_size, 1, 1, seq_len] 或其他可广播形状
#         """
#         # 如果没有传入 K, V，说明是自注意力，Q=K=V
#         if K is None:
#             K = Q
#         if V is None:
#             V = Q

#         batch_size = Q.size(0)
#         residual = Q  # 残差连接，这里面是保存了线性变换之前的Q

#         # 线性变换
#         Q = self.W_q(Q)
#         K = self.W_k(K)
#         V = self.W_v(V)

#         # 拆分多头
#         "Q, K, V: [batch_size, heads, seq_len, depth]"
#         "后续计算需要将多个头的维度进行合并"

#         Q = self.split_heads(Q, batch_size)
#         K = self.split_heads(K, batch_size)
#         V = self.split_heads(V, batch_size)

#         # 计算注意力
#         attn_output = self.scaled_dot_product_attention(Q, K, V, mask)

#         # 合并多头输出
#         attn_output = attn_output.permute(0, 2, 1, 3).contiguous()  # 调整维度为[batch_size, seq_len, num_heads, depth]，并转换为连续张量
#         # print("attn_output_multiheads:", attn_output.shape)
#         # print("attn_output_multiheads:", attn_output[1, :, :, :])
#         attn_output = attn_output.view(batch_size, -1, self.d_model)
#         # print("attn_output_mergedheads:", attn_output.shape)
#         # print("attn_output_mergedheads:", attn_output[1, :, :])

#         # 输出线性层
#         output = self.W_o(attn_output)

#         # 残差连接 + 层归一化（注意 residual 是原始 Q）
#         output = self.layer_norm(output + residual)

#         return output





    ## 多头注意力机制（固定策略分配）
class MultiHeadAttention(nn.Module):


    def __init__(self, d_model, num_heads, strategy= 'equal', dropout=0.1):
        super(MultiHeadAttention, self).__init__() 
        self.d_model = d_model
        self.num_heads = num_heads
        self.strategy = strategy

        # 计算每个头的维度
        self.head_dims = self._get_head_dims(d_model, num_heads, strategy)
        assert sum(self.head_dims) == d_model, f"Head dims sum {sum(self.head_dims)} must equal d_model {d_model}"

        # 为每个头分别初始化 Q, K, V 的线性层
        self.q_linears = nn.ModuleList([
            nn.Linear(d_model, dim) for dim in self.head_dims
        ])
        self.k_linears = nn.ModuleList([
            nn.Linear(d_model, dim) for dim in self.head_dims
        ])
        self.v_linears = nn.ModuleList([
            nn.Linear(d_model, dim) for dim in self.head_dims
        ])
        # print(self.q_linears)

        # 合并后的输出映射层
        self.linear_out = nn.Linear(d_model, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def _get_head_dims(self, d_model, num_heads, strategy):
        if strategy == 'equal':
            return [d_model // num_heads] * num_heads
        elif strategy == 'arithmetic':
            # 等差：例如 1, 2, ..., n，再归一化到 d_model
            base = torch.arange(1, num_heads + 1, dtype=torch.float)
        elif strategy == 'geometric':
            # 等比：例如 1, 2, 4, 8...
            base = torch.tensor([2 ** i for i in range(num_heads)], dtype=torch.float)
        elif strategy == 'fibonacci':
            base = torch.tensor(self._fibonacci_seq(num_heads), dtype=torch.float)
        else:
            raise ValueError(f"Unknown strategy {strategy}")


        # 按比例分配到 d_model 上
        base = base / base.sum() * d_model
        base = base.round().int()

        # 调整总和误差（补偿/削减）
        while base.sum() < d_model:
            base[base.argmin()] += 1
        while base.sum() > d_model:
            base[base.argmax()] -= 1

        return base.tolist()


    def _fibonacci_seq(self, n):
        seq = [1, 1]
        while len(seq) < n:
            seq.append(seq[-1] + seq[-2])
        return seq[:n]


    def forward(self, query, key=None, value=None, mask=None, need_weights=False):
        # print("query_shape:", query.shape)

        if key == None:
            key = query
        if value == None:
            value = query

        residual = query # 残差连接，这里面是保存了线性变换之前的Q（自注意力机制）

        batch_size, seq_len, _ = query.size()
        attn_outputs = []
        attn_weights = []

        # 预处理 mask（只做一次）
        use_mask = mask is not None
        if use_mask:
            if mask.dim() == 4:
                mask = mask.squeeze(1).squeeze(1)
            elif mask.dim() == 3:
                pass
            else:
                raise ValueError(f"Unsupported mask shape: {mask.shape}")


        for i in range(self.num_heads):
            q = self.q_linears[i](query)  # [batch, seq_len, head_dim]
            k = self.k_linears[i](key)
            v = self.v_linears[i](value)
            # print("q shape:", q.shape)
            # print("k shape:", k.shape)
            # print("v shape:", v.shape)
            # 缩放点积注意力
            scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dims[i] ** 0.5)
            # print("score_shape:", scores.shape)

            if use_mask:
                if mask.dim() == 2:
                    scores = scores.masked_fill(mask.unsqueeze(1) == 0, float('-inf'))
                elif mask.dim() == 3:
                    scores = scores.masked_fill(mask == 0, float('-inf'))


            attn = F.softmax(scores, dim=-1)
            attn = self.dropout(attn)
            output = torch.matmul(attn, v)  # [batch, seq_len, head_dim]
            attn_weights.append(attn)
            attn_outputs.append(output)
            # print(f"Head {i}:")
            # print("attn shape:", attn.shape)
            # print("output shape:", output.shape)


        # print("attn_outputs[0]_shape:", attn_outputs[0].shape)
        # 拼接所有头的输出
        concat = torch.cat(attn_outputs, dim=-1)  # [batch, seq_len, d_model]
        # print(concat.shape)
        output = self.linear_out(concat)
        output = self.layer_norm(output + residual)
        # print("output_shape:", output.shape)

        if need_weights:
            return output, attn_weights
        else:
            return output





#     ## 多头注意力机制（可学习动态维度分配）
# class MultiHeadAttention(nn.Module):


#     def __init__(self, d_model, num_heads):
#         super(MultiHeadAttention, self).__init__()
#         self.d_model = d_model
#         self.num_heads = num_heads

#         # 可学习的维度权重 logits（通过 softmax 分配 head 维度比例）
#         self.head_weight_logits = nn.Parameter(torch.randn(num_heads))

#         # 公共线性变换（先做统一映射，后分配 head_dim）
#         self.q_linear = nn.Linear(d_model, d_model)
#         self.k_linear = nn.Linear(d_model, d_model)
#         self.v_linear = nn.Linear(d_model, d_model)

#         self.final_linear = nn.Linear(d_model, d_model)
#         self.layer_norm = nn.LayerNorm(d_model)



#     def forward(self, query, key=None, value=None, mask=None):
#         if key is None:
#             key = query
#         if value is None:
#             value = query

#         B, T, _ = query.size()


#         # === Step 1: 动态维度分配（带最小维度限制）===
#         head_ratios = F.softmax(self.head_weight_logits, dim=0)  # [num_heads]
#         raw_dims = head_ratios * self.d_model

#         # min_dim = self.d_model // (4*self.num_heads)
#         min_dim = 8
#         min_total = min_dim * self.num_heads
#         if min_total > self.d_model:
#             raise ValueError(f"Minimum total head dim {min_total} exceeds d_model {self.d_model}")

#         # 分配剩余维度
#         adjustable_dim = self.d_model - min_total
#         adjustable_ratios = head_ratios / head_ratios.sum()
#         adjustable_dims = adjustable_ratios * adjustable_dim

#         # 加上最小值，得到最终 float 维度
#         head_dims_float = adjustable_dims + min_dim

#         # 离散化（floor + 残差调和）
#         head_dims = torch.floor(head_dims_float).int().tolist()
#         total = sum(head_dims)
#         diff = self.d_model - total

#         if diff != 0:
#             residuals = [(head_dims_float[i] - head_dims[i]).item() for i in range(self.num_heads)]
#             sorted_indices = sorted(range(self.num_heads), key=lambda i: -residuals[i]) if diff > 0 \
#                            else sorted(range(self.num_heads), key=lambda i: residuals[i])
#             for i in range(abs(diff)):
#                 head_dims[sorted_indices[i]] += 1 if diff > 0 else -1

#         assert sum(head_dims) == self.d_model, "最终 head_dims 总和必须等于 d_model"



#         # === Step 2: 投影到 d_model 维度 ===
#         q_all = self.q_linear(query)  # [B, T, d_model]
#         k_all = self.k_linear(key)
#         v_all = self.v_linear(value)


#         # === Step 3: 根据分配的 head_dim 切片 ===
#         head_outputs = []
#         start = 0
#         for i in range(self.num_heads):
#             dim = head_dims[i]
#             q = q_all[:, :, start:start+dim]
#             k = k_all[:, :, start:start+dim]
#             v = v_all[:, :, start:start+dim]
#             start += dim

#             scores = torch.matmul(q, k.transpose(-2, -1)) / (dim ** 0.5)  # [B, T, T]
#             if mask is not None:
#                 if mask.dim() == 2:
#                     scores = scores.masked_fill(mask.unsqueeze(1) == 0, float('-inf'))
#                 elif mask.dim() == 3:
#                     scores = scores.masked_fill(mask == 0, float('-inf'))

#             attn_weights = F.softmax(scores, dim=-1)
#             output = torch.matmul(attn_weights, v)  # [B, T, dim]
#             head_outputs.append(output)


#         # === Step 4: 拼接所有 head 输出 ===
#         concat = torch.cat(head_outputs, dim=-1)  # [B, T, d_model]
#         output = self.final_linear(concat)
#         output = self.layer_norm(output + query)
#         return output
    



    ## 前馈层（MLP：非线性激活+线性层）
class FeedForward(nn.Module):
    """
    dff: 前馈网络中间隐藏层的大小
    d_model: 输入数据的维度embeding_dim (input_dim)
    dropout: 正则化参数
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
    
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)  # 第一层线性变换（升维）
        self.gelu = nn.GELU()                # 非线性激活函数
        self.fc2 = nn.Linear(d_ff, d_model)  # 第二层线性变换（降维）
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)   # 防止过拟合


    def forward(self, x):
        
        ff_out = self.fc2(self.dropout(self.gelu(self.fc1(x))))
        ff_out  = self.layer_norm(ff_out + x)

        return ff_out




    ## 编码器层
class EncoderLayer(nn.Module):


    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        """
        一个 Transformer 编码器层，包含多头注意力和前馈神经网络。
        
        参数:
        - d_model: 词向量的维度
        - num_heads: 多头注意力的头数
        - d_ff: 前馈网络的隐藏层维度
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
        # print("x_shape:", x.shape)
        attn_output = self.multihead_attention(x, x, x, mask) # 这里面x,x,x代表自注意力机制，即QKV都是通过x线性变换得到

        x = self.layer_norm1(attn_output + x)  # 残差连接 + 层归一化

        # 前馈网络
        ff_output = self.feed_forward(x)
        
        x = self.layer_norm2(ff_output + x)  # 残差连接 + 层归一化

        return x
    



    ## 多层编码器叠加
class Encoder(nn.Module):

    
    def __init__(self, features_num, embedding_dim, num_layers, num_heads, d_ff, max_len=512, dropout=0.1):
        """
        Transformer 编码器。
        
        参数:
        - features_num: 输入的特征数量
        - embedding_dim: 词嵌入维度
        - num_layers: 编码器的层数（有多少个编码器堆叠）
        - num_heads: 多头注意力的头数
        - d_ff: 前馈网络的隐藏层维度
        - max_len: 句子的最大长度
        """
        super(Encoder, self).__init__()

        # 词嵌入和位置编码
        self.embedding = Embedding(features_num, embedding_dim, max_len)

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
            # print("enconder_output.shape:", x.shape)
            
   
        return x




    # 解码器层
class DecoderLayer(nn.Module):


    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()

        # 自注意力（带mask）
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)

        # 编码器-解码器注意力（交叉注意力）
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.norm2 = nn.LayerNorm(d_model)

        # 前馈层
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)


    def forward(self, x, enc_output, tgt_mask=None, memory_mask=None):
        # 自注意力（目标语言）
        _x = self.self_attn(x, x, x, tgt_mask)  # 注意力使用目标语言自己与自己的注意力
        x = self.norm1(x + self.dropout(_x))  # 残差连接和归一化

        # 编码器-解码器注意力（源-目标交互）
        _x = self.cross_attn(x, enc_output, enc_output, memory_mask)  # 使用编码器输出和目标进行交互
        x = self.norm2(x + self.dropout(_x))  # 残差连接和归一化

        # 前馈网络
        _x = self.feed_forward(x)
        x = self.norm3(x + self.dropout(_x))  # 残差连接和归一化

        return x




    # 多层解码器的叠加
class Decoder(nn.Module):


    def __init__(self, features_num, embedding_dim, num_layers, num_heads, d_ff, max_len=512, dropout=0.1, label_num=1):
        super(Decoder, self).__init__()

        # 词嵌入层 + 位置编码
        self.embedding = Embedding(features_num, embedding_dim, max_len, False)
        
        # 解码器的每一层
        self.layers = nn.ModuleList([
            DecoderLayer(embedding_dim, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        # print(self.layers)

        # 输出层（将解码器的输出映射到输出空间）
        self.fc_out = nn.Linear(embedding_dim, label_num)

        # dropout
        self.dropout = nn.Dropout(dropout)


    def forward(self, x, enc_output, tgt_mask=None, memory_mask=None):
        # 词嵌入 + 位置编码
        x = self.embedding(x)  # x 的形状为 (batch_size, seq_len)
        
        # 通过解码器层
        for layer in self.layers:
            x = layer(x, enc_output, tgt_mask, memory_mask)  # 每一层解码器的输出

        # 通过输出层生成词汇概率分布
        # print(x.shape)
        # x = self.fc_out(x)  # (batch_size, seq_len, label_num)
            # print("deconder_output.shape:", x.shape)

        return x

    


## 封装为tarnsformer
class Transformer(nn.Module):


    def __init__(self, features_num, embedding_dim, num_layers, num_heads, d_ff, max_len=512, dropout=0.1):
        super(Transformer, self).__init__()
        self.encoder = Encoder(features_num, embedding_dim, num_layers, num_heads, d_ff, max_len, dropout)
        self.decoder = Decoder(features_num, embedding_dim, num_layers, num_heads, d_ff, max_len, dropout)
        self.fc_out = nn.Linear(embedding_dim, features_num)  # 输出层，映射到词汇表大小


    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # 编码器输出
        enc_output = self.encoder(src, src_mask)
        print(enc_output.shape)
        
        # 解码器输出
        decoder_output = self.decoder(tgt, enc_output, tgt_mask, src_mask)
        print(decoder_output.shape)
        
        # 通过全连接层进行输出映射
        # 只取最后一个时间步的 decoder 输出
        last_decoder_output = decoder_output[:, -1, :]  # [batch_size, embedding_dim]
        output = self.fc_out(last_decoder_output)       # [batch_size, features_num]

        print(output.shape)

        return output


    def generate_tgt_mask(self, seq_len):
        # 生成目标序列的mask（防止看见未来的词）
        mask = torch.tril(torch.ones(seq_len, seq_len)).bool()  # 下三角矩阵
        return mask.unsqueeze(0).unsqueeze(0)  # 扩展到 (1, 1, seq_len, seq_len) 的形状


    # def generate_src_mask(self, seq_len):
    #     # 生成源序列的mask（如果需要）
    #     return torch.ones(1, 1, seq_len, seq_len).to(torch.bool)




def train(model, dataloader, criterion, optimizer, epoch, device, pred_len):
    model.train()
    total_loss = 0.0

    for batch_idx, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)  # x: [B, T_src, D], y: [B, T_tgt+1, 1]

        # 构造解码器输入 & 输出（shift 1）
        tgt_input = y[:, :-1, :]         # [B, T_tgt, 1]
        tgt_output = y[:, 1:, :]         # [B, T_tgt, 1]

        # 前向传播
        optimizer.zero_grad()
        
        # 用多步预测的方式进行训练
        decoder_input = tgt_input
        predictions = []
        for _ in range(pred_len):  # 每次预测一个时间步
            output = model(x, decoder_input)     # 模型预测
            predictions.append(output[:, -1:, :])  # 只保留每次预测的最后一个时间步

            # 更新decoder_input，用预测的值作为下一次输入
            decoder_input = torch.cat([decoder_input, output[:, -1:, :]], dim=1)

        predictions = torch.cat(predictions, dim=1)  # 合并所有时间步的预测

        # 维度检查（强烈推荐）
        assert predictions.shape == tgt_output.shape, \
            f"Predictions shape {predictions.shape} != Target shape {tgt_output.shape}"

        # 计算损失
        loss = criterion(predictions, tgt_output)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if (batch_idx + 1) % 10 == 0:
            print(f"[Epoch {epoch}] Batch {batch_idx+1}/{len(dataloader)} Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(dataloader)
    print(f"[Epoch {epoch}] Train Avg Loss: {avg_loss:.4f}")



