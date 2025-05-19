__author__ = 'minjinwu'

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import pandas as pd 


class Embedding(nn.Module):
    def __init__(self, features_num, embedding_dim, max_len=512, is_encoder=True, label_num=1, dropout=0.1):
        """
        词嵌入类，支持位置编码和不同的线性变换。
        
        参数:
        - features_num: 特征维度大小(例如6)
        - embedding_dim: 词嵌入维度
        - max_len: 最大序列长度(位置编码的长度)
        - is_encoder: 是否是编码器(编码器和解码器的处理可能不同)
        - label_num: 预测目标的数量(默认1,表示回归任务中预测一个目标)
        """
        super(Embedding, self).__init__()

        # 根据是否是编码器来设置不同的线性变换
        if is_encoder:
            # 编码器输入特征维度是 features_num
            self.embedding = nn.Linear(features_num, embedding_dim)
        else:
            # 解码器的输入通常是目标值序列，假设每个目标值是标量
            self.embedding = nn.Linear(label_num, embedding_dim)  # 支持多目标预测

        self.dropout = nn.Dropout(dropout)

        # 位置编码
        self.register_buffer("positional_encoding", self.create_positional_encoding(max_len, embedding_dim))


    def create_positional_encoding(self, max_len, embedding_dim):
        """
        生成位置编码，采用正弦余弦方式。
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
        - 输入特征 -> 词嵌入
        - 词嵌入 + 位置编码

        输入:
        - input_ids: (batch_size, seq_len, features_num) 形状的张量，表示时间序列数据的每个时间步的特征
        """
        # 词嵌入
        embedded = self.embedding(input_ids)  # 形状: (batch_size, seq_len, embedding_dim)
        
        # 取前 seq_len 个位置编码
        seq_len = input_ids.size(1)
        positional_enc = self.positional_encoding[:, :seq_len, :].to(embedded.device)

        # 返回嵌入后的张量和位置编码的和
        return self.dropout(embedded + positional_enc)  # (batch_size, seq_len, embedding_dim)





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

        if key == None: key = query
        if value == None: value = query

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


#     def __init__(self, d_model, num_heads, dropout=0.1, return_attn=False):
#         super(MultiHeadAttention, self).__init__()
#         self.d_model = d_model
#         self.num_heads = num_heads
#         self.return_attn = return_attn

#         # 可学习的维度权重 logits（通过 softmax 分配 head 维度比例）
#         self.head_weight_logits = nn.Parameter(torch.randn(num_heads), requires_grad=True)

#         # 公共线性变换（先做统一映射，后分配 head_dim）
#         self.q_linear = nn.Linear(d_model, d_model)
#         self.k_linear = nn.Linear(d_model, d_model)
#         self.v_linear = nn.Linear(d_model, d_model)

#         self.final_linear = nn.Linear(d_model, d_model)
#         self.layer_norm = nn.LayerNorm(d_model)
#         self.dropout = nn.Dropout(dropout)



#     def forward(self, query, key=None, value=None, mask=None):
#         if key is None: key = query
#         if value is None: value = query

#         B, T, _ = query.size()


#         # === Step 1: 动态维度分配 ===
#         head_ratios = F.softmax(self.head_weight_logits, dim=0)
#         min_dim = 8
#         adjustable_dim = self.d_model - min_dim * self.num_heads
#         if adjustable_dim < 0:
#             raise ValueError("d_model太小，无法满足最小维度分配")

#         raw_dims = head_ratios * adjustable_dim
#         head_dims_float = raw_dims + min_dim
#         head_dims = torch.floor(head_dims_float).int().tolist()

#         # 调整残差
#         diff = self.d_model - sum(head_dims)
#         if diff != 0:
#             residuals = [(head_dims_float[i] - head_dims[i]).item() for i in range(self.num_heads)]
#             sorted_idx = sorted(range(self.num_heads), key=lambda i: -residuals[i] if diff > 0 else residuals[i])
#             for i in range(abs(diff)):
#                 head_dims[sorted_idx[i]] += 1 if diff > 0 else -1

#         assert sum(head_dims) == self.d_model



#         # === Step 2: 投影到 d_model 维度 ===
#         q_all = self.q_linear(query)  # [B, T, d_model]
#         k_all = self.k_linear(key)
#         v_all = self.v_linear(value)


#         # === Step 3: 根据分配的 head_dim 切片 ===
#         head_outputs = []
#         attn_weights_all = [] if self.return_attn else None
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

#             attn = F.softmax(scores, dim=-1)
#             attn = self.dropout(attn)

#             output = torch.matmul(attn, v)  # [B, T, dim]
#             head_outputs.append(output)
#             if self.return_attn:
#                 attn_weights_all.append(attn.detach())


#         # === Step 4: 拼接所有 head 输出 ===
#         concat = torch.cat(head_outputs, dim=-1)  # [B, T, d_model]
#         output = self.final_linear(concat)
#         output = self.layer_norm(output + query)
        
#         if self.return_attn:
#             return output, attn_weights_all
#         else:
#             return output




## 前馈层（MLP：非线性激活+线性层）
class FeedForward(nn.Module):
    """
    dff: 前馈网络中间隐藏层的大小
    d_model: 输入数据的维度embeding_dim 
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

        x = self.layer_norm1(x + self.dropout(attn_output))  # 残差连接 + 层归一化

        # 前馈网络
        ff_output = self.feed_forward(x)
        
        x = self.layer_norm2(x + self.dropout(ff_output))  # 残差连接 + 层归一化

        return x
    



     ## 多层编码器叠加
class Encoder(nn.Module):

    
    def __init__(self, features_num, embedding_dim, num_layers, num_heads, d_ff, max_len=512, dropout=0.1):
        """
        参数:
        - features_num: 输入的特征维度（即每个时间步的变量数）
        - embedding_dim: 嵌入后的维度（即 d_model）
        - num_layers: 编码器层数
        - num_heads: 注意力头数
        - d_ff: 前馈网络隐藏层维度
        - max_len: 最大序列长度（用于位置编码）
        - dropout: Dropout 比例
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
        前向传播：输入时间序列经过多层编码器处理。

        输入:
        - x: (batch_size, seq_len, features_num )连续时间序列
        - mask: 可选的 attention 掩码

        输出:
        - (batch_size, seq_len, embedding_dim)
        """
        x = self.embedding(x)  # 词嵌入 + 位置编码
        
        for layer in self.layers:
            x = layer(x, mask)  # 通过每一层编码器
            # print("enconder_output.shape:", x.shape)
            
   
        return x
    



     ## 解码器层
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


    def __init__(self, features_num, embedding_dim, num_layers, num_heads, d_ff, max_len=512, dropout=0.1):
        super(Decoder, self).__init__()

        # 词嵌入层 + 位置编码
        self.embedding = Embedding(features_num, embedding_dim, max_len, False)
        
        # 解码器的每一层
        self.layers = nn.ModuleList([
            DecoderLayer(embedding_dim, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        # print(self.layers)

        # dropout
        self.dropout = nn.Dropout(dropout)


    def forward(self, x, enc_output, tgt_mask=None, memory_mask=None):
        # 词嵌入 + 位置编码
        x = self.embedding(x)  # x 的形状为 (batch_size, seq_len，label_num)
        
        # 通过解码器层
        for layer in self.layers:
            x = layer(x, enc_output, tgt_mask, memory_mask)  # 每一层解码器的输出

        return x         


## 封装为tarnsformer
class Transformer(nn.Module):


    def __init__(self, features_num, embedding_dim, num_layers, num_heads, d_ff, max_len=512, dropout=0.1, label_num=1):

        super(Transformer, self).__init__()
        self.encoder = Encoder(features_num, embedding_dim, num_layers, num_heads, d_ff, max_len, dropout)
        self.decoder = Decoder(features_num, embedding_dim, num_layers, num_heads, d_ff, max_len, dropout)
        self.fc_out = nn.Linear(embedding_dim, label_num)  # 输出层，映射到输出空间


    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # 编码器输出
        enc_output = self.encoder(src, src_mask)
        # print(enc_output.shape)
        
        # 解码器输出
        decoder_output = self.decoder(tgt, enc_output, tgt_mask, src_mask)
        # print(decoder_output.shape)
        
        # # 只取最后一个时间步的 decoder 输出
        # last_decoder_output = decoder_output[:, -1, :]  # [batch_size, seq_len, embedding_dim]
        # output = self.fc_out(last_decoder_output)       # [batch_size, seq_len, label_num]

        # 多个时间步输出
        output = self.fc_out(decoder_output)       # [batch_size, seq_len, label_num]
        # print(output.shape)

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
    # for name, param in model.named_parameters():
    #     if "head_weight_logits" in name:
    #         print(f"[OK] Found: {name}, shape = {param.shape}, grad:{param.grad}")


    for batch_idx, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)  # x: [B, T_src, D], y: [B, T_tgt+1, 1]
        # print(x[0:1, :, :].shape)
        # print(x[0:1, :, :])

        # break
        # 构造解码器输入 & 输出（shift 1）
        tgt_input = y[:, :1, :]         # [B, 1, 1] 只取目标序列的第一个时间步作为解码器的输入
        tgt_output = y[:, 1:, :]        # [B, T_tgt, 1] 目标序列的剩余作为输出
        # print("tgt_input.shape:", tgt_input.shape)
        # print(tgt_input[0:1, :, :])
        # print("tgt_output.shape:", tgt_output.shape)
        # print(tgt_output[0:1, :, :])

        # 前向传播
        optimizer.zero_grad()
        
        # 用多步预测的方式进行训练
        decoder_input = tgt_input
        predictions = []
        for i in range(pred_len):  # 每次预测一个时间步
            output = model(x, decoder_input)     # 模型预测
            predictions.append(output[:, -1:, :])  # 只保留每次预测的最后一个时间步

            # print("output.shape:", output.shape)
            # print(output)
            # print("x.shape:", x.shape)
            # print("x:", x)
            # print("deconder_input:", decoder_input)

            # 更新decoder_input，用已知的标签值进行concatenate(区别于test)
            decoder_input = torch.cat([decoder_input, tgt_output[:, i:i+1, :]], dim=1)
        
        # print(output.shape)

        # 只取最后 pred_len 个时间步的预测
        predictions = torch.cat(predictions, dim=1)  # 合并所有时间步的预测
        # break

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
    pass




def test(model, dataloader, criterion, device, pred_len):
    model.eval()  # 切换到评估模式
    total_loss = 0.0
    predictions_all = []
    ground_truth_all = []

    with torch.no_grad():  # 不计算梯度，节省内存
        for batch_idx, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)  # x: [B, T_src, D], y: [B, T_tgt+1, D]

            # 解码器的初始输入是已知的前序数据
            decoder_input = y[:, :1, :]  # 使用y的第一个时间步（例如y0）作为起始输入

            pred_seq = []
            for _ in range(pred_len):
                output = model(x, decoder_input)           # 模型预测
                next_step = output[:, -1:, :]              # 只取最后一个时间步预测
                pred_seq.append(next_step)
                decoder_input = torch.cat([decoder_input, next_step], dim=1)  # 自回归更新输入

            preds = torch.cat(pred_seq, dim=1)  # 合并所有时间步的预测结果 [B, pred_len, D]
            targets = y[:, 1:pred_len+1, :]     # 对应的真实标签 [B, pred_len, D]（跳过y0）

            # 将预测和真实标签存储下来
            predictions_all.append(preds.cpu())
            ground_truth_all.append(targets.cpu())

            # 计算当前 batch 的损失
            loss = criterion(preds, targets)
            total_loss += loss.item()

        # 合并所有 batch 的预测与标签
        predictions_all = torch.cat(predictions_all, dim=0)   # [N, pred_len, D]
        ground_truth_all = torch.cat(ground_truth_all, dim=0) # [N, pred_len, D]

        # 输出每个 epoch 的平均损失
        avg_loss = total_loss / len(dataloader)
        print(f"Test Avg Loss: {avg_loss:.4f}")

    return predictions_all, ground_truth_all
