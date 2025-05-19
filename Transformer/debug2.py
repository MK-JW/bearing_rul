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




class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=512, num_heads=6, min_dim=8, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.min_dim = min_dim
        
        # 可学习参数
        self.head_weight_logits = nn.Parameter(torch.randn(num_heads))
        
        # 投影层
        self.qkv_proj = nn.Linear(d_model, d_model*3)
        self.out_proj = nn.Linear(d_model, d_model)
        
        # 位置编码
        self.register_buffer("pos_range", torch.arange(d_model).float())

    def _compute_integer_dims(self, head_dims_float):
        """ 生成整数维度分配（带约束调整）"""
        with torch.no_grad():
            # 初始取整
            head_dims_int = torch.floor(head_dims_float).int()
            diff = self.d_model - head_dims_int.sum()
            
            # 残差调整
            residuals = head_dims_float - head_dims_float.floor()
            if diff > 0:
                # 需要增加维度
                _, indices = torch.topk(residuals, k=abs(diff))
                head_dims_int[indices] += 1
            else:
                # 需要减少维度
                _, indices = torch.topk(-residuals, k=abs(diff))
                head_dims_int[indices] -= 1
            
            return head_dims_int

    def forward(self, query, key=None, value=None, mask=None):
        B, T, _ = query.shape

        if key == None: key = query
        if value == None: value = query
        
        # === 浮点维度计算 ===
        head_ratios = F.softmax(self.head_weight_logits, dim=0)
        head_dims_float = self.min_dim + head_ratios * (self.d_model - self.min_dim*self.num_heads)
        
        # === 整数维度生成 ===
        head_dims_int = self._compute_integer_dims(head_dims_float)
        
        # === 动态投影（双模式）===
        q_all, k_all, v_all = self.qkv_proj(query).chunk(3, dim=-1)
        
        # 模式2：整数维度硬切片（实际计算）
        start_int = 0
        outputs = []
        for i in range(self.num_heads):
            dim = head_dims_int[i].item()
            q = q_all[:, :, start_int:start_int+dim]  # 硬切片
            k = k_all[:, :, start_int:start_int+dim]
            v = v_all[:, :, start_int:start_int+dim]
            start_int += dim
            
            # 注意力计算
            scores = torch.matmul(q, k.transpose(-2, -1)) / (dim**0.5)
            attn = F.softmax(scores, dim=-1)
            outputs.append(torch.matmul(attn, v))
        
        # === 损失计算 ===
        output = self.out_proj(torch.cat(outputs, dim=-1))
       
        # 浮点损失（优化head_weight_logits）
        float_loss = (head_dims_float.sum() - self.d_model)**2 + \
                    F.relu(self.min_dim - head_dims_float).mean()
        
        # 整数损失（辅助监督）
        int_loss = F.mse_loss(head_dims_float, head_dims_int.float())
        
        return output, float_loss + 0.5 * int_loss
    

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
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1, min_dim=16):
        """
        一个 Transformer 编码器层，包含多头注意力和前馈神经网络。
        
        参数:
        - d_model: 词向量的维度
        - num_heads: 多头注意力的头数
        - d_ff: 前馈网络的隐藏层维度
        - min_dim: 每个头的最小维度
        """
        super(EncoderLayer, self).__init__()

        self.multihead_attention = MultiHeadAttention(d_model, num_heads, min_dim)
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
        - 辅助损失
        """
        # 多头注意力层
        attn_output, attn_loss = self.multihead_attention(x, x, x)  # 返回输出和辅助损失
        
        x = self.layer_norm1(x + self.dropout(attn_output))  # 残差连接 + 层归一化

        # 前馈网络
        ff_output = self.feed_forward(x)
        
        x = self.layer_norm2(x + self.dropout(ff_output))  # 残差连接 + 层归一化

        return x, attn_loss  # 返回编码器的输出和注意力损失
    




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
        - total_aux_loss: 汇总的辅助损失
        """
        x = self.embedding(x)  # 词嵌入 + 位置编码
        
        total_aux_loss = 0.0  # 初始化辅助损失

        for layer in self.layers:
            x, aux_loss = layer(x, mask)  # 通过每一层编码器，并返回辅助损失
            total_aux_loss += aux_loss  # 累加所有层的辅助损失

        return x, total_aux_loss  # 返回最终的编码器输出和总辅助损失



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
        _x, self_attn_aux_loss = self.self_attn(x, x, x, tgt_mask)  # 自注意力
        x = self.norm1(x + self.dropout(_x))  # 残差连接和归一化

        # 编码器-解码器注意力（源-目标交互）
        _x, cross_attn_aux_loss = self.cross_attn(x, enc_output, enc_output, memory_mask)  # 编码器-解码器交互
        x = self.norm2(x + self.dropout(_x))  # 残差连接和归一化

        # 前馈网络
        _x = self.feed_forward(x)
        x = self.norm3(x + self.dropout(_x))  # 残差连接和归一化

        # 返回输出和各部分的辅助损失
        total_aux_loss = self_attn_aux_loss + cross_attn_aux_loss
        return x, total_aux_loss



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
        # dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, tgt_mask=None, memory_mask=None):
        # 词嵌入 + 位置编码
        x = self.embedding(x)  # x 的形状为 (batch_size, seq_len，label_num)
        
        total_aux_loss = 0.0  # 初始化辅助损失

        # 通过解码器层
        for layer in self.layers:
            x, aux_loss = layer(x, enc_output, tgt_mask, memory_mask)  # 每一层解码器的输出，并返回辅助损失
            total_aux_loss += aux_loss  # 累加所有层的辅助损失

        return x, total_aux_loss  # 返回最终的解码器输出和总辅助损失




class Transformer(nn.Module):
    def __init__(self, features_num, embedding_dim, num_layers, num_heads, d_ff, max_len=512, dropout=0.1, label_num=1):
        super(Transformer, self).__init__()
        
        # 编码器和解码器
        self.encoder = Encoder(features_num, embedding_dim, num_layers, num_heads, d_ff, max_len, dropout)
        self.decoder = Decoder(features_num, embedding_dim, num_layers, num_heads, d_ff, max_len, dropout)
        
        # 输出层，映射到输出空间
        self.fc_out = nn.Linear(embedding_dim, label_num)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # 编码器输出
        enc_output, enc_aux_loss = self.encoder(src, src_mask)  # 返回编码器输出和辅助损失
        
        # 解码器输出
        decoder_output, dec_aux_loss = self.decoder(tgt, enc_output, tgt_mask, src_mask)  # 返回解码器输出和辅助损失
        
        # 多个时间步输出
        output = self.fc_out(decoder_output)  # [batch_size, seq_len, label_num]

        # 总辅助损失
        total_aux_loss = enc_aux_loss + dec_aux_loss
        
        return output, total_aux_loss

    def generate_tgt_mask(self, seq_len):
        """
        生成目标序列的mask（防止解码器访问未来的时间步）。
        :param seq_len: 序列长度
        :return: 目标掩码 (1, 1, seq_len, seq_len)
        """
        mask = torch.tril(torch.ones(seq_len, seq_len)).bool()  # 生成下三角矩阵，防止访问未来
        return mask.unsqueeze(0).unsqueeze(0)  # 扩展为 (1, 1, seq_len, seq_len) 形状




def train(model, dataloader, criterion, optimizer, epoch, device, pred_len, aux_loss_weight=1.0):
    model.train()
    total_loss = 0.0

    for batch_idx, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)  # x: [B, T_src, D], y: [B, T_tgt+1, 1]

        tgt_input = y[:, :1, :]         # [B, 15, 6]
        tgt_output = y[:, 1:, :]        # [B, T_tgt, 1]

        optimizer.zero_grad()

        decoder_input = tgt_input
        predictions = []
        total_aux_loss = 0.0  # <<< 初始化辅助损失

        for i in range(pred_len):
            # <<< 假设模型返回两个值：主输出 + 当前步的辅助损失（或累计）
            output, aux_loss = model(x, decoder_input)
            predictions.append(output[:, -1:, :])
            total_aux_loss += aux_loss  # <<< 累加辅助损失（可能是标量或 tensor）

            decoder_input = torch.cat([decoder_input, tgt_output[:, i:i+1, :]], dim=1)

        predictions = torch.cat(predictions, dim=1)

        assert predictions.shape == tgt_output.shape, \
            f"Predictions shape {predictions.shape} != Target shape {tgt_output.shape}"

        loss = criterion(predictions, tgt_output)

        # <<< 总损失 = 主损失 + 辅助损失（可加权）
        total_loss_step = loss + aux_loss_weight * total_aux_loss

        total_loss_step.backward()
        optimizer.step()

        total_loss += loss.item()  # <<< 保持记录主损失

        if (batch_idx + 1) % 10 == 0:
            print(f"[Epoch {epoch}] Batch {batch_idx+1}/{len(dataloader)} "
                  f"Loss: {loss.item():.4f} + Aux: {aux_loss_weight * total_aux_loss:.4f}")

    avg_loss = total_loss / len(dataloader)
    print(f"[Epoch {epoch}] Train Avg Loss: {avg_loss:.4f}")




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
