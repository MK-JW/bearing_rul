__author__ = 'minjinwu'


import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class DualLossDynamicMHA(nn.Module):

    def __init__(self, d_model=512, num_heads=8, min_dim=9, dropout=0.1):
            super().__init__()
            self.d_model = d_model
            self.num_heads = num_heads
            self.min_dim = min_dim
            
            # 可学习参数 
            self.head_weight_logits = nn.Parameter(torch.randn(num_heads))
            
            # 投影层
            self.qkv_proj = nn.Linear(d_model, d_model*3)
            self.out_proj = nn.Linear(d_model, d_model)
            

    def _compute_integer_dims(self, head_dims_float, q_all, k_all, v_all):
        """ 生成整数维度分配（带约束调整），并进行硬切片操作 """
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
        
        # 切片操作（硬切片）
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
            print(head_dims_int)

        return head_dims_int, torch.cat(outputs, dim=-1)

    def forward(self, query, return_aux_loss=False):
        B, T, _ = query.shape
        
        # === 浮动维度计算 ===
        head_ratios = F.softmax(self.head_weight_logits, dim=0)
        head_dims_float = self.min_dim + head_ratios * (self.d_model - self.min_dim * self.num_heads)
        print(head_dims_float)
        
        # === 整数维度生成和硬切片操作 ===
        q_all, k_all, v_all = self.qkv_proj(query).chunk(3, dim=-1)
        head_dims_int, outputs = self._compute_integer_dims(head_dims_float, q_all, k_all, v_all)
        
        # === 损失计算 ===
        output = self.out_proj(outputs)
        
        if return_aux_loss:
            # 浮动损失（优化head_weight_logits）
            float_loss = (head_dims_float.sum() - self.d_model)**2 +  \
                        F.relu(self.min_dim - head_dims_float).mean()
            
            # 整数损失（辅助监督）
            int_loss = F.mse_loss(head_dims_float, head_dims_int.float())
            
            return output, float_loss + 0.5 * int_loss
        
        return output
    

model = DualLossDynamicMHA(d_model=64, num_heads=4)
x = torch.randn(2, 10, 64, requires_grad=True)

# 前向计算
output, aux_loss = model(x, return_aux_loss=True)
loss = output.mean() + aux_loss

# 反向传播
loss.backward()

# 检查梯度
print("head_weight_logits梯度:", model.head_weight_logits.grad)  # 应非空
print("qkv_proj梯度:", model.qkv_proj.weight.grad.norm().item())  # 应非零