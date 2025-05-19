__author__ = 'minjinwu'

import torch
import torch.nn as nn

class FeedForward(nn.Module):


    def __init__(self, d_model, d_ff, dropout=0.1):
    
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)  # 第一层线性变换（升维）
        self.relu = nn.ReLU()                # 非线性激活函数
        self.fc2 = nn.Linear(d_ff, d_model)  # 第二层线性变换（降维）
        self.dropout = nn.Dropout(dropout)   # 防止过拟合


    def forward(self, x):
        
        return self.fc2(self.dropout(self.relu(self.fc1(x))))


if __name__ == '__main__':

    # 示例：输入维度 (batch_size=2, seq_len=5, d_model=512)
    x = torch.randn(2, 5, 512)
    ffn = FeedForward(d_model=512, d_ff=2048, dropout=0.1)
    output = ffn(x)
    print(output.shape)  # 预期输出: torch.Size([2, 5, 512])