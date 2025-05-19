__author__ = 'minjinwu'

import math
import torch
import torch.nn as nn

class Embedding(nn.Module):


    def __init__(self, vocab_size, embedding_dim, max_len=512, shared_weight=None):
        """
        词嵌入类，支持共享词嵌入和位置编码。
        
        参数:
        - vocab_size: 词汇表大小   总共有多少个词
        - embedding_dim: 词嵌入维度  每一个词转换为向量的维度大小
        - max_len: 句子最大长度（用于位置编码） 一次识别最多多少个词,少会padding,多会忽略
        - shared_weight: 可选，是否共享已有的嵌入层 (nn.Embedding)
        """
        super(Embedding, self).__init__()

        # 如果传入 shared_weight，则使用共享的嵌入层（就是encod和decoding进行权重共享）
        if shared_weight is not None:
            self.embedding = shared_weight  # 共享权重
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_dim)

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
        这里面的seq_len是你数据实际输入时候的长度,它可以比句子最大长度max_len要大,也可以小一些
        
        输出:
        - 嵌入后的张量，形状为 (batch_size, seq_len, embedding_dim)
        """
        embedded = self.embedding(input_ids)  # 词嵌入
        seq_len = input_ids.size(1)
        positional_enc = self.positional_encoding[:, :seq_len, :].to(embedded.device)  # 取前 seq_len 个位置编码

        return embedded + positional_enc  # 词嵌入 + 位置编码
    


# if __name__ == '__main__':
#     # 设置参数
#     vocab_size = 10  # 假设词汇表大小为100
#     embedding_dim = 16  # 词嵌入维度为16
#     max_len = 10  # 句子最大长度

#     # 创建词嵌入模型
#     embedding_layer = Embedding(vocab_size, embedding_dim, max_len)

#     # 生成测试输入（batch_size=2, seq_len=5），随机选择词索引
#     input_ids = torch.randint(0, vocab_size, (2, 5))
#     print("输入词索引:")
#     print(input_ids)

#     # 前向传播
#     output = embedding_layer(input_ids)
#     print("\n输出词嵌入+位置编码:")
#     print(output)
#     print(output.shape)

if __name__ == '__main__':

    word_to_index = {
    "apple": 0,
    "banana": 1,
    "cherry": 2,
    "date": 3,
    "elderberry": 4,
    "fig": 5,
    "grape": 6,
    "honeydew": 7,
    "kiwi": 8,
    "lemon": 9
}

index_to_word = {index: word for word, index in word_to_index.items()}

# 设置参数
vocab_size = len(word_to_index)  # 词汇表大小
embedding_dim = 16  # 词嵌入维度
max_len = 5  # 假设句子的最大长度为5

# 创建词嵌入模型
embedding_layer = Embedding(vocab_size, embedding_dim, max_len)

# 输入为单词
input_words = ["apple", "banana", "cherry", "date"]  # 一个例子句子

# 将单词转换为词索引
input_ids = torch.tensor([word_to_index[word] for word in input_words]).unsqueeze(0)  # 转换为 (1, seq_len)
print("输入词索引:")
print(input_ids)

# 前向传播
output = embedding_layer(input_ids)
print("\n输出词嵌入+位置编码:")
print(output)
print(output.shape)
