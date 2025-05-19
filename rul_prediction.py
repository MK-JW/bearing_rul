__author__ = 'minjinwu'


import os
import torch
import numpy as np
import pandas as pd
import torch.nn as nn 
import torch.optim as optim
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis
from torch.utils.data import TensorDataset, DataLoader


from Transformer.debug import train
from Transformer.debug import test
from Transformer.debug import Transformer

# 检验GPU是否可用
if torch.cuda.is_available():
    print("CUDA is available. PyTorch will use GPU.")
    print("Current device:", torch.cuda.current_device())
    print("GPU Name:", torch.cuda.get_device_name(torch.cuda.current_device()))
else:
    print("CUDA is not available. PyTorch will use CPU.")



## 处理时域数据
def calculate_time_domain_features_label(df, idx, csv_files):

    # 计算时域特征
    variance = df.var()
    rms = np.sqrt((df ** 2).mean())  # 均方根计算
    max_val = df.max()
    min_val = df.min()
    peak_to_peak = (max_val - min_val)
    skewness = skew(df.values)
    kurt = kurtosis(df.values)

    # 计算的标签值
    rul = (len(csv_files) - idx - 1) / (len(csv_files) - 1) 
    
    # 将所有特征保存在一个字典里
    features = {
        'peak_to_peak': peak_to_peak,
        'skewness': skewness,
        'kurt': kurt,
        'variance': variance,
        'rms': rms,
        'max': max_val,
        'rul': rul
    }
    
    return features

## 生成序列函数
def create_sequences(df, src_len, tgt_len):

    data = df.astype('float32')
    features_num = data.shape[1] - 1  # 除去最后一列的RUL标签

    src_seq = []
    tgt_seq = []

    for i in range(len(data) - src_len - tgt_len):
        src = data.iloc[i:i+src_len, :features_num].values
        tgt = data.iloc[i+src_len - 1:i+src_len+tgt_len - 1, -1].values.reshape(-1, 1)

        src_seq.append(src)
        tgt_seq.append(tgt)

    src_seq = torch.tensor(np.array(src_seq), dtype=torch.float32)
    tgt_seq = torch.tensor(np.array(tgt_seq), dtype=torch.float32)

    return src_seq, tgt_seq



## 批量处理文件文件夹中的CSV文件
def csv_process(folder_path, src_len, tgt_len):

    # 获取文件夹中的所有CSV文件
    csv_files = sorted([f for f in os.listdir(folder_path) if f.endswith(".csv")], 
                       key=lambda x: int(x.split('.')[0]))
    
    # 用于存储所有CSV文件的特征
    h_features_list = []
    v_features_list = []
    
    # 遍历所有CSV文件
    for idx, file in enumerate(csv_files):
        file_path = os.path.join(folder_path, file)
        
        # 读取CSV文件
        df = pd.read_csv(file_path)
        
        # 读取当前文件的数据
        h_signal = df.iloc[:, 0]  # 水平振动信号
        v_signal = df.iloc[:, 1]  # 垂直振动信号

        
        # 计算时域特征
        h_features = calculate_time_domain_features_label(h_signal, idx, csv_files)
        v_features = calculate_time_domain_features_label(v_signal, idx, csv_files)
       
        
        # 特征存储
        h_features_list.append(h_features)
        v_features_list.append(v_features)

    h_src, h_tgt  = create_sequences(pd.DataFrame(h_features_list), src_len, tgt_len)
    v_src, v_tgt = create_sequences(pd.DataFrame(v_features_list), src_len, tgt_len)

    # 返回所有文件的特征数据
    return h_src, h_tgt, v_src, v_tgt


## 批量处理同一工况中的文件夹（例如：bearing1_1，bearing1_5）
def folder_process(folder_list, src_len, tgt_len):
    
    h_signal_srcall = []
    h_signal_tgtall = []
    v_signal_srcall = []
    v_signal_tgtall = []

    for folder_path in folder_list:
        h_signal_src, h_signal_tgt, v_signal_src, v_signal_tgt = csv_process(folder_path, src_len, tgt_len)
        h_signal_srcall.append(h_signal_src)
        h_signal_tgtall.append(h_signal_tgt)
        v_signal_srcall.append(v_signal_src)
        v_signal_tgtall.append(v_signal_tgt)

    return torch.cat(h_signal_srcall, dim=0), torch.cat(h_signal_tgtall, dim=0) \
        ,torch.cat(v_signal_srcall, dim=0), torch.cat(v_signal_tgtall, dim=0)


## 数据处理与模型的构建
train_folder_path = [
    r"D:\Mjw\desktop\轴承数据\XJTU-SY_Bearing_Datasets\Data\XJTU-SY_Bearing_Datasets\35Hz12kN\Bearing1_3",
    r"D:\Mjw\desktop\轴承数据\XJTU-SY_Bearing_Datasets\Data\XJTU-SY_Bearing_Datasets\35Hz12kN\Bearing1_1",
    r"D:\Mjw\desktop\轴承数据\XJTU-SY_Bearing_Datasets\Data\XJTU-SY_Bearing_Datasets\37.5Hz11kN\Bearing2_2",
    r"D:\Mjw\desktop\轴承数据\XJTU-SY_Bearing_Datasets\Data\XJTU-SY_Bearing_Datasets\37.5Hz11kN\Bearing2_5",
    r"D:\Mjw\desktop\轴承数据\XJTU-SY_Bearing_Datasets\Data\XJTU-SY_Bearing_Datasets\40Hz10kN\Bearing3_1"
]

test_folder_path = [
    r"D:\Mjw\desktop\轴承数据\XJTU-SY_Bearing_Datasets\Data\XJTU-SY_Bearing_Datasets\35Hz12kN\Bearing1_2",
    r"D:\Mjw\desktop\轴承数据\XJTU-SY_Bearing_Datasets\Data\XJTU-SY_Bearing_Datasets\37.5Hz11kN\Bearing2_4",
    r"D:\Mjw\desktop\轴承数据\XJTU-SY_Bearing_Datasets\Data\XJTU-SY_Bearing_Datasets\40Hz10kN\Bearing3_5"
]



## 模型训练

if __name__ == '__main__':

    Epoch = 50
    batch_size = 32
    lr = 0.0001
    src_len = 5  
    tgt_len = 6   
    pre_len = 5  

    features_num = 6
    embedding_dim = 128
    num_layers = 3
    num_heads = 4
    d_ff = 512
    max_len = 512
    dropout = 0.2

    # 处理数据
    h_signal_srctrain, h_signal_tgttrain, \
        v_signal_srctrain, v_signal_tgttrain= folder_process(train_folder_path, src_len, tgt_len)
    
    h_signal_srctest, h_signal_tgttest, \
        v_signal_srctest, v_signal_tgttest  = folder_process(test_folder_path, src_len, tgt_len)

    # print(h_signal_train.values.shape)
    # print(h_signal_test.values.shape)

    # 构建训练集
    dataset_train = TensorDataset(h_signal_srctrain, h_signal_tgttrain)
    train_loader = DataLoader(dataset_train, batch_size=32, shuffle=False)

    # 构建测试集
    dataset_test = TensorDataset(h_signal_srctest, h_signal_tgttest)
    test_loader = DataLoader(dataset_test, batch_size=32, shuffle=False)



    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Transformer(features_num, embedding_dim, num_layers, num_heads, d_ff, max_len,\
                         dropout).to(device)
    criterion = nn.MSELoss()  # 均方误差用于回归
    optimizer = optim.Adam(model.parameters(), lr=lr)


    #开始训练
    for epoch in range(1, Epoch + 1):

        train(model, train_loader, criterion, optimizer, epoch, device, pre_len)

    # 模型测试
    test(model, test_loader, criterion, device, pre_len)
    # 训练模型
    # train_transformer(model, train_data, embedding_dim, num_epochs=5) 