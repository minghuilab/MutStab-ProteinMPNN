import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import pickle as p
import torch.optim as optim
import os
import time
from torch import Tensor
from torch.utils.data import DataLoader 
from torch.utils.data import Dataset

import pandas as pd
import numpy as np
import os, re
from collections import defaultdict,Counter,deque
from multiprocessing.dummy import Pool
from string import ascii_uppercase
from string import ascii_lowercase
import subprocess, sys, getopt
import math, copy
import linecache
from itertools import combinations
import random
import matplotlib.pyplot as plt
from scipy import stats
from joblib import Parallel, delayed
import joblib, json
import ast
import time
import warnings
# from pandarallel import pandarallel
warnings.filterwarnings('ignore')

from torch import Tensor
from torch.utils.data import DataLoader 
from torch.utils.data import Dataset

## 读取数据，得到label标签
workdir = '/public/home/zff/'

f4038_scan = pd.read_csv('/public/home/zff/premps_deep/f4038scan.txt',sep='\t')
f2648_scan = pd.read_csv('/public/home/zff/premps_deep/f2648scan.txt',sep='\t')
# mega_scan = pd.read_csv('/public/home/zff/premps_deep/megascan.txt',sep='\t')
# dst_scan = pd.read_csv('/public/home/zff/premps_deep/dstscan.txt',sep='\t')

s4038 = f4038_scan[f4038_scan['data_label'] == True]
s2648 = f2648_scan[f2648_scan['data_label'] == True]
s4038_2648_keys = list(s4038['key']) + list(s2648['key'])

s4038_keys = [key.split('_')[0] + '_' + key.split('_')[1][1] +'_'+key.split('_')[1][:1] + key.split('_')[1][2:] for key in list(s4038['key'])]

s4038_label = {}
for key in list(s4038['key']):
    ddG = s4038[s4038['key']==key]['ddG'].values[0]
    if ddG >=0:
        label = 0
    if ddG < 0:
        label = 1
    key = key.split('_')[0] + '_' + key.split('_')[1][1] +'_'+key.split('_')[1][:1] + key.split('_')[1][2:]
    s4038_label[key] = label

s2648_keys = [key.split('_')[0] + '_' + key.split('_')[1][1] +'_'+key.split('_')[1][:1] + key.split('_')[1][2:] for key in list(s2648['key'])]

s2648_label = {}
for key in list(s2648['key']):
    ddG = s2648[s2648['key']==key]['ddG'].values[0]
    if ddG >=0:
        label = 0
    if ddG < 0:
        label = 1
    key = key.split('_')[0] + '_' + key.split('_')[1][1] +'_'+key.split('_')[1][:1] + key.split('_')[1][2:]
    s2648_label[key] = label

f4038_scan = f4038_scan[f4038_scan['data_label'] == False]
f2648_scan = f2648_scan[f2648_scan['data_label'] == False]

f2648_scan = f2648_scan[~f2648_scan['key'].isin(f4038_scan['key'])]

f4038_scan_sta_keys = list(f4038_scan[f4038_scan['sta_label'] == True]['key'])
f2648_scan_sta_keys = list(f2648_scan[f2648_scan['sta_label'] == True]['key'])

len(f4038_scan_sta_keys),len(f2648_scan_sta_keys)

f4038_scan_desta_keys = list(f4038_scan[f4038_scan['desta_label'] == True]['key'])
f2648_scan_desta_keys = list(f2648_scan[f2648_scan['desta_label'] == True]['key'])

len(f4038_scan_desta_keys),len(f2648_scan_desta_keys)

sta_key = f4038_scan_sta_keys + f2648_scan_sta_keys

desta_key = f4038_scan_desta_keys + f2648_scan_desta_keys

len(sta_key),len(desta_key)

mega_scan = pd.read_csv(workdir + 'premps_deep/mega_single_final.txt',sep='\t')
mega_scan = mega_scan[mega_scan['mut_type'] != 'wt']
mega_scan['ddG'] = mega_scan['ddG_ML']


mega_scan['key'] = mega_scan['name'].str.split('.').str[0].str.replace('_','').str.upper() +'_A_' + mega_scan['mut_type']
mega_scan['esm_key'] =  mega_scan['name'].str.split('.').str[0].str.replace('_','-') +'_A_' + mega_scan['mut_type']
key_esm_dict = dict(zip(mega_scan['key'], mega_scan['esm_key']))

mega_keys = list(mega_scan['key'])

dst_scan = pd.read_csv(workdir + 'premps_deep/dstscan.txt',sep='\t')
dst_scan['key'] = dst_scan['pdbid'].str.upper() + '_A_' + dst_scan['Mutation_PDB']
dst_scan['esm_key'] = dst_scan['pdbid'] + '_A_' + dst_scan['Mutation_PDB']
dst_scan['hand_key'] = dst_scan['pdbid'].str.replace('_','').str.upper() + '_A_' + dst_scan['Mutation_PDB']
key_esm_dict_dst = dict(zip(dst_scan['key'], dst_scan['esm_key']))
key_hand_dict_dst = dict(zip(dst_scan['key'], dst_scan['hand_key']))

dst_keys = list(dst_scan['key'])

# 创建一个空的标签字典
from joblib import Parallel, delayed
from tqdm import tqdm
mega_label = {}

# 遍历 DataFrame 中的每一行
for index, row in tqdm(mega_scan.iterrows(), total=len(mega_scan), desc='Processing'):
    # 提取当前行对应的键和 ddG 值
    key = row['key']
    ddG = row['ddG']
    # 使用条件判断来确定标签值
    label = 0 if ddG >= 0 else 1
    # 将标签添加到标签字典中
    mega_label[key] = label


# 创建一个空的标签字典
dst_label = {}

# 遍历 DataFrame 中的每一行
for index, row in tqdm(dst_scan.iterrows(), total=len(dst_scan), desc='Processing'):
    # 提取当前行对应的键和 ddG 值
    key = row['key']
    ddG = row['ddG']
    
    # 使用条件判断来确定标签值
    label = 0 if ddG >= 0 else 1
    # 将标签添加到标签字典中
    dst_label[key] = label
    

label_dict = {}
for key in sta_key:
    key = key.split('_')[0] + '_' + key.split('_')[1][1] +'_'+key.split('_')[1][:1] + key.split('_')[1][2:]
    label_dict[key] = 1
for key in desta_key:
    key = key.split('_')[0] + '_' + key.split('_')[1][1] +'_'+key.split('_')[1][:1] + key.split('_')[1][2:]
    label_dict[key] = 0
    
label_dict.update(s4038_label)
label_dict.update(s2648_label)
label_dict.update(mega_label)
label_dict.update(dst_label)
print('label_dict ready')


## 
import pickle
## ['esm','hand','mpnn']
fea_name = 'esm'
path_save = '/public/home/zff/premps_deep/model_result/mlp_te2_esm2_3b/'
jobpath = '/public/home/zff/premps_deep/'

feadim = 2560

# with open(jobpath+f'all_torch_protein_{fea_name}.pkl', 'rb') as f:
#     all_torch_protein_mpnn = pickle.load(f)
with open(jobpath+f'all_torch_mutation_{fea_name}.pkl', 'rb') as f:
    all_torch_mutation = pickle.load(f)
# with open(jobpath+f'all_torch_wt_neigh_{fea_name}.pkl', 'rb') as f:
#     all_torch_wt_neigh_mpnn = pickle.load(f)
# with open(jobpath+f'all_torch_mut_neigh_{fea_name}.pkl', 'rb') as f:
#     all_torch_mut_neigh_mpnn = pickle.load(f)



## 让我想想，改写Dataset , 
class CustomDataset(Dataset):

    def __init__(self, keys):
        self.keys = keys
        
    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):

        key = self.keys[index]

        mutation = all_torch_mutation[key].to(torch.float32)          
        
        labels = torch.from_numpy(np.array(label_dict[key])).to(torch.float32)
        
        sample = {
                  'mutation':mutation,
                  'labels': labels, \
                  'key': key, \
                  }
        return sample

class CustomDataset_mega(Dataset):

    def __init__(self, keys):
        self.keys = keys
        
    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        hand_path = workdir+'embedding/hand/process/mega/'
        esm_path = workdir+'embedding/esm2_3b/process/mega/'
        mpnn_path = workdir+'embedding/proteinmpnn/process/mega/'
        
        key = self.keys[index]
        if fea_name in ['mpnn']:
            embedding = torch.load(f'{mpnn_path}{key}.pt')
        if fea_name in ['esm']:
            embedding = torch.load(f'{esm_path}{key_esm_dict[key]}.pt')
        if fea_name in ['hand']:
            embedding = torch.load(f'{hand_path}{key}.pt')       

        mutation= embedding['mutation'].to(torch.float32)

        labels = torch.from_numpy(np.array(label_dict[key])).to(torch.float32)
        
        sample = {

                  'mutation':mutation,

                  'labels': labels, \
                  'key': key, \
                  }
        return sample

class CustomDataset_dst(Dataset):

    def __init__(self, keys):
        self.keys = keys
        
    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        hand_path = workdir+'embedding/hand/process/dst/'
        esm_path = workdir+'embedding/esm2_3b/process/dst/'
        mpnn_path = workdir+'embedding/proteinmpnn/process/dst/'
        
        key = self.keys[index]
        
        if fea_name in ['mpnn']:
            embedding = torch.load(f'{mpnn_path}{key}.pt')
        if fea_name in ['esm']:
            embedding = torch.load(f'{esm_path}{key_esm_dict_dst[key]}.pt')
        if fea_name in ['hand']:
            embedding = torch.load(f'{hand_path}{key_hand_dict_dst[key]}.pt')  

        mutation = embedding['mutation'].to(torch.float32)

        labels = torch.from_numpy(np.array(label_dict[key])).to(torch.float32)
        
        sample = {
                
                  'mutation':mutation,
                  'labels': labels, \
                  'key': key, \
                  }
        return sample
import torch
import torch.nn as nn
import numpy as np
class MultiHeadSelfAttention(nn.Module):
    # 类的构造方法
    def __init__(self, embed_dim, num_heads=8):
        #继承自MultiHeadSelfAttention父类
        super(MultiHeadSelfAttention, self).__init__()
        # 输入embedding特征的维度(每个残基的embedding特征的维度)
        self.embed_dim = embed_dim
        # 多头注意力的头数heads
        self.num_heads = num_heads
        # 注意：embedding的维度，应该能被heads整除？？？？？？？？？？？？？？？？？？？？？？？？？？？
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embedding dimension = {embed_dim} should be divisible by number of heads = {num_heads}"
            )
        # 初始化self.projection_dim？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？
        self.projection_dim = embed_dim // num_heads
        #构建计算q、k、v的线性层Wq、Wk、Wv
        self.query_dense = nn.Linear(embed_dim, embed_dim)
        self.key_dense = nn.Linear(embed_dim, embed_dim)
        self.value_dense = nn.Linear(embed_dim, embed_dim)
        # 线性变化层：用于将多头注意力的输出合并成最终输出？？？？？？？？？？？？？？？？？？？？？
        self.combine_heads = nn.Linear(embed_dim, embed_dim)

    # 注意力attention计算方法，即上面图片中的公式过程
    def attention(self, query, key, value, mask=None):
        # torch.matmul执行矩阵乘法，计算自注意力分数矩阵，用于衡量每个位置的权重
        # transpose(-2, -1)将key张量的最后两个维度进行调换，相当于转置T
        score = torch.matmul(query, key.transpose(-2, -1))
        # 计算张量最后一个维度的大小（每个残基的特征维度）
        # key.shape[-1]表示key张量，的特征维度（每个残基的特征维度），将最后一个特征维度保存为torch.float32的形式
        dim_key = torch.tensor(key.shape[-1], dtype=torch.float32)
        
        # 在计算自注意力机制中，对注意力分数进行缩放，目的是控制注意力分数的范围，使其更加稳定，方便训练
        scaled_score = score / torch.sqrt(dim_key)
        
        # 在自注意力机制中，应用掩码操作（mask），在某些位置上抑制或屏蔽注意力
        # mask：一个与注意力分数矩阵相同形状的二元掩码矩阵，1表示屏蔽，0表示保留？？？？？？？？？？？？？？？
        # scaled_score：经过缩放后的原始注意力分数矩阵
        if mask is not None:
            scaled_score += (mask * -1e9)
        
        # 对缩放后的注意力分数，进行softmax操作，以计算注意力权重
        # 再将softmax后的权重矩阵，与value（v）做矩阵乘法
        weights = F.softmax(scaled_score, dim=-1)
        output = torch.matmul(weights, value)
        
        # 最终输出注意力分数Z和权重矩阵
        return output, weights
    
    # 将输入张量重塑形状，以便在多头注意力机制中分离不同的注意力头（在多头注意力中，输入张量通常被分成多个部分，每个部分对应一个注意力头）
    # view：将x张量，重塑为其他形状，但是要求新形状的元素总数和原张量相同
    # 交换张量的维度，使得到的最终形状为(0, 2, 1, 3) = (batch_size, num_heads, seq_len, projection_dim)
    def separate_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.projection_dim)
        return x.permute(0, 2, 1, 3)

    # forward函数：运行多头注意力的过程
    # 输入：inputs是输入张量，mask掩码，默认是没有掩码None
    def forward(self, inputs, mask=None):
        # x.shape = [batch_size, seq_len, embedding_dim]
        #获取输入张量的批次batch_size，即这一批有多少个蛋白质（多少个序列）
        batch_size = inputs.size(0)
        
        #计算q、k、v
        query = self.query_dense(inputs)  # (batch_size, seq_len, embed_dim)
        key = self.key_dense(inputs)  # (batch_size, seq_len, embed_dim)
        value = self.value_dense(inputs)  # (batch_size, seq_len, embed_dim)
        
        # 所有蛋白质的q在同一个张量中，此步骤将不同蛋白质的q、k、v分离出来
        query = self.separate_heads(query, batch_size)  # (batch_size, num_heads, seq_len, projection_dim)
        key = self.separate_heads(key, batch_size)  # (batch_size, num_heads, seq_len, projection_dim)
        value = self.separate_heads(value, batch_size)  # (batch_size, num_heads, seq_len, projection_dim)
        
        # 计算多头注意力的结果，和权重矩阵
        attention, weights = self.attention(query, key, value, mask)
        # 对得到的结果矩阵的张量，进行重塑
        attention = attention.permute(0, 2, 1, 3)  # (batch_size, seq_len, num_heads, projection_dim)
        # 对多头注意力得到的结果进行reshape，再将reshape后的多头注意力concat在一起
        # concat后得到的输出结果，就是多头自注意力的最终输出（维度不变）
        concat_attention = attention.reshape(batch_size, -1, self.embed_dim)  # (batch_size, seq_len, embed_dim)
        output = self.combine_heads(concat_attention)  # (batch_size, seq_len, embed_dim)
        return output

    # 为自定义的 Keras 层提供配置信息，使得在保存和加载模型时，能够保存和恢复该层的参数。
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'embed_dim': self.embed_dim,
        })
        return config

class transformer_block(nn.Module):
    def __init__(self, input_dim,output_dim, head,dropout_rate):
        # 调用父类cnn1d_block
        super(transformer_block, self).__init__()
        # 定义一个多头注意力的类class，只需输入两个参数
        # 输入特征的总维度（每个头的维度dim1* 头数head）
        # head：多头注意力的头数，可以通过设置head的有无，选择是否加入多头注意力模块
        self.attention1 = MultiHeadSelfAttention(input_dim,head) if head is not None else None     
        
        # 层归一化layerNorm
        # eps=1e-6 : 为了稳定性
        self.layer_norm1 = nn.LayerNorm(normalized_shape=input_dim , eps = 1e-06)
        self.layer_norm2 = nn.LayerNorm(normalized_shape=input_dim , eps = 1e-06)
        self.fnn = nn.Sequential(nn.Linear(input_dim, output_dim),nn.ReLU(),nn.Linear(output_dim, output_dim))
        
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.dropout2 = nn.Dropout(p=dropout_rate)
        
    def forward(self,x):
        x1 = self.attention1(x)
        x1 = self.dropout1(x1)
        x2 = self.layer_norm1(x + x1)
        x3 = self.fnn(x2)
        x3 = self.dropout2(x3)
        x4 = self.layer_norm2(x2 + x3)
        return x4
class TokenPositionEmbedding(nn.Module):
    def __init__(self, maxlen, pos_embed_dim):
        super().__init__()
        self.maxlen = maxlen
        self.pos_embed_dim = pos_embed_dim
        self.pos_emb = nn.Embedding(maxlen, pos_embed_dim)
        self.init_weights()

    def init_weights(self):
        position_enc = np.array([
            [pos / np.power(10000, 2 * (i // 2) / self.pos_embed_dim) for i in range(self.pos_embed_dim)]
            if pos != 0 else np.zeros(self.pos_embed_dim)
            for pos in range(self.maxlen)
        ])
        position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2])  # dim 2i
        position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2])  # dim 2i+1
        self.pos_emb.weight.data.copy_(torch.tensor(position_enc))

    def forward(self, x):
        batch_size, seq_len, input_dim = x.size()
        positions = torch.arange(0, seq_len).unsqueeze(0).repeat(batch_size, 1).to(x.device)
        position_emb = self.pos_emb(positions)
        x1 = x + position_emb
        return x1

class cnn1d_Residual_block(nn.Module):
    def __init__(self, in_channel, out_channel,out_channel2, relu_par0, pool_kernel_size, stride=1):
        super(cnn1d_Residual_block, self).__init__()
        self.conv1 = nn.Conv1d(in_channel, out_channel, 3, stride=stride, padding=1)
        nn.init.kaiming_uniform_(self.conv1.weight, mode='fan_in', nonlinearity='relu')
        self.conv2 = nn.Conv1d(out_channel, out_channel2, 3, stride=stride, padding=1)
        nn.init.kaiming_uniform_(self.conv2.weight, mode='fan_in', nonlinearity='relu')
        self.leakyReLu1 = nn.LeakyReLU(negative_slope=relu_par0)
        self.maxpooling1 = nn.MaxPool1d(pool_kernel_size)

    def forward(self, x):
        ## x = [batch,fea,seq]
  
        x1 = x.permute(0, 2, 1)
 
        x2 = self.conv1(x1.float())

        x3 = self.leakyReLu1(x2)

        x4 = self.conv2(x3.float())

        x5 = x4.permute(0, 2, 1)

        
        # 调整x的维度使得与x5匹配
#         x_adjusted = x.permute(0, 2, 1)
        x6 = torch.add(x, x5)
        
        x7 = self.leakyReLu1(x6)
        x8 = self.maxpooling1(x7)
        x9 = x8.permute(0, 2, 1)

        return x9

class MLPs_block(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.fc1 = nn.Linear(self.embed_dim, self.embed_dim // 2)
        self.bn1 = nn.BatchNorm1d(self.embed_dim // 2)
        self.fc2 = nn.Linear(self.embed_dim // 2, self.embed_dim // 8)
        self.bn2 = nn.BatchNorm1d(self.embed_dim // 8)
        self.fc3 = nn.Linear(self.embed_dim // 8, 1)
        self.leakyReLU = nn.LeakyReLU(negative_slope=0.2)
        self.sigmoid = nn.Sigmoid()  # 使用LogSoftmax作为输出层的激活函数，方便计算交叉熵损失
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.leakyReLU(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out = self.leakyReLU(out)
        out = self.fc3(out)
        out = self.sigmoid(out)
        return out

class PremPS2(nn.Module):
    def __init__(self, input_dim=feadim):
        super().__init__()
        self.embed_dim = input_dim  # 添加这一行
        self.bn0 = nn.BatchNorm1d(input_dim)
        # 突变通道
        self.PE_mut = TokenPositionEmbedding(maxlen = input_dim, pos_embed_dim = 1)
        # Transformer模块：Multi-head attention + feed forward
        self.transa1 = transformer_block(1,1,head = 1,dropout_rate = 0.1)

        self.mlps = MLPs_block(self.embed_dim)

    def forward(self, mutation):
        # 对输入数据进行批量归一化
        mutation = self.bn0(mutation)
        mutation = torch.unsqueeze(mutation, 1)
        # mutation = mutation.unsqueeze(mutation, 1)
        mutation = mutation.permute(0,2,1)
        mutation = self.PE_mut(mutation)
#         # Transformer
        mutation = self.transa1(mutation.to(torch.float32)) 
        mutation = mutation.permute(0,2,1)
        mutation = torch.squeeze(mutation)
        
        # 使用 MLPs_block 处理批量归一化后的数据
        results1 = self.mlps(mutation)

        return results1

import math
from torch.optim.lr_scheduler import LambdaLR
def get_cosine_schedule_with_warmup(optimizer, num_training_steps,num_cycles=7. / 16.,num_warmup_steps = 0,last_epoch=-1):
#     '''
#     Get cosine scheduler (LambdaLR).
#     if warmup is needed, set num_warmup_steps (int) > 0.
#     '''
    def _lr_lambda(current_step):
        '''
        _lr_lambda returns a multiplicative factor given an interger parameter epochs.
        Decaying criteria: last_epoch
        '''
        if current_step < num_warmup_steps:
            _lr = float(current_step) / float(max(1, num_warmup_steps))
        else:
            num_cos_steps = float(current_step - num_warmup_steps)
            num_cos_steps = num_cos_steps / float(max(1, num_training_steps - num_warmup_steps))
            _lr = max(0.0, math.cos(math.pi * num_cycles * num_cos_steps))
        return _lr
    return LambdaLR(optimizer, _lr_lambda, last_epoch)

import numpy as np
import torch
'''
该脚本来自于:https://github.com/Bjarten/early-stopping-pytorch
'''
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=20, verbose=False, delta=0, trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.trace_func = trace_func
    def __call__(self, val_loss):

        score = val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss)
        elif score+self.delta > self.best_score:
            self.counter += 1
            self.trace_func(f'loss changed ({self.val_loss_min:.6f} --> {val_loss:.6f}). EarlyStopping counter: {self.counter}/{self.patience}. Saving model ...')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss)
            self.counter = 0

    def save_checkpoint(self, val_loss):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...')
        self.val_loss_min = val_loss



device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

import time
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import random
from sklearn.metrics import confusion_matrix


# 设置随机种子以确保结果的可重复性
seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True

num_epochs = 100
train_losses = []  # 保存每个 epoch 的训练损失
valid_losses = []  # 保存每个 epoch 的验证损失

train_keys=np.load('/public/home/zff/premps_deep/train_keys2.npy')
train_keys=train_keys.tolist()
valid_keys=np.load('/public/home/zff/premps_deep/valid_keys2.npy')
valid_keys=valid_keys.tolist()

print(len(train_keys),len(valid_keys))

train_dataset = CustomDataset(train_keys)
valid_dataset = CustomDataset(valid_keys)
train_dataloader = DataLoader(train_dataset, batch_size=512, shuffle=True,num_workers=40)
valid_dataloader = DataLoader(valid_dataset, batch_size=512, shuffle=True,num_workers=40)
loss_fn = nn.BCELoss()

model = PremPS2().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.00001)
scheduler = get_cosine_schedule_with_warmup(optimizer,num_epochs)


# 寻找最新的模型文件
latest_epoch = 0
latest_model_path = None
for file in os.listdir(os.path.join(path_save, 'models')):
    if file.endswith(".pt"):
        epoch = int(file.split("_")[2].split(".")[0])
        if epoch > latest_epoch:
            latest_epoch = epoch
            latest_model_path = os.path.join(path_save, 'models', file)

if latest_model_path is not None:
    # 加载最新的模型
    print(f'load model {latest_model_path}')
    model.load_state_dict(torch.load(latest_model_path, map_location=torch.device('cpu')))
    start_epoch = latest_epoch + 1
    print(f"Resuming from epoch {start_epoch}")

else:
    start_epoch = 0
    print("No previous model checkpoint found. Starting from epoch 0")

early_stopping = EarlyStopping(20, verbose=True, delta=0)
for epoch in range(start_epoch, num_epochs):
    st = time.time()
    model.train()
    epoch_train_loss = 0.0  # 保存当前 epoch 的总训练损失
    for i_batch, sample in enumerate(train_dataloader):
        model.zero_grad()
        labels = sample['labels'].to(device)
        key = sample['key']

        mutation = sample['mutation'].to(device)
        # mutation = torch.unsqueeze(mutation, 1)
    

        logits = model(mutation)
        
        pred = torch.squeeze(logits, 1)
        loss = loss_fn(pred, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()
        epoch_train_loss += loss.item() * mutation.size(0)  # 累加当前批次的损失

    # 计算当前 epoch 的平均训练损失
    epoch_train_loss /= len(train_dataset)
    train_losses.append(epoch_train_loss)
    # 保存模型
    torch.save(model.state_dict(), f"{path_save}/models/model_epoch_{epoch}.pt")
    # 计算验证集损失和指标
    model.eval()
    with torch.no_grad():
        epoch_valid_loss = 0.0  # 保存当前 epoch 的总验证损失
        valid_preds = []
        valid_labels = []
        for sample in valid_dataloader:
            labels = sample['labels'].to(device)
            valid_labels.extend(labels.cpu().numpy())
            

            mutation = sample['mutation'].to(device)
            # mutation = torch.unsqueeze(mutation, 1)
            
            logits = model(mutation)

            pred = torch.squeeze(logits, 1)
            loss = loss_fn(pred, labels)
            epoch_valid_loss += loss.item() * mutation.size(0)  # 累加当前批次的损失
            pred_binary = (pred >= 0.5).int()
            valid_preds.extend(pred_binary.cpu().numpy())

        epoch_valid_loss /= len(valid_dataset)
        valid_losses.append(epoch_valid_loss)

        valid_preds = np.array(valid_preds)
        valid_labels = np.array(valid_labels)
        early_stopping(epoch_valid_loss)
        if early_stopping.early_stop:
            print("Early stopping:" + str(epoch + 1))
            break

        tn, fp, fn, tp = confusion_matrix(valid_labels, valid_preds).ravel()
        tpr = tp / (tp + fn)
        fpr = fp / (fp + tn)
        precision = tp / (tp + fp)
        recall = tpr
        f1_score = 2 * (precision * recall) / (precision + recall)
        mcc = (tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

        print(f"Epoch [{epoch + 1}/{num_epochs}], Valid Loss: {epoch_valid_loss:.4f}, TPR: {tpr:.4f}, FPR: {fpr:.4f}, MCC: {mcc:.4f}, Precision: {precision:.4f}, F1_score: {f1_score:.4f}, Time: {time.time() - st:.2f}s")

a=np.array(train_losses)
np.save(f'{path_save}/train_loss.npy',a) # 保存为.npy格式
b = np.array(valid_losses)
np.save(f'{path_save}/valid_losses.npy',b) # 保存为.npy格式



## 测试：
import os
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import random

# 设置随机种子
seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True

# 定义模型加载函数
def load_model(model_path, device):
    model = PremPS2()  # 创建模型实例
    model.load_state_dict(torch.load(model_path))  # 加载模型参数
    model.to(device)
    model.eval()
    return model

# 获取指定路径下的所有.pt文件
def get_model_files(path):
    return [file for file in os.listdir(path) if file.endswith('.pt')]

# 定义测试函数
def test_model(model, test_dataloader, device):
    pred_data = []
    label_data = []
    key_data = []
    for i_batch, sample in enumerate(test_dataloader):
        labels = sample['labels'].to(device)
        key = sample['key']

        mutation = sample['mutation'].to(device) 


        logits = model(mutation)
        
        pred = torch.squeeze(logits, 1)
        pred_data.extend(pred.cpu().detach().numpy())
        label_data.extend(labels.cpu().detach().numpy())
        key_data.extend(key)
    return key_data, pred_data, label_data

# 计算指标
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, f1_score, precision_score, recall_score, matthews_corrcoef
def calculate_metrics(df):
    tpr = recall_score(df['Actual'], df['Predicted'] > 0.5)
    tn, fp, fn, tp = confusion_matrix(df['Actual'], df['Predicted'] > 0.5).ravel()
    fpr = fp / (fp + tn)
    mcc = matthews_corrcoef(df['Actual'], df['Predicted'] > 0.5)
    # fpr, tpr, _ = roc_curve(df['Actual'], df['Predicted'])
    auc_roc = roc_auc_score(df['Actual'], df['Predicted'])
    f1 = f1_score(df['Actual'], df['Predicted'] > 0.5)
    precision = precision_score(df['Actual'], df['Predicted'] > 0.5)
    return tpr, fpr, mcc, auc_roc, f1, precision



# 获取模型文件列表
model_files = get_model_files(f'{path_save}/models/')
epoch = max([int(model_file.split('_')[-1].split('.')[0]) for model_file in model_files])

print(f'final epoch : {epoch}')

# 存储所有指标的列表
metrics_list_4038 = []
metrics_list_2648 = []
metrics_list_mega = []
metrics_list_dst = []

model_path = f'{path_save}/models/model_epoch_{epoch}.pt'
model = load_model(model_path, device)

## S4038
test_dataset = CustomDataset(s4038_keys)
test_dataloader = DataLoader(test_dataset, batch_size=512, shuffle=True, num_workers=40)
keys, preds, labels = test_model(model, test_dataloader, device)

# 创建当前 epoch 的结果 DataFrame
result_df = pd.DataFrame({'Epoch': [epoch]*len(keys),
                          'Key': keys,
                          'Predicted': preds,
                          'Actual': labels})

# 将结果 DataFrame 写入到文件
result_df.to_csv( f'{path_save}/result/S4038_result_epoch_{epoch}.csv',sep = '\t', index=False)

# 计算指标
tpr, fpr, mcc, auc_roc, f1, precision = calculate_metrics(result_df)

# 将指标存储到列表中
metrics_list_4038.append({'Epoch': epoch,
                     'TPR': tpr,
                     'FPR': fpr,
                     'MCC': mcc,
                     'AUC-ROC': auc_roc,
                     'F1-Score': f1,
                     'Precision': precision})
## S2648
test_dataset = CustomDataset(s2648_keys)
test_dataloader = DataLoader(test_dataset, batch_size=512, shuffle=True, num_workers=40)
keys, preds, labels = test_model(model, test_dataloader, device)

# 创建当前 epoch 的结果 DataFrame
result_df = pd.DataFrame({'Epoch': [epoch]*len(keys),
                          'Key': keys,
                          'Predicted': preds,
                          'Actual': labels})

# 将结果 DataFrame 写入到文件
result_df.to_csv( f'{path_save}/result/S2648_result_epoch_{epoch}.csv',sep = '\t', index=False)

# 计算指标
tpr, fpr, mcc, auc_roc, f1, precision = calculate_metrics(result_df)

# 将指标存储到列表中
metrics_list_2648.append({'Epoch': epoch,
                     'TPR': tpr,
                     'FPR': fpr,
                     'MCC': mcc,
                     'AUC-ROC': auc_roc,
                     'F1-Score': f1,
                     'Precision': precision})


##测试mega

## mega
test_dataset = CustomDataset_mega(mega_keys)
test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=True, num_workers=40)
keys, preds, labels = test_model(model, test_dataloader, device)

# 创建当前 epoch 的结果 DataFrame
result_df = pd.DataFrame({
                          'Key': keys,
                          'Predicted': preds,
                          'Actual': labels})

# 将结果 DataFrame 写入到文件
result_df.to_csv(f'{path_save}/result/mega_result_epoch_{epoch}.csv',sep = '\t', index=False)

# 计算指标
tpr, fpr, mcc, auc_roc, f1, precision = calculate_metrics(result_df)

# 将指标存储到列表中
metrics_list_mega.append({
                     'TPR': tpr,
                     'FPR': fpr,
                     'MCC': mcc,
                     'AUC-ROC': auc_roc,
                     'F1-Score': f1,
                     'Precision': precision})
## dst
test_dataset = CustomDataset_dst(dst_keys)
test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=True, num_workers=40)
keys, preds, labels = test_model(model, test_dataloader, device)

# 创建当前 epoch 的结果 DataFrame
result_df = pd.DataFrame({
                          'Key': keys,
                          'Predicted': preds,
                          'Actual': labels})

# 将结果 DataFrame 写入到文件
result_df.to_csv( f'{path_save}/result/dst_result_epoch_{epoch}.csv',sep = '\t', index=False)

# 计算指标
tpr, fpr, mcc, auc_roc, f1, precision = calculate_metrics(result_df)

# 将指标存储到列表中
metrics_list_dst.append({
                     'TPR': tpr,
                     'FPR': fpr,
                     'MCC': mcc,
                     'AUC-ROC': auc_roc,
                     'F1-Score': f1,
                     'Precision': precision})
# metrics_df_mega = pd.DataFrame(metrics_list_mega)
metrics_df_dst = pd.DataFrame(metrics_list_dst)
# metrics_df_mega.to_csv(os.path.join(workdir, f'premps_deep/model_result/three/mega_all_epochs_metrics.txt'),sep = '\t', index=False)
# metrics_df_dst.to_csv( f'{path_save}/dst_all_epochs_metrics.txt',sep = '\t', index=False)