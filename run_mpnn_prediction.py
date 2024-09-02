#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import re
import numpy as np
import pandas as pd
import torch
import torch.utils.data.dataset as Dataset
from pathlib import Path
import torch.utils.data.dataloader as DataLoader
import os
from torch import nn
import torch.nn.functional as F
from typing import List,Dict,Union
MultiHeadAttention = nn.MultiheadAttention
from loguru import logger
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from copy import deepcopy
import sys
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
from sklearn.model_selection import train_test_split
from typing import List
from pathlib import Path
import yaml
import argparse

parser = argparse.ArgumentParser(description='Process structure to predict the mutation.')
parser.add_argument('-p', '--pdbfile', type=str, help='Input the PDB file')
parser.add_argument('-m', '--mutation', type=str, help='Mutation need to Predict')
parser.add_argument('-f', '--featurePath', type=str, help='The ProteinMPNN feature file for the PDB')
parser.add_argument('-o', '--output', type=str, help='Path to save the predition file')

args = parser.parse_args()
pdb_file_path = args.pdbfile
pdb_file = pdb_file_path.split('/')[-1]
mut_chain = args.mutation
chain = mut_chain.split('_')[1]
mut = mut_chain.split('_')[0]
fea_path = args.featurePath
output_path = args.output

have_line = []
with open(pdb_file_path,'r') as f:
    for line1 in f:
        if line1[:4] == 'ATOM':
            have_line.append(line1)
have_line1 = [[x[17:20].strip(),x[20:22].strip(),x[22:27].strip()] for x in have_line]
have_cols = ['Res','Chain','Pos']
have_dat = pd.DataFrame(have_line1, columns=have_cols)
pdb_dat = have_dat.drop_duplicates('Pos').reset_index(drop=True)
pdb_dat['clean_Pos'] = [x+1 for x in range(len(pdb_dat))]
pos = mut[1:-1]
newpos = pdb_dat[pdb_dat['Pos']==pos].reset_index(drop=True)['clean_Pos'][0]
clean_mut = mut[0] + str(newpos) + mut[-1]

feafile = dict(np.load(fea_path + pdb_file.split('.')[0] + '.npz'))
ref_hid_emb = feafile['hid'].reshape(1,feafile['hid'].shape[1],-1)
ref_embedding = np.squeeze(np.concatenate((feafile['embedding'],ref_hid_emb),axis=-1))

mutfile = dict(np.load(fea_path + pdb_file.split('.')[0] + '_' + chain + '_' + mut + '.npz'))
mut_hid_emb = mutfile['hid'].reshape(1,mutfile['hid'].shape[1],-1)
mut_embedding = np.squeeze(np.concatenate((mutfile['embedding'],mut_hid_emb),axis=-1))

def pad_three(arr,p,q,numi):
    pad_shape0 = np.zeros((p,numi))
    pad_shape1 = np.zeros((q,numi))
    arr1 = np.concatenate((pad_shape0,arr,pad_shape1),axis=0)
    return arr1

def get_need_refmut_three(arr,mutpos,need_len,p,q,numi):
    if mutpos < need_len and len(arr) - mutpos-1 >= need_len:
        return pad_three(arr[:mutpos + need_len + 1,:],p,q,numi)
    elif mutpos < need_len and len(arr) - mutpos-1 < need_len:
        return pad_three(arr[:,:],p,q,numi)
    elif mutpos >= need_len and len(arr) - mutpos-1 >= need_len:
        return arr[mutpos-need_len:mutpos+need_len + 1,:]
    else:
        return pad_three(arr[mutpos - need_len:,:],p,q,numi)


if int(newpos) <= 11:
    pad_p = 11 + 1 - int(newpos)
else:
    pad_p = 0
if feafile['hid'].shape[1] - int(newpos) - 1 - 11 >= 0:
    pad_q = 11 - (feafile['hid'].shape[1] - int(newpos))
else:
    pad_q = 0

ref_embedding_need = get_need_refmut_three(ref_embedding,int(newpos),11,pad_p,pad_q,512)
mut_embedding_need = get_need_refmut_three(mut_embedding,int(newpos),11,pad_p,pad_q,512)

#######################################
# run_Prediction

# module
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=8):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embedding dimension = {embed_dim} should be divisible by number of heads = {num_heads}"
            )
        self.projection_dim = embed_dim // num_heads
        self.query_dense = nn.Linear(embed_dim, embed_dim)
        self.key_dense = nn.Linear(embed_dim, embed_dim)
        self.value_dense = nn.Linear(embed_dim, embed_dim)
        self.combine_heads = nn.Linear(embed_dim, embed_dim)

    def attention(self, query, key, value, mask=None):
        score = torch.matmul(query, key.transpose(-2, -1))
        dim_key = torch.tensor(key.shape[-1], dtype=torch.float32)
        scaled_score = score / torch.sqrt(dim_key)
        if mask is not None:
            scaled_score += (mask * -1e9)
        weights = F.softmax(scaled_score, dim=-1)
        output = torch.matmul(weights, value)
        return output, weights
    def separate_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.projection_dim)
        return x.permute(0, 2, 1, 3)

    def forward(self, inputs, mask=None):
        batch_size = inputs.size(0)
        query = self.query_dense(inputs)  # (batch_size, seq_len, embed_dim)
        key = self.key_dense(inputs)  # (batch_size, seq_len, embed_dim)
        value = self.value_dense(inputs)  # (batch_size, seq_len, embed_dim)
        query = self.separate_heads(query, batch_size)  # (batch_size, num_heads, seq_len, projection_dim)
        key = self.separate_heads(key, batch_size)  # (batch_size, num_heads, seq_len, projection_dim)
        value = self.separate_heads(value, batch_size)  # (batch_size, num_heads, seq_len, projection_dim)
        attention, weights = self.attention(query, key, value, mask)
        attention = attention.permute(0, 2, 1, 3)  # (batch_size, seq_len, num_heads, projection_dim)
        concat_attention = attention.reshape(batch_size, -1, self.embed_dim)  # (batch_size, seq_len, embed_dim)
        output = self.combine_heads(concat_attention)  # (batch_size, seq_len, embed_dim)
        return output

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'embed_dim': self.embed_dim,
        })
        return config

class transformer_block(nn.Module):
    def __init__(self, input_dim,output_dim, head,dropout_rate):
        super(transformer_block, self).__init__()
        self.attention1 = MultiHeadSelfAttention(input_dim,head) if head is not None else None     
        self.layer_norm1 = nn.LayerNorm(normalized_shape=input_dim , eps = 1e-06)
        self.layer_norm2 = nn.LayerNorm(normalized_shape=input_dim , eps = 1e-06)
        self.fnn = nn.Sequential(nn.Linear(input_dim, output_dim),nn.ReLU(),nn.Linear(output_dim, output_dim))
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.dropout2 = nn.Dropout(p=dropout_rate)
        
    def forward(self,x,mask):
        x1 = self.attention1(x,mask)
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
        self.sigmoid = nn.Sigmoid()
    
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
    def __init__(self, input_dim_neigh = 512):
        super().__init__() 
        self.bn0 = nn.BatchNorm1d(input_dim_neigh)
        self.PE_neigh = TokenPositionEmbedding(23,pos_embed_dim = input_dim_neigh)
        self.transb1 = transformer_block(input_dim_neigh,input_dim_neigh,head = 2,dropout_rate=0.1)
        self.transb2 = transformer_block(input_dim_neigh,input_dim_neigh,head= 2,dropout_rate=0.1)
        relu_par0 = 0.1
        self.cnnb1 = cnn1d_Residual_block(23,23,23,relu_par0, pool_kernel_size = 23)
        self.in2_channel  = input_dim_neigh 
        self.mlps = MLPs_block(self.in2_channel)

    def get_attn_pad_mask(self, seq_x):
        feature_dim = seq_x.size(-1)
        pad_mask = (seq_x.sum(dim=-1) == 0).unsqueeze(1)
        return pad_mask

    def forward(self,wt_neigh,mut_neigh):
        mask_wt = self.get_attn_pad_mask(wt_neigh).unsqueeze(1)
        mask_mut = self.get_attn_pad_mask(mut_neigh).unsqueeze(1)
        wt_neigh = self.bn0(wt_neigh.permute(0, 2, 1))
        mut_neigh = self.bn0(mut_neigh.permute(0, 2, 1))
        wt_neigh = wt_neigh.permute(0, 2, 1)
        mut_neigh = mut_neigh.permute(0, 2, 1)
        wt_neigh = self.PE_neigh(wt_neigh)
        mut_neigh = self.PE_neigh(mut_neigh)
        # Transformer
        wt_neigh = self.transb1(wt_neigh.to(torch.float32),mask_wt)
        mut_neigh = self.transb1(mut_neigh.to(torch.float32),mask_mut)
        wt_neigh = self.transb2(wt_neigh,mask_wt)
        mut_neigh = self.transb2(mut_neigh,mask_mut)
        wt_neigh = self.cnnb1(wt_neigh.permute(0,2,1))
        mut_neigh = self.cnnb1(mut_neigh.permute(0,2,1))
        neigh_sub = torch.sub(wt_neigh,mut_neigh)
        neigh_div = torch.where(mut_neigh != 0, wt_neigh / mut_neigh, wt_neigh / (mut_neigh+1e-08))
        # all0 = torch.cat((neigh_sub,neigh_div),dim=2)
        all0 = neigh_sub
        result = torch.squeeze(all0)
        results1 = self.mlps(result.unsqueeze(0))
        return results1

mutsta_mpnn = PremPS2()
mutsta_mpnn_weights = torch.load('~/MutStab-ProteinMPNN/Model/neigh2_mpnn.pt',map_location=torch.device('cpu'))
mutsta_mpnn.load_state_dict(mutsta_mpnn_weights)

mutsta_mpnn.eval()

torch_ref_embedding_need = torch.from_numpy(ref_embedding_need).unsqueeze(0)
torch_mut_embedding_need = torch.from_numpy(mut_embedding_need).unsqueeze(0)

with torch.no_grad():
    output = mutsta_mpnn(torch_ref_embedding_need,torch_mut_embedding_need)
if output > 0.5:
    output_label = 1
    out_type = 'Stabilizing'
else :
    output_label = 0
    out_type = 'Destablizing'

result_list = []
outlist = [pdb_file.split('.')[0], chain, mut, output.tolist()[0][0], output_label, out_type]
out_col = ['PDB', 'Chain', 'mutation', 'output_logits', 'label', 'type']
result_list.append(outlist)
outdat = pd.DataFrame(result_list,columns=out_col)
outdat.to_csv(output_path + '/' + pdb_file.split('.')[0] + '_' + chain + '_' + mut + '.txt',index=False,sep='\t')

print('Finish predict!')
