import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import pickle as p
import torch.optim as optim
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
import warnings
# from pandarallel import pandarallel
warnings.filterwarnings('ignore')

## 读取数据，得到label标签
workdir = '/public/home/zff/'

f4038_scan = pd.read_csv('/public/home/zff/premps_deep/f4038scan.txt',sep='\t')
f2648_scan = pd.read_csv('/public/home/zff/premps_deep/f2648scan.txt',sep='\t')
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

f4038_scan_desta_keys = list(f4038_scan[f4038_scan['desta_label'] == True]['key'])
f2648_scan_desta_keys = list(f2648_scan[f2648_scan['desta_label'] == True]['key'])

sta_key = f4038_scan_sta_keys + f2648_scan_sta_keys

desta_key = f4038_scan_desta_keys + f2648_scan_desta_keys

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

# label dict
from joblib import Parallel, delayed
from tqdm import tqdm

mega_label = {}
for index, row in tqdm(mega_scan.iterrows(), total=len(mega_scan), desc='Processing'):
    key = row['key']
    ddG = row['ddG']
    label = 0 if ddG >= 0 else 1
    mega_label[key] = label


dst_label = {}
for index, row in tqdm(dst_scan.iterrows(), total=len(dst_scan), desc='Processing'):
    key = row['key']
    ddG = row['ddG']
    label = 0 if ddG >= 0 else 1
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


class CustomDataset(Dataset):
    def __init__(self, keys):
        self.keys = keys
    def __len__(self):
        return len(self.keys)
    def __getitem__(self, index):
        hand_path = workdir+'embedding/hand/select/data/'
        esm_path = workdir+'embedding/esm2_3b/select/data/'
        mpnn_path = workdir+'embedding/proteinmpnn/select/data/'
        key = self.keys[index]
        if fea_name in ['mpnn']:
            embedding = torch.load(f'{mpnn_path}{key}.pt')
        if fea_name in ['esm']:
            embedding = torch.load(f'{esm_path}{key}.pt')
        if fea_name in ['hand']:
            embedding = torch.load(f'{hand_path}{key}.pt')
        neight_wt = embedding['neigh_wt'].to(torch.float32)
        neight_mut = embedding['neigh_mut'].to(torch.float32)            
        labels = torch.from_numpy(np.array(label_dict[key])).to(torch.float32)
        sample = {
                  'neigh_wt':neight_wt,
                  'neigh_mut':neight_mut,
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
        neight_wt = embedding['neigh_wt'].to(torch.float32)
        neight_mut = embedding['neigh_mut'].to(torch.float32)   
        labels = torch.from_numpy(np.array(label_dict[key])).to(torch.float32)
        sample = {
                  'neigh_wt':neight_wt,
                  'neigh_mut':neight_mut,

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
        neight_wt = embedding['neigh_wt'].to(torch.float32)
        neight_mut = embedding['neigh_mut'].to(torch.float32)  
        labels = torch.from_numpy(np.array(label_dict[key])).to(torch.float32)
        sample = {
                  'neigh_wt':neight_wt,
                  'neigh_mut':neight_mut,

                  'labels': labels, \
                  'key': key, \
                  }
        return sample

import torch
import torch.nn as nn
import numpy as np
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

import math
from torch.optim.lr_scheduler import LambdaLR
def get_cosine_schedule_with_warmup(optimizer, num_training_steps,num_cycles=7. / 16.,num_warmup_steps = 0,last_epoch=-1):
    def _lr_lambda(current_step):
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
class EarlyStopping:
    def __init__(self, patience=20, verbose=False, delta=0, trace_func=print):
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
        if self.verbose:
            self.trace_func(f'loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...')
        self.val_loss_min = val_loss

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

from sklearn.metrics import confusion_matrix
import pickle
fea_name = 'mpnn'
jobpath = '/public/home/zff/premps_deep/'
feadim = 512

for head_num in [8,16]:
    path_save = f'/public/home/zff/premps_deep/model_result/neigh2_{fea_name}_head{head_num}/'
    os.system(f'rm -r {path_save}')
    os.system(f'mkdir {path_save}')
    os.system(f'mkdir {path_save}/result')
    os.system(f'mkdir {path_save}/models')

    class PremPS2(nn.Module):
        def __init__(self, input_dim_neigh = feadim ):
            super().__init__()
            self.bn0 = nn.BatchNorm1d(input_dim_neigh)
            self.PE_neigh = TokenPositionEmbedding(23,pos_embed_dim = input_dim_neigh)
            self.transb1 = transformer_block(input_dim_neigh,input_dim_neigh,head = head_num,dropout_rate=0.1)
            self.transb2 = transformer_block(input_dim_neigh,input_dim_neigh,head= head_num,dropout_rate=0.1)
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
            wt_neigh = self.transb1(wt_neigh.to(torch.float32),mask_wt)
            mut_neigh = self.transb1(mut_neigh.to(torch.float32),mask_mut)
            wt_neigh = self.transb2(wt_neigh,mask_wt)
            mut_neigh = self.transb2(mut_neigh,mask_mut)
            wt_neigh = self.cnnb1(wt_neigh.permute(0,2,1))
            mut_neigh = self.cnnb1(mut_neigh.permute(0,2,1))
            neigh_sub = torch.sub(wt_neigh,mut_neigh)
            neigh_div = torch.where(mut_neigh != 0, wt_neigh / mut_neigh, wt_neigh / (mut_neigh+1e-08))
            all0 = neigh_sub
            result = torch.squeeze(all0)
            results1 = self.mlps(result)
            return results1

 
    seed = 0
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

    num_epochs = 100
    train_losses = []  
    valid_losses = []  

    train_keys=np.load('/public/home/zff/premps_deep/train_keys2.npy')
    train_keys=train_keys.tolist()
    valid_keys=np.load('/public/home/zff/premps_deep/valid_keys2.npy')
    valid_keys=valid_keys.tolist()

    train_dataset = CustomDataset(train_keys)
    valid_dataset = CustomDataset(valid_keys)
    train_dataloader = DataLoader(train_dataset, batch_size=512, shuffle=True,num_workers=40)
    valid_dataloader = DataLoader(valid_dataset, batch_size=512, shuffle=True,num_workers=40)
    loss_fn = nn.BCELoss()

    model = PremPS2().to(device)
    izer = .Adam(model.parameters(), lr=0.001, weight_decay=0.00001)
    scheduler = get_cosine_schedule_with_warmup(izer,num_epochs)

    
    latest_epoch = 0
    latest_model_path = None
    for file in os.listdir(os.path.join(path_save, 'models')):
        if file.endswith(".pt"):
            epoch = int(file.split("_")[2].split(".")[0])
            if epoch > latest_epoch:
                latest_epoch = epoch
                latest_model_path = os.path.join(path_save, 'models', file)

    if latest_model_path is not None:
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
        epoch_train_loss = 0.0 
        for i_batch, sample in enumerate(train_dataloader):
            model.zero_grad()
            labels = sample['labels'].to(device)
            key = sample['key']
            neight_wt = sample['neigh_wt'].to(torch.float32).to(device)
            neight_mut = sample['neigh_mut'].to(torch.float32).to(device)   
            logits = model(neight_wt,neight_mut)
            pred = torch.squeeze(logits, 1)
            loss = loss_fn(pred, labels)
            loss.backward()
            izer.step()
            scheduler.step()
            epoch_train_loss += loss.item() * neight_wt.size(0)


        epoch_train_loss /= len(train_dataset)
        train_losses.append(epoch_train_loss)
        torch.save(model.state_dict(), f"{path_save}/models/model_epoch_{epoch}.pt")
        model.eval()
        with torch.no_grad():
            epoch_valid_loss = 0.0
            valid_preds = []
            valid_labels = []
            for sample in valid_dataloader:
                labels = sample['labels'].to(device)
                valid_labels.extend(labels.cpu().numpy())
                neight_wt = sample['neigh_wt'].to(torch.float32).to(device)
                neight_mut = sample['neigh_mut'].to(torch.float32).to(device)   
                logits = model(neight_wt,neight_mut)
                pred = torch.squeeze(logits, 1)
                loss = loss_fn(pred, labels)
                epoch_valid_loss += loss.item() * neight_wt.size(0)
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

    
    seed = 0
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    
    def load_model(model_path, device):
        model = PremPS2() 
        model.load_state_dict(torch.load(model_path)) 
        model.to(device)
        model.eval()
        return model
    def get_model_files(path):
        return [file for file in os.listdir(path) if file.endswith('.pt')]
    def test_model(model, test_dataloader, device):
        pred_data = []
        label_data = []
        key_data = []
        for i_batch, sample in enumerate(test_dataloader):
            labels = sample['labels'].to(device)
            key = sample['key']
            neight_wt = sample['neigh_wt'].to(torch.float32).to(device)
            neight_mut = sample['neigh_mut'].to(torch.float32).to(device)  
            logits = model(neight_wt,neight_mut)
            pred = torch.squeeze(logits, 1)
            pred_data.extend(pred.cpu().detach().numpy())
            label_data.extend(labels.cpu().detach().numpy())
            key_data.extend(key)
        return key_data, pred_data, label_data

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


    model_files = get_model_files(f'{path_save}/models/')
    epoch = max([int(model_file.split('_')[-1].split('.')[0]) for model_file in model_files])
    print(f'final epoch : {epoch}')

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
    result_df = pd.DataFrame({'Epoch': [epoch]*len(keys),
                              'Key': keys,
                              'Predicted': preds,
                              'Actual': labels})
    result_df.to_csv( f'{path_save}/result/S4038_result_epoch_{epoch}.csv',sep = '\t', index=False)
    tpr, fpr, mcc, auc_roc, f1, precision = calculate_metrics(result_df)
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
    result_df = pd.DataFrame({'Epoch': [epoch]*len(keys),
                              'Key': keys,
                              'Predicted': preds,
                              'Actual': labels})
    result_df.to_csv( f'{path_save}/result/S2648_result_epoch_{epoch}.csv',sep = '\t', index=False)
    tpr, fpr, mcc, auc_roc, f1, precision = calculate_metrics(result_df)
    metrics_list_2648.append({'Epoch': epoch,
                         'TPR': tpr,
                         'FPR': fpr,
                         'MCC': mcc,
                         'AUC-ROC': auc_roc,
                         'F1-Score': f1,
                         'Precision': precision})

    ## mega
    test_dataset = CustomDataset_mega(mega_keys)
    test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=True, num_workers=40)
    keys, preds, labels = test_model(model, test_dataloader, device)
    result_df = pd.DataFrame({
                              'Key': keys,
                              'Predicted': preds,
                              'Actual': labels})
    result_df.to_csv(f'{path_save}/result/mega_result_epoch_{epoch}.csv',sep = '\t', index=False)
    tpr, fpr, mcc, auc_roc, f1, precision = calculate_metrics(result_df)
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
    result_df = pd.DataFrame({
                              'Key': keys,
                              'Predicted': preds,
                              'Actual': labels})
    result_df.to_csv( f'{path_save}/result/dst_result_epoch_{epoch}.csv',sep = '\t', index=False)
    tpr, fpr, mcc, auc_roc, f1, precision = calculate_metrics(result_df)
    metrics_list_dst.append({
                         'TPR': tpr,
                         'FPR': fpr,
                         'MCC': mcc,
                         'AUC-ROC': auc_roc,
                         'F1-Score': f1,
                         'Precision': precision})
    metrics_df_dst = pd.DataFrame(metrics_list_dst)
    metrics_df_dst.to_csv( f'{path_save}/dst_all_epochs_metrics.txt',sep = '\t', index=False)
