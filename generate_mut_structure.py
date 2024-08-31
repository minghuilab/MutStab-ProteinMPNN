#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import os
import pandas as pd
import sys
from tqdm import tqdm

def get_mutfile_txt(pdb1,mut1,chain1,path1):
    muts = []
    x_mut = mut1[0] + chain1 + mut1[1:] + ';'
    muts.append(x_mut)
    muts_dat = pd.DataFrame({'mut':muts})
    muts_dat.to_csv(path1 + '/individual_list_'+pdb1+'_'+chain1+'_1_mut.txt',index=False,header=None,sep='\t')

i = 0
while i < len(sys.argv):
    if sys.argv[i] == "-w":
        # 工作路径
        workdir = sys.argv[i + 1]
        i += 2  # 跳过当前参数及其值
    elif sys.argv[i] == "-p":
        # 要处理的文件地址
        file = sys.argv[i + 1]
        i += 2  # 跳过当前参数及其值 
    elif sys.argv[i] == "-m":
        # 要处理的文件地址
        mut_chain = sys.argv[i + 1]
        i += 2  # 跳过当前参数及其值 
    else:
        # 如果当前参数不是我们想要的参数，则继续检查下一个参数
        i += 1

os.chdir(workdir)
chain = mut_chain.split('_')[1]
mut = mut_chain.split('_')[0]

# 生成突变信息文件
get_mutfile_txt(file.split('.')[0],mut,chain,workdir)

# 运行foldx，产生突变体结构
os.system('./foldx --command=BuildModel --pdb='+file.split('.')[0]+'.pdb --mutant-file=individual_list_'+file.split('.')[0]+'_'+chain+'_1_mut.txt')

mutdat = pd.read_csv(workdir + '/individual_list_'+file.split('.')[0]+'_'+chain+'_1_mut.txt',header=None,sep='\t').rename(columns={0:'mut'})
for i in range(len(mutdat)):
    mut_use = mutdat['mut'][i]
    mut_use1 = mut_use[0] + mut_use[2:-1]
    os.system('cp ' + file.split('.')[0] + '_' + str(i+1) + '.pdb ' + file.split('.')[0] + '_' + chain + '_' + mut_use1 + '.pdb')

