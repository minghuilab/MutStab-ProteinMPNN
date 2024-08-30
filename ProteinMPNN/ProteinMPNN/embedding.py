#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   embedding.py
@Time    :   2024/01/02 14:20:22
@Author  :   wenyuhao 
@Version :   1.0
@Contact :   925201392@qq.com
@Desc    :   None
'''

# here put the import lib

import json, time, os, sys, glob
import shutil
import warnings
import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split, Subset
import copy
import torch.nn as nn
import torch.nn.functional as F
import random
import os.path
import subprocess

from protein_mpnn_utils import loss_nll, loss_smoothed, gather_edges, gather_nodes, gather_nodes_t, cat_neighbors_nodes, _scores, _S_to_seq, tied_featurize, parse_PDB, parse_fasta
from protein_mpnn_utils import StructureDataset, StructureDatasetPDB, ProteinMPNN
from pathlib import Path
import argparse
import os.path

hidden_dim = 128
num_layers = 3 

def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)   