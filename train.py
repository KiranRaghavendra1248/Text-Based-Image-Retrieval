# Imports
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from spacy.tokenizer import Tokenizer
from sklearn.model_selection import train_test_split
import spacy
import pandas as pd
import numpy as np
import os
import pickle
import re
import random
from tqdm import tqdm
from PIL import Image
import math
import matplotlib.pyplot as plt
import json
from collections import defaultdict
from transformers import AutoModel, AutoTokenizer
import chromadb

from dataset import *
from models import *
from utils import *

# Params
embedding_dim = 512
num_epochs = 5
num_workers = 4
learning_rate = 0.0001
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 8
load_weights = True

model = CLIP_like(device)
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(trainable_params)

train_df = create_dataframe("/coco-2017-dataset/coco2017","captions_train2017.json","train2017")
val_df = create_dataframe("/coco-2017-dataset/coco2017","captions_val2017.json","val2017")

# Specify optimizer and LR scheduler
criterion = CustomLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-9)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0)

# Create transform
transform =transforms.Compose([
    transforms.Resize((512,512)),
    transforms.ToTensor()
])

# Create train & val dataloaders
train_dataset = CocoDataset(train_df, transform)
val_dataset = CocoDataset(val_df, transform)

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

# You HAVE to load weights for optim after moving model to cuda, cause Adam optim objects are loaded differntly for cuda & cpu. & cuase error on training
if load_weights:
    weights_path = "/weights.pth.tar"
    load_checkpoint(weights_path,model,optimizer)

# Start training
train_loop(model,train_loader,criterion,optimizer,scheduler,device, num_epochs)

# See results on val set
test_loop(model,val_loader,optimizer,scheduler,device)
