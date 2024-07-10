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

from models import *
from utils import *

# params
load_weights = True
device = "cuda" if torch.cuda.is_available() else "cpu"

model = CLIP_like(device)
if load_weights:
    weights_path = "/weights.pth.tar"
    load_checkpoint(weights_path,model)

train_df = create_dataframe("/coco-2017-dataset/coco2017","captions_train2017.json","train2017")

# Create persistence Chroma Collection
chroma_client = chromadb.PersistentClient(path="./")
collection_name = "image_embedding"
collection = chroma_client.get_or_create_collection(collection_name, metadata={"hnsw:space": "cosine"})

# Only needed 1 time
create_image_embeddings(model, train_df, device,collection)

# Perform search


