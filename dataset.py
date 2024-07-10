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


# Define the custom dataset for MSCOCO

class CocoDataset(Dataset):
    def __init__(self, dataframe, image_transform=None):
        self.dataframe = dataframe
        self.image_transform = image_transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        image_path = self.dataframe['image'][idx]
        caption = self.dataframe['caption'][idx]

        image = Image.open(image_path).convert("RGB")

        if self.image_transform:
            image = self.image_transform(image)

        return {"image": image, "caption": caption}