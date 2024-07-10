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


# Define Embedding Mapping head for Image Encoder and Text Encoder
# this is the guy who will be trained to map images & texts to common embedding space

class EmbeddingMapper(nn.Module):
    def __init__(self, dim_in, dim_out, dropout=0.3):
        super(EmbeddingMapper, self).__init__()
        self.linear1 = nn.Linear(dim_in, dim_out, bias=False)
        self.linear2 = nn.Linear(dim_out, dim_out, bias=False)
        self.layer_norm = nn.LayerNorm(dim_out)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs):
        # input shape : [B X dim_in]
        inputs1 = self.linear1(inputs)
        # input1 shape : [B X dim_out]
        inputs2 = self.dropout(self.linear2(self.dropout(F.gelu(inputs1))))
        # input2 shape : [B X dim_out]
        return self.layer_norm(inputs1 + inputs2)

# Define image embedding extractor - can be any SOTA CNN - Resnet50, ViT etc

class ImageEmbeddingExtractor(nn.Module):
    def __init__(self, out_dim=512):
        super(ImageEmbeddingExtractor, self).__init__()
        self.base = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)  # last layer is a linear layer with out_dim = 1000
        self.embeddingmapper = EmbeddingMapper(dim_in=1000, dim_out=out_dim)
        # freeze weights of base model
        for layer in self.base.parameters():
            layer.requires_grad = False

    def forward(self, images):
        # images shape : [B X C X H X W]
        inputs1 = self.base(images)
        # inputs1 shape : [B X 1000]
        embeddings = self.embeddingmapper(inputs1)
        # embeddings shape : [B X out_dim]
        embeddings_len = torch.linalg.norm(embeddings, dim=-1).unsqueeze(dim=1)
        return embeddings / embeddings_len  # convert to unit vectors


# Define text embedding extractor - can be any transformer architecture - Autoregressive(GPT-like) or Autoencoding(Bert-like)

class TextEmbeddingExtractor(nn.Module):
    def __init__(self, device, out_dim=512):
        super(TextEmbeddingExtractor,self).__init__()
        self.base_tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L12-v2")
        self.base = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L12-v2")
        self.embeddingmapper = EmbeddingMapper(dim_in=384, dim_out=out_dim)
        self.device = device
        # freeze weights of base model
        for layer in self.base.parameters():
            layer.requires_grad = False

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9)

    def forward(self, captions):
        # captions : raw sentences
        encoded_input = self.base_tokenizer(captions, padding=True,truncation=True,return_tensors='pt').to(self.device)
        # encoded_input - contains tokens, attention mask, input_ids
        model_output = self.base(**encoded_input)
        # model_output - contains last hidden state of model
        pooled_outputs = self.mean_pooling(model_output, encoded_input['attention_mask'])
        embeddings = self.embeddingmapper(pooled_outputs)
        # embeddings shape : [B X out_dim]
        embeddings_len = torch.linalg.norm(embeddings, dim=-1).unsqueeze(dim=1)
        # or can use
        # unit_embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings / embeddings_len  # convert to unit vectors


class CLIP_like(nn.Module):
    def __init__(self, device, embedding_dim=512):
        super(CLIP_like, self).__init__()
        self.embedding_dim = embedding_dim
        self.device = device
        self.image_encoder = ImageEmbeddingExtractor(self.embedding_dim)
        self.text_encoder = TextEmbeddingExtractor(device, self.embedding_dim)

    def forward(self, images=None, text=None):
        image_embeddings, text_embeddings, similarity = None, None, None
        # image tensors, raw text

        if images != None:
            image_embeddings = self.image_encoder(images)
        # image embeddings shape - [B X embedding_dim]

        if text != None:
            text_embeddings = self.text_encoder(text)
        # text embeddings shape - [B X embedding_dim]

        if images != None and text != None:
            similarity = image_embeddings @ text_embeddings.T
        # similary matrix shape : [B X B] - represents similarity scores
        # only elements in on the principal diag i.e [1,1] , [2,2] , [3,3] ... should be nearing one
        # all other elements should be zero
        return similarity, image_embeddings, text_embeddings