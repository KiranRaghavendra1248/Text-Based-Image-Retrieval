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

def create_dataframe(BASE_PATH,json_file,image_folder):
    path = os.path.join(BASE_PATH,"annotations/"+json_file)
    with open(path) as f:
        data = json.load(f)
        data = data['annotations']

    img_cap_pairs = []

    for sample in data:
        img_name = '%012d.jpg' % sample['image_id']
        img_cap_pairs.append([img_name, sample['caption']])

    captions = pd.DataFrame(img_cap_pairs, columns=['image', 'caption'])
    captions['image'] = captions['image'].apply(
        lambda x: f'{BASE_PATH}/{image_folder}/{x}'
    )
    captions = captions.reset_index(drop=True)
    return captions

# Save checkpoint
def save_checkpoint(state,filename='weights.pth.tar'):
    print('Saving weights-->')
    torch.save(state,filename)

# Load checkpoint
def load_checkpoint(filename,model,optim=None):
    print('Loading weights-->')
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['state_dict'])
    if optim != None:
        optim.load_state_dict(checkpoint['optimizer'])

# Visualize the images
def visualize_image(dataframe, idx):
    image_path = dataframe['image'][idx]
    image = Image.open(image_path)

    # Define the title
    title = dataframe['caption'][idx]

    # Display the image with the title
    plt.figure(figsize=(8, 5))  # Adjust the figure size as needed
    plt.imshow(image)
    plt.title(title, fontsize=10)  # Adjust the font size as needed
    plt.axis('off')  # Hide axes
    plt.show()

# check accuracy function

def check_accuracy(similarities):
    n = similarities.shape[1]
    y = torch.arange(n).to(similarities.device)
    img2cap_match_idx = similarities.argmax(dim=1)
    img_acc = (img2cap_match_idx == y).float().mean()

    return img_acc


# custom loss using cross entropy loss function

class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, similarities):
        n = similarities.shape[1]

        # Create labels tensor
        labels = torch.arange(n,device=similarities.device)

        # Calculate cross entropy loss
        loss = F.cross_entropy(similarities, labels)

        return loss


def test_loop(model, dataloader, optimizer, scheduler, device):
    model.eval()
    model.to(device)
    losses = []
    samples, correct = 0, 0
    loop = tqdm(enumerate(dataloader),total=len(dataloader), leave=True)
    with torch.no_grad():
        for batch, data in loop:
            images = data['image']
            text = data['caption']

            # put on cuda
            images = images.to(device)

            # forward pass
            similarities = model(images, text)

            # accuracy over entire dataset
            num_samples = similarities.shape[0]
            samples += num_samples
            y = torch.arange(num_samples)
            img2cap_match_idx = similarities.argmax(dim=1)
            correct += (img2cap_match_idx == y).sum().item()

    print("Final Test Accuracy = ", 100 * (correct / samples))


def train_loop(model, dataloader, loss_fun, optimizer, scheduler, device, num_epochs):
    model.train()
    model.to(device)
    min_loss = None
    for epoch in range(num_epochs):
        losses = []
        accuracies = []
        loop = tqdm(enumerate(dataloader), total=len(dataloader),leave=True)
        for batch, data in loop:
            images = data['image']
            text = data['caption']
            # put on cuda
            images = images.to(device)

            # forward pass
            similarities, _, _ = model(images, text)

            # calculate loss & accuracy
            loss = loss_fun(similarities)
            losses.append(loss.detach().item())

            accuracy = check_accuracy(similarities)
            accuracies.append(accuracy.detach().item())

            # zero out prior gradients
            optimizer.zero_grad()

            # backprop
            loss.backward()

            # update weights
            optimizer.step()
            scheduler.step()

            # Update TQDM progress bar
            loop.set_description(f"Epoch [{epoch}/{num_epochs}] ")
            loop.set_postfix(loss=loss.detach().item(), accuracy=accuracy.detach().item())

        moving_loss = sum(losses) / len(losses)
        moving_accuracy = sum(accuracies) / len(
            accuracies)
        checkpoint = {
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        # Save check point
        if min_loss == None:
            min_loss = moving_loss
            save_checkpoint(checkpoint)
        elif moving_loss < min_loss:
            min_loss = moving_loss
            save_checkpoint(checkpoint)
        print('Epoch {0} : Loss = {1} , Training Accuracy={2}'.format(epoch, moving_loss, moving_accuracy))

# Generate & store image
def create_image_embeddings(model, dataframe, device, collection):
    # Put model to evat
    model.eval()
    model.to(device)

    transform =transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])

    index = 0

    for image_path in dataframe['image']:
        # open image & preprocess
        image = Image.open(image_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0)

        # move to cuda
        image_tensor = image_tensor.to(device)

        # forward prop
        _, image_embedding, _ = model(image_tensor)

        # insert to Chroma DB
        collection.add(ids=[str(index)],
                       embeddings=image_embedding.tolist(),
                       metadatas=[{"image_path": image_path}])
        index += 1