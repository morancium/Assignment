# here all the preprocessing and training loop of the model lies
import torch
from torch.utils.data import Dataset,DataLoader
import config
import os
import glob
import pandas as pd
import numpy as np
from model import Model
import torch.optim as optim
import torch.nn as nn
from utils import Data
import random
from utils import Utils
from tqdm import tqdm
import matplotlib.pyplot as plt

image_paths=glob.glob('dataset/*.png')
# print(image_paths)
pass
# print((image_paths[:5]))



random.shuffle(image_paths)
train_paths=image_paths[:int(len(image_paths)*config.TRAIN_TEST_SPLIT)]
# print(len(train_paths))
val_paths=image_paths[int(len(image_paths)*config.TRAIN_TEST_SPLIT):]
# print(image_paths[:5])

def train_fn(model, train_loader, optimizer, criterion, device):
    utils=Utils()
    model.train()
    final_loss = 0
    loop = tqdm(train_loader,leave=True)
    for data in loop:
        image, label = data["image"], data["label"]
        # label=torch.transpose(label,0,1)
        # print("this is label:" ,label)
        # print("this is image:" ,image)
        # utils.plot_images(images=image,labels=label)
        image, label = image.to(device), label.to(device).squeeze()
        optimizer.zero_grad()
        output = model(image)
        # print("this is output:",output)
        loss = criterion(output, label.float())
        loss.backward()
        optimizer.step()
        final_loss += loss.item()
        # break
    return final_loss / len(train_loader)


def main():
    Loss_chart=[]
    utils=Utils()
    model = Model().to(config.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    MSE=nn.MSELoss()
    train_data=Data(image_paths=train_paths, label_paths='dataset/label.json')
    train_loader = DataLoader(
        train_data, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS
    )
    for epoch in range(config.NUM_EPOCHS):
        loss = train_fn(model, train_loader, optimizer, MSE, config.DEVICE)
        Loss_chart.append(loss)
        print(f'Epoch: {epoch+1}, Loss: {loss}')
        if epoch % 2 == 0:
            utils.save_checkpoint(model, optimizer, filename=config.CHECKPOINT_MODEL)
        # break
    pass

    # graph plot
    plt.plot(Loss_chart)
    plt.show()

if __name__ == "__main__":
    main()
    pass