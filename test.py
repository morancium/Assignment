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


def main():
    final_loss=0
    utils=Utils()
    image_paths=glob.glob('Test_dataset/*.png')

    test_data=Data(image_paths=image_paths, label_paths='Test_dataset/label.json')
    test_loader = DataLoader(
        test_data, batch_size=16, shuffle=True, num_workers=config.NUM_WORKERS
    )
    
    model = Model().to(config.DEVICE)
    model.eval()
    
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    MSE=nn.MSELoss()
    
    print("=> Loading checkpoint")
    
    checkpoint = torch.load('model.pth', map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    for param_group in optimizer.param_groups:
        param_group["lr"] = config.LEARNING_RATE
    
    
    print("Wt Loaded")
    
    loop = tqdm(test_loader,leave=True)
    for data in loop:
        image, label = data["image"], data["label"]
        image, label = image.to(config.DEVICE), label.to(config.DEVICE).squeeze()
        output = model(image)
        loss = MSE(output, label.float())
        print("label:",label)
        print("Output:",output)
        final_loss += loss.item()
        # break
    utils.Visualize(output,label)
    print(final_loss / len(test_loader))
    



if __name__ == '__main__':
    main()
    # print(config.DEVICE)