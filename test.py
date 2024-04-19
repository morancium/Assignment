#here we are testing the model after training it on the dataset
#importing the required libraries
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
    """
    Evaluates the trained model on the testing dataset.

    This function loads the saved model checkpoint, calculates the loss on the testing data,
    and visualizes some predictions using the `Utils.Visualize` function.
    """
    final_loss=0
    utils=Utils()
    
    # Load image paths from the test dataset directory
    image_paths=glob.glob('Test_dataset/*.png')

    test_data=Data(image_paths=image_paths, label_paths='Test_dataset/label.json')
    
    # Create a DataLoader for the test data with defined batch size, shuffling, and number of worker threads
    test_loader = DataLoader(
        test_data, batch_size=16, shuffle=True, num_workers=config.NUM_WORKERS
    )
    
    model = Model().to(config.DEVICE)
    
    # Set the model to evaluation mode (to disable dropout layers etc.)
    model.eval()
    
    # Define the optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    MSE=nn.MSELoss()
    
    # Load the model checkpoint
    print("=> Loading checkpoint")
    checkpoint = torch.load('model.pth', map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    for param_group in optimizer.param_groups:
        param_group["lr"] = config.LEARNING_RATE
    print("Wt Loaded")
    
    # Testing loop
    loop = tqdm(test_loader,leave=True)
    for data in loop:
        image, label = data["image"], data["label"]
        image, label = image.to(config.DEVICE), label.to(config.DEVICE).squeeze()
        output = model(image)
        loss = MSE(output, label.float())
        print("label:",label)
        print("Output:",output)
        final_loss += loss.item()
    
    # Visualize some predictions
    utils.Visualize(output,label)
    
    # Calculate the average loss over the entire testing set and print it
    print(final_loss / len(test_loader))
    



if __name__ == '__main__':
    main()