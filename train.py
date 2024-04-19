# here all the preprocessing and training loop of the model lies
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

# Load all image paths from the 'dataset' directory with extension '.png'
image_paths=glob.glob('dataset/*.png')

# Shuffle the image paths to avoid order-based biases in training and validation sets
random.shuffle(image_paths)

# Calculate the number of images for training and validation sets based on the split ratio and extracting it
train_paths=image_paths[:int(len(image_paths)*config.TRAIN_TEST_SPLIT)]

val_paths=image_paths[int(len(image_paths)*config.TRAIN_TEST_SPLIT):]

print(f"Number of training images: {len(train_paths)}")
print(f"Number of validation images: {len(val_paths)}")

def train_fn(model, train_loader, optimizer, criterion, device):
    """
    Trains the model on the provided training data loader.

    Args:
        model (torch.nn.Module): The model to be trained.
        train_loader (torch.utils.data.DataLoader): The data loader for training data.
        optimizer (torch.optim.Optimizer): The optimizer used for training.
        criterion (torch.nn.Module): The loss function used for calculating loss.
        device (str): The device to use for training (CPU or GPU).

    Returns:
        float: The average training loss over the entire training set.
    """
    # Set the model to train mode
    model.train()
    final_loss = 0.0
    
    # Use tqdm to iterate through the training data loader with a progress bar
    loop = tqdm(train_loader,leave=True)
    for data in loop:
        
        # Extract images and labels from the current batch
        image, label = data["image"], data["label"]
        
        # Move data to the specified device (CPU or GPU)
        image, label = image.to(device), label.to(device).squeeze()
        
        # Clear gradients from the previous iteration
        optimizer.zero_grad()
        
        # Pass the image through the model to get the output prediction
        output = model(image)
        # print("this is output:",output)
        
        # Calculate the loss between the model's output and the ground truth label
        loss = criterion(output, label.float())
        
        # Backpropagate the loss to calculate gradients
        loss.backward()
        
        # Update model weights based on the calculated gradients using the optimizer
        optimizer.step()
        
        # Accumulate the current batch's loss
        final_loss += loss.item()
        # Calculate the average loss over the entire training set and return it
    return final_loss / len(train_loader)


def main():
    """
    The main function that trains the model and saves checkpoints.
    """
    Loss_chart=[]
    utils=Utils()
    
    # Define the model and move it to the specified device (CPU or GPU)
    model = Model().to(config.DEVICE)
    
    # Define the Adam optimizer and Mean squared error loss function
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    MSE=nn.MSELoss()
    train_data=Data(image_paths=train_paths, label_paths='dataset/label.json')
    train_loader = DataLoader(
        train_data, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS
    )
    
    # Training loop
    for epoch in range(config.NUM_EPOCHS):
        
        # Train the model on the training data loader
        loss = train_fn(model, train_loader, optimizer, MSE, config.DEVICE)
        Loss_chart.append(loss)
        print(f'Epoch: {epoch+1}, Loss: {loss}')
        # Save a checkpoint every other epoch
        if epoch % 2 == 0:
            utils.save_checkpoint(model, optimizer, filename=config.CHECKPOINT_MODEL)

    # Finally plotting the loss graph plot
    plt.plot(Loss_chart)
    plt.show()

if __name__ == "__main__":
    main()
    pass