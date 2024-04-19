# Description: Pre-train a ResNet-18 model on the given dataset
# Here we are Using a Pre-trained ResNet-18 model to train on the dataset we created in the previous task. And Test It on the Test Dataset.

#importing the required libraries
import torch
import torch.nn as nn
from torchvision import models
import torch.optim as optim
import torch.nn as nn
import config
import glob
import random
from utils import Data
from torch.utils.data import Dataset,DataLoader
from tqdm import tqdm
from utils import Utils
import matplotlib.pyplot as plt


utils=Utils()

# Load all image paths from the 'dataset' directory with extension '.png'
image_paths=glob.glob('dataset/*.png')

# Shuffle the image paths to avoid order-based biases in training and validation sets
random.shuffle(image_paths)
train_paths=image_paths[:int(len(image_paths)*config.TRAIN_TEST_SPLIT)]

# Create a DataLoader for the training data with defined batch size, shuffling, and number of worker threads
train_data=Data(image_paths=train_paths, label_paths='dataset/label.json')
train_loader = DataLoader(
    train_data, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS
)

# Load image paths from the test dataset directory
image_paths_test=glob.glob('Test_dataset/*.png')

# Create a DataLoader for the test data with defined batch size, shuffling, and number of worker threads
test_data=Data(image_paths=image_paths_test, label_paths='Test_dataset/label.json')
test_loader = DataLoader(
    test_data, batch_size=16, shuffle=True, num_workers=config.NUM_WORKERS
)


# Load the pre-trained ResNet model
model = models.resnet18(pretrained=True)

# Modify the input layer
model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

# Modify the output layer
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)

x = torch.randn(1,1,50,50)
torch.onnx.export(model, x, 'PreTrainedResNet18.onnx', input_names=["features"], output_names=["Coordinates"])

# Set the model to the specified device (CPU or GPU)
model = model.to(config.DEVICE)

# Define the optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
MSE=nn.MSELoss()

def main():
    """
    The main function that trains the model and saves checkpoints.

    This function iterates through the specified number of epochs, performs the following steps in each epoch:
        1. Loops through the training data loader using tqdm for a progress bar.
        2. Initializes a variable to store the total loss for the current epoch.
        3. Iterates through each batch in the data loader:
            - Extracts image and label data from the batch.
            - Moves data to the specified device (CPU or GPU).
            - Clears gradients from the previous iteration.
            - Passes the image through the model to get the output prediction.
            - Calculates the loss between the model's output and the ground truth label using the MSE loss function.
            - Performs backpropagation to calculate gradients.
            - Updates model weights using the optimizer.
            - Accumulates the current batch's loss.
        4. Calculates the average loss for the entire training epoch.
        5. Prints the epoch number and average training loss.
        6. Saves a checkpoint every other epoch (with filename defined in config).
    7. After training, plots the training loss chart (optional).
    """
    Loss_chart=[]
    for epoch in range(config.NUM_EPOCHS):
        loop = tqdm(train_loader,leave=True)
        final_loss = 0
        for data in loop:
            image, label = data["image"], data["label"]
            image, label = image.to(config.DEVICE), label.to(config.DEVICE).squeeze()
            optimizer.zero_grad()
            output = model(image)
            # print("this is output:",output)
            loss = MSE(output, label.float())
            loss.backward()
            optimizer.step()
            final_loss += loss.item()
        Loss_chart.append(final_loss / len(train_loader))
        print(f'Epoch {epoch+1}, Loss: {final_loss / len(train_loader):.4f}')
        if epoch % 2 == 0:
                utils.save_checkpoint(model, optimizer, filename="Pretrained_resnet18.pth")
    plt.plot(Loss_chart)
    plt.show()

def eval():
    """
    Evaluates the trained model on the testing dataset.

    This function performs the following steps:
        1. Disables gradient calculation with `torch.no_grad()` for efficiency during testing.
        2. Sets the model to evaluation mode (`model.eval()`) to potentially disable dropout layers etc.
        3. Iterates through the test data loader using tqdm for a progress bar.
        4. Initializes a variable to store the total loss for the testing set.
        5. Iterates through each batch in the data loader:
            - Extracts image and label data from the batch.
            - Moves data to the specified device (CPU or GPU).
            - Passes the image through the model to get the output prediction (without calculating gradients).
            - Calculates the loss between the model's output and the ground truth label using the MSE loss function.
            - Prints label and output for comparison (might be for debugging purposes).
            - Accumulates the current batch's loss.
            - Visualizes some predictions (might be helpful for qualitative evaluation).
        6. Calculates the average loss for the entire testing set.
        7. Prints the average testing loss.
    """
    final_loss=0
    with torch.no_grad():
        model.eval()
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
    pass
if __name__ == "__main__":
    #main()
    #eval()
    pass