# here we will define the utility functions that will be used in the main script

#importing the required libraries
import matplotlib.pyplot as plt
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import pandas as pd
import json
import torch
import config

class Utils:
    """
    This class provides utility functions for visualization, checkpoint saving, and other tasks.
    """
    def __init__(self):
        """
        Initializes the Utils object.

        This constructor doesn't require any arguments for initialization.
        """
        pass

    def plot_image(self, image, label):
        """
        Plots a single image with its corresponding label.

        Args:
            image (torch.Tensor): Image tensor (grayscale with shape 1x50x50).
            label (str): Label associated with the image.
        """
        plt.imshow(image, cmap='gray')
        plt.title(label)
        plt.show()

    def plot_images(self, images, labels):
        """
        Plots an array of images with their corresponding labels.

        Args:
            images (list): List of image tensors.
            labels (list): List of labels corresponding to the images.
        """
        fig, axes = plt.subplots(1, len(images), figsize=(20, 20))
        for i, ax in enumerate(axes):
            ax.imshow(images[i].squeeze(), cmap='gray')
            ax.set_title(labels[i])
        plt.show()
    
    def save_checkpoint(self,model, optimizer, filename):
        """
        Saves the model and optimizer state dictionaries to a checkpoint file.

        Args:
            model (torch.nn.Module): The model to be saved.
            optimizer (torch.optim.Optimizer): The optimizer used for training.
            filename (str): Path to the checkpoint file.
        """
        print("=> Saving checkpoint")
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        torch.save(checkpoint, filename)
    
    def Visualize(self,output,label):
        """
        Visualizes the model's output and ground truth labels on a 4x4 grid.
        The label will be displayed in white pixel, and the output will be displayed in gray pixel.

        Args:
            output (torch.Tensor): Output tensor from the model (likely with predicted coordinates).
            label (torch.Tensor): Ground truth label tensor (likely with actual coordinates).
        """
        test_visualize = []
        label = label.cpu().numpy()
        output = output.cpu().detach().numpy()
        for i in range(len(output)):
            image = np.zeros((50, 50, 1), dtype=np.uint8)
            image[int(output[i][1])-1, int(output[i][0])-1] = 100
            image[int(label[i][1])-1, int(label[i][0])-1] = 255
            test_visualize.append(image)
            
        fig, axes = plt.subplots(4, 4, figsize=(10, 10))
        for i in range(4):
            for j in range(4):
                index = i * 4 + j
                if index < len(test_visualize):
                    print(index)
                    axes[i, j].imshow(test_visualize[index].squeeze(), cmap='gray')
                    axes[i, j].set_title(label[index])
        plt.tight_layout()
        plt.show()
        pass
        
class Data(Dataset):
    """
    This class represents a custom dataset for loading images and labels.

    It inherits from the `Dataset` class and provides methods for accessing images
    and their corresponding labels based on their paths.
    """
    def __init__(self, image_paths, label_paths,split='train'):
        """
        Initializes the Data object.

        Args:
            image_paths (list): List of paths to the image files.
            label_paths (str): Path to the JSON file containing labels.
            split (str, optional): Split type ('train' or 'test'). Defaults to 'train'.
        """
        self.image_paths = image_paths
        self.label_path = label_paths
        self.split = split
        pass
    
    def __len__(self):
        """
        Returns the length of the dataset (number of images).
        """
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """
        Retrieves a specific image and its corresponding label at the given index.

        Args:
            idx (int): Index of the image to retrieve.

        Returns:
            dict: A dictionary containing the image as a tensor and the label as a tensor.
        """
        # Load label information from the JSON file
        with open(self.label_path, 'r') as openfile:
            json_object = json.load(openfile)

        # Extract image filename from the path
        img_file = self.image_paths[idx]
        image_name=img_file[-8:]
        
        # Retrieve label for the current image from the JSON data
        label = json_object[image_name]
        
        # Load image
        img_file = Image.open(img_file)
        img_file = np.array(img_file)
        
        # Convert image to a PyTorch tensor and normalize (assuming grayscale)
        img_file = transforms.ToTensor()(img_file)
        
        # Convert label to a NumPy array and then to a PyTorch tensor
        label = [[label[0],label[1]]]
        # print("I am form the dataloader class: ", label)
        label = np.array(label)
        label= transforms.ToTensor()(label)
        
        # Return a dictionary containing the image and label tensors
        return {"image": img_file, "label":label}
    pass