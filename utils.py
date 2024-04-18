# here we will define the utility functions that will be used in the main script
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
    def __init__(self):
        pass

    def plot_image(self, image, label):
        plt.imshow(image, cmap='gray')
        plt.title(label)
        plt.show()

    def plot_images(self, images, labels):
        fig, axes = plt.subplots(1, len(images), figsize=(20, 20))
        for i, ax in enumerate(axes):
            ax.imshow(images[i].squeeze(), cmap='gray')
            ax.set_title(labels[i])
        plt.show()
    
    def save_checkpoint(self,model, optimizer, filename):
        print("=> Saving checkpoint")
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        torch.save(checkpoint, filename)
    
    def Visualize(self,output,label):
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
                    axes[i, j].set_title(label[index])  # Add title to each subplot
        # Adjust layout (optional)
        plt.tight_layout()
        # Show the plot
        plt.show()
        pass
        
class Data(Dataset):
    def __init__(self, image_paths, label_paths,split='train'):
        self.image_paths = image_paths
        self.label_paths = label_paths
        pass
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        with open(self.label_paths, 'r') as openfile:
            json_object = json.load(openfile)
        img_file = self.image_paths[idx]
        image_name=img_file[-8:]
        label = json_object[image_name]
        
        img_file = Image.open(img_file)
        img_file = np.array(img_file)
        img_file = transforms.ToTensor()(img_file)
        label = [[label[0],label[1]]]
        # print("I am form the dataloader class: ", label)
        label = np.array(label)
        label= transforms.ToTensor()(label)
        return {"image": img_file, "label":label}
    pass