# Aim of this file is to generate the dataset for the model which be image of 50x50 pixels in which a random pixle will be white and rest will be black.
# We will save the image in the folder named 'dataset' and the label in the file named 'label.csv'
#  we will make dataset of 4000 images and labels
# The dataset will be split into 80% training and 20% testing

#importing the required libraries
import numpy as np
import config
import random
import cv2
import pandas as pd
import os
import time
import json

class Dataset:
    """
    This class represents a dataset for machine learning tasks.
    It can generate, save, and load datasets consisting of images and corresponding labels.
    """
    
    def __init__(self, path, split, size):
        """
        Initializes the Dataset object.
        Args:
            path (str): Path to the directory where the dataset will be stored or loaded.
            split (float): Ratio of the dataset to be used for training (0.0 to 1.0).
            size (int): Total number of images in the dataset.
        """
        self.dataset = []
        self.labels = {}
        self.dataset_size = size
        self.train_size =int( size * split )
        self.test_size = size - self.train_size
        self.image_size = 50
        self.image_channels = 1
        self.label_size = 2
        self.label_names = {"File_name":['X',"Y"]}
        self.dataset_path = path
        self.label_path = path+'/label.json'

    def generate_dataset(self):
        """Generates a dataset of random images with corresponding labels.

        This function creates the dataset by generating images with a single white pixel
        at a random location and storing the coordinates as labels.
        """
        for i in range(self.dataset_size):
            image = np.zeros((self.image_size, self.image_size, self.image_channels), dtype=np.uint8)
            label = np.zeros(self.label_size, dtype=np.uint8)
            x = random.randint(0, self.image_size - 1)
            y = random.randint(0, self.image_size - 1)
            image[y, x] = 255
            label = [x,y]
            self.dataset.append(image)
            self.labels['{:04d}.png'.format(i+1)]=label
        # print(self.labels)

    def save_dataset(self):
        """Saves the generated dataset to images and labels in JSON format.

        This function creates the dataset directory if it doesn't exist and then saves
        each image as a PNG file and the labels as a JSON file.
        """
        if not os.path.exists(self.dataset_path):
            os.makedirs(self.dataset_path)
        for i in range(self.dataset_size):
            image = self.dataset[i]
            image_path = os.path.join(self.dataset_path, '{:04d}.png'.format(i+1))
            cv2.imwrite(image_path, image)
        with open(self.label_path, 'w') as f:
            json.dump(self.labels, f)

    def get_train_data(self):
        """Returns the training data (images and labels).

        This function returns the entire dataset (images and labels) which can be used for training.

        Returns:
            self.dataset: Images
            self.labels: Labels
        """
        return self.dataset, self.labels

if __name__ == '__main__':
    start = time.time()
    dataset = Dataset(config.PATH_OF_DATASET,config.TRAIN_TEST_SPLIT,config.SIZE_OF_DATASET)
    dataset.generate_dataset()
    dataset.save_dataset()
    # dataset.load_dataset()
    dataset_load, labels_load = dataset.get_train_data()
    end = time.time()
    #printing Usfull information about the dataset
    print('Dataset size:', len(dataset_load))
    print('Labels size:', len(dataset_load))
    print('Train dataset size:', len(dataset_load))
    print('Train labels size:', len(labels_load))
    print('Label names:', dataset.label_names)
    print('Label shape:', dataset.label_size)
    print('Image shape:', dataset.image_size, dataset.image_size, dataset.image_channels)
    print('Label:', labels_load['0001.png'])
    print('Image:', dataset_load[0].shape)
    print('Image:', dataset_load[0])
    print('Image:', dataset_load[0].dtype)
    print('Image:', dataset_load[0].max())
    print('Image:', dataset_load[0].min())
    print('Total time taken for Dataset generation: ',end - start)