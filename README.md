# Assignment
The Problem Statement:\
Using Deep Learning techniques, predict the coordinates (x,y) of a pixel which has a value of
255 for 1 pixel in a given 50x50 pixel grayscale image and all other pixels are 0. The pixel with a
value of 255 is randomly assigned. You may generate a dataset as required for solving the
problem. Please explain your rationale behind dataset choices.

## Installation
1. Clone the repository:
```
git clone https://github.com/morancium/Assignment.git
```
2. Navigate to the project directory:
``` 
cd Assignment
```
3. Create a virtual environment (recommended) and install the required dependencies:
```
python3 -m venv venv
source venv/bin/activate  # Activate virtual environment (Linux/macOS)
pip install -r requirements.txt  # Install dependencies
```

## Usage

- **config.py**\
Here all the important variables, Hyperparameters, and Paths are stored so that they can be used in the code easily.

- **Dataset.py**\
The Script contains the code to generate the dataset and save it.

- **utils.py** \
This script has two classes, namely: `Utils` and `Data`\
    > `Utils`\
    > This class provides utility functions for the following tasks: 
    > - Visualization:\
       Visualizes the model's output and ground truth labels on a 4x4 grid. The label will be displayed in white pixel, and the output will be displayed in gray pixel along with the plot title as the Label cordinates
    > - Checkpoint saving:\
        It saves the model and optimizer state dictionaries to a checkpoint file.
    > - Plot Image and Plot Images:\
        Both are used to plot the image or an array of Images only with single label may it be the output or ground truth label
    
    > `Data`  
    > This class represents a custom dataset for loading images and labels.
    > It inherits from the `Dataset` class and provides methods for accessing images
    and their corresponding labels based on their paths.

- **model.py**  
This is Where the baseline model resides.  
The following image is a summary of the model
![The baseline CNN](baseline.onnx.png "The Baseline CNN Model")  
It consists of 3 CNN layers each having ReLu activation function and a Maxpool layer followed by 2 Fully connected layers each followed by a Dropout layer and the final layer predicting the coordinates

- **train.py**  
This is where we train our baseline model  
Used the following Model Hyperparameters:  
    > - Batch Size == 128  
    > - Learning Rate == 0.001
    > - Number of Epochs == 64
    > - The loss Function is Mean Squared Error Function
    > - And used Adam Optimizer as the optimizer  

    Save the weights using the Utils class and finally visualizing the loss graph  
    [-] Note: The following Hyperparameters are the best parameters I found after testing different combinations of them.

- **pre_trained_resnet18.py**  
This is where we import the pre-trained Resnet18 model.  
Here we Fine-tune the model as well as test the final model after the training is done.  
The following image is a summary of the model
![The pre-trained Resnet18 model](PreTrainedResNet18.onnx.png "The pre-trained Resnet18 Model")  
Added a CNN layer to adjust the input of the model and also added a final layer to give only two nodes for predicting coordinates.
Used the following Model Hyperparameters:  
    > - Batch Size == 128  
    > - Learning Rate == 0.001
    > - Number of Epochs == 16
    > - The loss Function is Mean Squared Error Function
    > - And used Adam Optimizer as the optimizer  

    Finally tested and saved the model and visualized the results with a Graph.

- **test.py**  
This script contains the code to test and load the trained model on the test dataset which has been created separately from the training dataset, and also visualizing the outputs.