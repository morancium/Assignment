# here we write the code for the model
# the model will a CNN model which will take an image of 50x50 pixels as input and will output the x and y coordinates of the white pixel in the image

#importing the required libraries
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    This class defines a convolutional neural network (CNN) model.

    The model takes an image as input (grayscale, 1 channel) and outputs a prediction
    with two values, corresponding to the X and Y coordinates of a point of interest.
    """
    def __init__(self,in_channels=1,out_channels=2):
        """
        Initializes the Model object.

        Args:
            in_channels (int, optional): Number of input channels (grayscale: 1). Defaults to 1.
            out_channels (int, optional): Number of output channels (likely 2 for X and Y coordinates). Defaults to 2.
        """
        super(Model, self).__init__()
        
        # Define convolutional layers with ReLU activation and padding for same image size
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels=1,out_channels= 16,kernel_size= 3, stride=1,padding='same')    #50x50
        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)                                             #25x25
        self.conv2 = nn.Conv2d(in_channels=16,out_channels= 32,kernel_size= 3, stride=1,padding='same')   #25x25
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)                                             #12x12
        self.conv3 = nn.Conv2d(in_channels=32,out_channels= 64,kernel_size= 3, stride=1,padding='same')   #12x12
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)                                             #6x6
        
        # Define fully-connected layers with ReLU activation and dropout for regularization
        self.fc1 = nn.Linear(64*6*6, 128)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(0.5)
        
        # Output layer with 2 units (likely for X and Y coordinates)
        self.fc3 = nn.Linear(64, 2)
    
    def forward(self, x):
        """
        Defines the forward pass of the model.

        This function takes an image (tensor) as input and propagates it through the
        convolutional and fully-connected layers, returning the final output.
        """
        x = self.relu(self.conv1(x))
        x = self.Maxpool1(x)
        x = self.relu(self.conv2(x))
        x = self.Maxpool2(x)
        x = self.relu(self.conv3(x))
        x = self.Maxpool3(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x
    
def test_model():
    """
    Tests the model by creating a random input and passing it through the model.

    This function helps verify that the model can be constructed and used.
    """
    x = torch.randn(1,1,50,50)
    model = Model()
    torch.onnx.export(model, x, 'baseline.onnx', input_names=["features"], output_names=["Coordinates"])
    return model(x)

if __name__ == "__main__":
    out = test_model()
    print(out.shape)