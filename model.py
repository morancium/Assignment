# here we write the code for the model
# the model will a CNN model which will take an image of 50x50 pixels as input and will output the x and y coordinates of the white pixel in the image
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self,in_channels=1,out_channels=1):
        super(Model, self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels=1,out_channels= 16,kernel_size= 3, stride=1,padding='same')    #50x50
        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)                                             #25x25
        self.conv2 = nn.Conv2d(in_channels=16,out_channels= 32,kernel_size= 3, stride=1,padding='same')   #25x25
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)                                             #12x12
        self.conv3 = nn.Conv2d(in_channels=32,out_channels= 64,kernel_size= 3, stride=1,padding='same')   #12x12
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)                                             #6x6
        self.fc1 = nn.Linear(64*6*6, 128)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(64, 2)
    
    def forward(self, x):
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
    x = torch.randn(1,1,50,50)
    model = Model()
    return model(x)

if __name__ == "__main__":
    out = test_model()
    print(out.shape)