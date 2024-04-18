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

image_paths=glob.glob('dataset/*.png')
random.shuffle(image_paths)
train_paths=image_paths[:int(len(image_paths)*config.TRAIN_TEST_SPLIT)]

train_data=Data(image_paths=train_paths, label_paths='dataset/label.json')
train_loader = DataLoader(
    train_data, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS
)

image_paths_test=glob.glob('Test_dataset/*.png')

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

model = model.to(config.DEVICE)
optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
MSE=nn.MSELoss()

def main():
    Loss_chart=[]
    for epoch in range(config.NUM_EPOCHS):
        loop = tqdm(train_loader,leave=True)
        final_loss = 0
        for data in loop:
            image, label = data["image"], data["label"]
            # label=torch.transpose(label,0,1)
            # print("this is label:" ,label)
            # print("this is image:" ,image)
            # utils.plot_images(images=image,labels=label)
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
    main()
    eval()
    pass