# here all the important variables are stored
# this file is imported by all other files
import torch


# Define paths for datasets
PATH_OF_DATASET = 'dataset'
PATH_OF_TEST_DATASET = 'Test_dataset'

# Define dataset size and split for training/testing
SIZE_OF_DATASET = 2000
TRAIN_TEST_SPLIT = 0.8

# Define training hyperparameters
LEARNING_RATE = 0.001
NUM_EPOCHS = 16
BATCH_SIZE = 128

# Define device for training/evaluation
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS=2

# Define paths for model checkpoints and pre-trained model
CHECKPOINT_MODEL='model.pth'
PRETRAINED_MODEL='Pretrained_resnet18.pth'