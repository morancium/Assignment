# here all the important variables are stored
# this file is imported by all other files
import torch

PATH_OF_DATASET = 'Test_dataset'
SIZE_OF_DATASET = 800
TRAIN_TEST_SPLIT = 0.8

LEARNING_RATE = 0.001
NUM_EPOCHS = 16
BATCH_SIZE = 128

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS=2

CHECKPOINT_MODEL='model.pth'