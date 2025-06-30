import os
import torch
class Config:
    DATA_ROOT = "vangogh2photo"
    IMG_HEIGHT = 256
    IMG_WIDTH = 256
    BATCH_SIZE = 1
    NUM_WORKERS = 4
    NUM_EPOCHS = 20
    LR = 0.0002
    BETA1 = 0.5
    BETA2 = 0.999
    LAMBDA_CYCLE = 10
    LAMBDA_ID = 5
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"