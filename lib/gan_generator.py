import os
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
import torchvision.models as models

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def count_parameters(model, verbose=True):
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        if verbose: print([name, param])
        total_params+=param
    if verbose: print(f"Total Trainable Params: {total_params}")
    return total_params

class ResidualCNN(nn.Module):
    
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()

        self.cnn1 = nn.Conv2d(in_channels, in_channels, 3, stride=stride, padding=1)
        self.cnn2 = nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1)
        self.is_shortcut = in_channels!=out_channels or stride!=1
        if self.is_shortcut:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 3, stride, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(0.1)

    def forward(self, x):
        residual = self.shortcut(x) if self.is_shortcut else x
        x = self.cnn1(x)
        x = self.cnn2(x)
        x += residual
        x = self.bn(x)
        x = self.act(x)
        return x

class Encoder(nn.Module):

    def __init__(self, hidden_size):
        super().__init__()
        self.net = models.resnet18(pretrained=False)
        self.net.fc = nn.Linear(512, hidden_size)

    def forward(self, x):
        h = self.net(x)
        return h

class Decoder(nn.Module):
    
    def __init__(self, hidden_size):
        super().__init__()
        self.net0 = nn.Sequential(
            nn.Linear(hidden_size, 4096),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(4096),
        )
        self.net1 = nn.Sequential(
            nn.ConvTranspose2d(256, 256, 2, stride=2, padding=0),
            ResidualCNN(256, 256, 1),
            ResidualCNN(256, 256, 1),
            ResidualCNN(256, 256, 1),
            nn.ConvTranspose2d(256, 128, 2, stride=2, padding=0),
            ResidualCNN(128, 128, 1),
            ResidualCNN(128, 128, 1),
            ResidualCNN(128, 128, 1),
            ResidualCNN(128, 128, 1),
            nn.ConvTranspose2d(128, 64, 2, stride=2, padding=0),
            ResidualCNN(64, 64, 1),
            ResidualCNN(64, 64, 1),
            ResidualCNN(64, 64, 1),
            ResidualCNN(64, 64, 1),
            ResidualCNN(64, 64, 1),
            ResidualCNN(64, 64, 1),
            ResidualCNN(64, 64, 1),
            ResidualCNN(64, 64, 1),
            nn.ConvTranspose2d(64, 32, 2, stride=2, padding=0),
            ResidualCNN(32, 32, 1),
            ResidualCNN(32, 32, 1),
            ResidualCNN(32, 32, 1),
            nn.Conv2d(32, 3, 1, stride=1, padding=0),
            nn.Sigmoid(),
        )
    
    def forward(self, x):
        h0 = torch.reshape(self.net0(x), [-1, 256, 4, 4])
        h1 = self.net1(h0)
        return h1

class Generator:
    
    def __init__(self, model_path, seed_path):
        self.hidden_size = 512
        self.device = torch.device("cpu")
        self.decoder = Decoder(self.hidden_size).to(self.device)
        self.decoder.load_state_dict(torch.load(model_path, map_location="cpu"))
        self.decoder = self.decoder.eval()
        self.seeds = np.load(seed_path)
        self.std = 1
    
    def generate(self, index=None, do_aug=True):
        if index is None:
            index = np.random.randint(self.seeds.shape[0])
        
        if do_aug:
            index1 = np.random.randint(self.seeds.shape[0])
            rate = np.random.random() * 0.5 + 0.5
            hidden = self.seeds[index:index+1] * rate + self.seeds[index1:index1+1] * (1-rate)
            hidden += np.random.normal(0, self.std, [1,self.hidden_size])
        else:
            hidden = self.seeds[index:index+1]
        
        hidden = torch.from_numpy(hidden.astype(np.float32)).to(self.device)
        hs = self.decoder(hidden).detach().cpu().numpy()[0].transpose([1,2,0])
        return hs
