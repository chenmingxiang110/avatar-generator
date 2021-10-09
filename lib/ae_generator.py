import os
import cv2
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
import torchvision.models as models

from tqdm import trange, tqdm

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

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

class Decoder(nn.Module):
    
    def __init__(self, hidden_size):
        super().__init__()
        self.net0 = nn.Sequential(
            nn.Linear(hidden_size, 1600),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(1600),
        )
        self.net1 = nn.Sequential(
            nn.ConvTranspose2d(16, 32, 2, stride=2, padding=0),
            ResidualCNN(32, 32, 1),
            ResidualCNN(32, 32, 1),
            nn.ConvTranspose2d(32, 64, 2, stride=2, padding=0),
            ResidualCNN(64, 64, 1),
            ResidualCNN(64, 64, 1),
            nn.ConvTranspose2d(64, 128, 2, stride=2, padding=0),
            ResidualCNN(128, 128, 1),
            ResidualCNN(128, 128, 1),
            nn.ConvTranspose2d(128, 256, 2, stride=2, padding=0),
            ResidualCNN(256, 256, 1),
            ResidualCNN(256, 256, 1),
            nn.Conv2d(256, 3, 1, stride=1, padding=0),
            nn.Sigmoid(),
        )
    
    def forward(self, x):
        h0 = torch.reshape(self.net0(x), [-1, 16, 10, 10])
        h1 = self.net1(h0)
        return h1

class Generator:
    
    def __init__(self, path, hidden_size, device=torch.device("cpu")):
        self.hidden_size = hidden_size
        self.device = device
        self.decoder = Decoder(self.hidden_size).to(self.device)
        if device==torch.device("cpu"):
            self.decoder.load_state_dict(torch.load(path, map_location='cpu'))
        else:
            self.decoder.load_state_dict(torch.load(path))
        self.decoder = self.decoder.eval()
    
    def generate_rgb(self):
        hidden = np.array([np.random.normal(0, 0.025, self.hidden_size)]).astype(np.float32)
        # hidden = (np.random.random([1,4])*0.08-0.04).astype(np.float32)
        hs_random = self.decoder(torch.from_numpy(hidden).to(self.device)).detach().cpu().numpy()
        _output = hs_random[0].transpose([1,2,0])[...,::-1]
        return _output
    
    def generate_png(self, output_path, size=160):
        hidden = np.array([np.random.normal(0, 0.025, self.hidden_size)]).astype(np.float32)
        # hidden = (np.random.random([1,4])*0.08-0.04).astype(np.float32)
        hs_random = self.decoder(torch.from_numpy(hidden).to(self.device)).detach().cpu().numpy()
        _output = np.clip(hs_random[0].transpose([1,2,0]) * 255, 0, 255).astype(np.uint8)
        if size!=160:
            _output = cv2.resize(_output, (size, size))
        cv2.imwrite(output_path, _output)
        return _output[...,::-1]