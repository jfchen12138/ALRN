import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

import matplotlib.pyplot as plt

from libs_path import *
from libs import *

import operator
from functools import reduce
from functools import partial

from timeit import default_timer
from utilities3 import *
import scipy

class UNet2d(nn.Module):

    def __init__(self, in_channels=3, out_channels=1, init_features=32):
        super(UNet2d, self).__init__()

        features = init_features
        self.encoder1 = UNet2d._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = UNet2d._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = UNet2d._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = UNet2d._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = UNet2d._block(features * 8, features * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose2d(features * 16, features * 8, kernel_size=2, stride=2)
        self.decoder4 = UNet2d._block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3 = UNet2d._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = UNet2d._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = UNet2d._block(features * 2, features, name="dec1")

        self.conv = nn.Conv2d(in_channels=features, out_channels=out_channels, kernel_size=1)

    def forward(self, x):
       
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))
        

        bottleneck = self.bottleneck(self.pool4(enc4))
       
        dec4 = self.upconv4(bottleneck)
        
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        return self.conv(dec1)

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "tanh1", nn.Tanh()),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "tanh2", nn.Tanh()),
                ]
            )
        )
    
        
def get_grid3d(S, T, time_scale=10):
    gridx = torch.tensor(np.linspace(0, 1, S + 1)[:-1], dtype=torch.float)
    gridx = gridx.reshape(1, S, 1, 1, 1).repeat([1, 1, S, T, 1])
    gridy = torch.tensor(np.linspace(0, 1, S + 1)[:-1], dtype=torch.float)
    gridy = gridy.reshape(1, 1, S, 1, 1).repeat([1, S, 1, T, 1])
    gridt = torch.tensor(np.linspace(0, 1 * time_scale, T), dtype=torch.float)
    gridt = gridt.reshape(1, 1, 1, T, 1).repeat([1, S, S, 1, 1])
    return gridx, gridy, gridt
        
def UNet_main():

    
    ################################################################
    # configs
    ################################################################
    PATH = './data/ns_V1e-3_N5000_T50.mat'
    
    ntrain = 1024  
    ntest = 100     
    
    batch_size = 2
    learning_rate = 0.001
    
    epochs = 500
    step_size = 100
    gamma = 0.5
    

    
    ################################################################
    # load data and data normalization
    ################################################################
    reader = MatReader(PATH)
    data = reader.read_field("u")  ## [5000, 64, 64, 50]
    a = data[..., :10]  # (N, n, n, T_0:T_1)
    u = data[...,10:20]
    
    
    x_train = a[:ntrain]   
    y_train = u[:ntrain]
    x_train = x_train.reshape(ntrain,64,64,10)
    x_train = x_train.permute(0,3, 1, 2 ) 
    y_train = y_train.reshape(ntrain,64,64,10)
    y_train = y_train.permute(0,3, 1, 2 ) 
    
    x_test = a[-ntest:]
    y_test = u[-ntest:]
    x_test = x_test.reshape(ntest,64,64,10)
    x_test = x_test.permute(0,3,1,2)
    y_test = y_test.reshape(ntest,64,64,10)
    y_test = y_test.permute(0,3,1,2)
    
    
    x_normalizer = UnitGaussianNormalizer(x_train)
    x_train = x_normalizer.encode(x_train)
    x_test = x_normalizer.encode(x_test)
    
    y_normalizer = UnitGaussianNormalizer(y_train)
    y_train = y_normalizer.encode(y_train)
    
    
    
    
    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False)
    
    sample = next(iter(train_loader))

    
    print("train x :", sample[0].shape)
    print("train y :", sample[1].shape)
    # ################################################################
    # # training and evaluation
    # ################################################################
    model = UNet2d(in_channels=10, out_channels=1, init_features=32).cuda()
    print(get_num_params(model))
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
   
    
    start_time = default_timer()
    myloss = LpLoss(size_average=False)
    y_normalizer.cuda()
    for ep in range(epochs):
        model.train()
        t1 = default_timer()
        train_l2 = 0
        train_mse = 0
        for x, y in train_loader:
            x, y = x.cuda(), y.cuda()
            
    
            optimizer.zero_grad()
            mse = 0
            preds = []
            for t in range(10):
                out = model(x)
                m = F.mse_loss(out.view(batch_size, -1), y[:,t:t+1,...].view(batch_size, -1), reduction='mean')
                mse += m
                x = torch.cat((x[:,1:,...], out), dim=1)
                preds.append(out)
            mse.backward()
            
            preds = torch.cat(preds, dim=1)
            # print(preds.shape)
            preds = y_normalizer.decode(preds)
            y = y_normalizer.decode(y)
            loss = myloss(preds.view(batch_size,-1), y.view(batch_size,-1))
    
            optimizer.step()
            train_mse += mse
            train_l2 += loss
    
        scheduler.step()
    
        model.eval()
        test_l2 = 0.0
        with torch.no_grad():
            for x, y in test_loader:
                out_all = []
                x, y = x.cuda(), y.cuda()
                for t in range(10):
                    out = model(x)
                    out_all.append(out)
                    x = torch.cat((x[:,1:,...], out), dim=1)
                    
                out_all = torch.cat(out_all, dim=1)    
                out_all = y_normalizer.decode(out_all)
                test_l2 += myloss(out_all.view(batch_size,-1), y.view(batch_size,-1)).item()
    
        train_mse /= len(train_loader)
        train_l2/= ntrain
        test_l2 /= ntest
    
        t2 = default_timer()
        # print(ep, t2-t1, train_l2, test_l2)
        print("Epoch: %d, time: %.3f, Train Loss: %.3e, Train l2: %.4f, Test l2: %.4f" 
                  % ( ep, t2-t1, train_mse, train_l2, test_l2) )

    elapsed = default_timer() - start_time
    print("\n=============================")
    print("Training done...")
    print('Training time: %.3f'%(elapsed))
    print("=============================\n")

if __name__ == "__main__":
    
    
    UNet_main()
