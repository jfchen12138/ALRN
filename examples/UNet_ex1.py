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

class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        #Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        #Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h**(self.d/self.p))*torch.norm(x.view(num_examples,-1) - y.view(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]

        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)

        return diff_norms/y_norms

    def __call__(self, x, y):
        return self.rel(x, y)

class UNet1d(nn.Module):
    def __init__(self, in_channels=2, out_channels=1, init_features=32):
        super().__init__()

        features = init_features
        self.encoder1 = UNet1d._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.encoder2 = UNet1d._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.encoder3 = UNet1d._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.encoder4 = UNet1d._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.bottleneck = UNet1d._block(features * 8, features * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose1d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = UNet1d._block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose1d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = UNet1d._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose1d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = UNet1d._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose1d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = UNet1d._block(features * 2, features, name="dec1")

        self.conv = nn.Conv1d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

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
                        nn.Conv1d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm1d(num_features=features)),
                    (name + "tanh1", nn.Tanh()),
                    (
                        name + "conv2",
                        nn.Conv1d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm1d(num_features=features)),
                    (name + "tanh2", nn.Tanh()),
                ]
            )
        )



def main():
    n_train = 1024
    n_test = 100
    
    batch_size = 8
    val_batch_size = 4
    epochs = 500
    step_size = 100
    gamma = 0.5
    


    data_path = os.path.join(DATA_PATH, 'burgers_data_R10.mat')
    data = loadmat(data_path)
    x = data['a']
    y = data['u']
    x_train = torch.from_numpy(x[:n_train, ::2])
    y_train = torch.from_numpy(y[:n_train, ::2])
    
    x_test = torch.from_numpy(x[-n_test:, ::2])
    y_test = torch.from_numpy(y[-n_test:, ::2])
    
    x_normalizer = UnitGaussianNormalizer(x_train)
    x_train = x_normalizer.encode(x_train)
    x_test = x_normalizer.encode(x_test)
    
    y_normalizer = UnitGaussianNormalizer(y_train)
    y_train = y_normalizer.encode(y_train)
    
    x_train = x_train.unsqueeze(dim=1)
   
    y_train = y_train.unsqueeze(dim=1) 
    x_test = x_test.unsqueeze(dim=1)
    y_test = y_test.unsqueeze(dim=1)
    
    x_train = x_train.to(torch.float32)
    x_test = x_test.to(torch.float32)
    y_train = y_train.to(torch.float32)
    y_test = y_test.to(torch.float32)
    
    
    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=val_batch_size, shuffle=False)
    
    sample = next(iter(train_loader))

    
    print("train x :", sample[0].shape)
    print("train y :", sample[1].shape)
    
    
    lr = 1e-3

    
    model = UNet1d(in_channels=1, out_channels=1, init_features = 48).cuda()
    
    print("num:", get_num_params(model))
    myloss =  LpLoss(size_average=False)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    # scheduler = OneCycleLR(optimizer, max_lr=lr, div_factor=1e4,
    #                        pct_start=0.2,
    #                        final_div_factor=1e4,
    #                        steps_per_epoch=len(train_loader), epochs=epochs)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    y_normalizer.cuda()
    for ep in range(epochs):
        train_mse = 0
        train_l2 = 0
        for x, y in train_loader:
            x, y = x.cuda(), y.cuda()
            optimizer.zero_grad()
            
            out = model(x)
            
            mse = F.mse_loss(out.view(batch_size, -1), y.view(batch_size, -1), reduction='mean')
            mse.backward()
            
            out = y_normalizer.decode(out)
            y = y_normalizer.decode(y)
            l2 = myloss(out.view(batch_size, -1), y.view(batch_size, -1))
           
            optimizer.step()
            scheduler.step()
            train_mse += mse.item()
            train_l2 += l2.item()
            
        model.eval()
        test_l2 = 0.0
        with torch.no_grad():
            for x,y in test_loader:
                x, y = x.cuda(), y.cuda()
                out = model(x)
                out = y_normalizer.decode(out)
                
                test_l2 += myloss(out.view(val_batch_size, -1), y.view(val_batch_size, -1)).item()
                
        train_mse /= len(train_loader)
        train_l2 /= n_train
        test_l2 /= n_test
        
        print("Epoch: %d,  Train Loss: %.3e, Train l2: %.3e, Test l2: %.3e" 
                  % ( ep, train_mse, train_l2, test_l2) )
       
    
                
            
    
if __name__=="__main__":
    main()