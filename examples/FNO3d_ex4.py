"""
@author: Zongyi Li
This file is the Fourier Neural Operator for 2D problem such as the Darcy Flow discussed in Section 5.2 in the [paper](https://arxiv.org/pdf/2010.08895.pdf).
"""
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
torch.set_default_dtype(torch.float32)

# torch.manual_seed(0)
# np.random.seed(0)

def remove_padding(x, num_pad):
    if max(num_pad) > 0:
        res = x[..., num_pad[0]:-num_pad[1]]
    else:
        res = x
    return res

def add_padding(x, num_pad):
    if max(num_pad) > 0:
        res = F.pad(x, (num_pad[0], num_pad[1]), 'constant', 0)
    else:
        res = x
    return res

def _get_act(act):
    if act == 'tanh':
        func = F.tanh
    elif act == 'gelu':
        func = F.gelu
    elif act == 'relu':
        func = F.relu_
    elif act == 'elu':
        func = F.elu_
    elif act == 'leaky_relu':
        func = F.leaky_relu_
    else:
        raise ValueError(f'{act} is not supported')
    return func


################################################################
# fourier layer
################################################################
class SpectralConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super(SpectralConv3d, self).__init__()

        """
        3D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2
        self.modes3 = modes3

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights3 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights4 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul3d(self, input, weights):
        # (batch, in_channel, x,y,t ), (in_channel, out_channel, x,y,t) -> (batch, out_channel, x,y,t)
        return torch.einsum("bixyz,ioxyz->boxyz", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfftn(x, dim=[-3,-2,-1])

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-3), x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, :self.modes2, :self.modes3], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, :self.modes2, :self.modes3], self.weights2)
        out_ft[:, :, :self.modes1, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, -self.modes2:, :self.modes3], self.weights3)
        out_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3], self.weights4)

        #Return to physical space
        x = torch.fft.irfftn(out_ft, s=(x.size(-3), x.size(-2), x.size(-1)))
        return x

class FNO3d(nn.Module):
    def __init__(self, modes1=8, modes2=8, modes3=8, width=20):
        super(FNO3d, self).__init__()


        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.width = width
        self.padding = 6 # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(4, self.width)
        

        self.conv0 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv1 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv2 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv3 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.w0 = nn.Conv3d(self.width, self.width, 1)
        self.w1 = nn.Conv3d(self.width, self.width, 1)
        self.w2 = nn.Conv3d(self.width, self.width, 1)
        self.w3 = nn.Conv3d(self.width, self.width, 1)
        self.bn0 = torch.nn.BatchNorm3d(self.width)
        self.bn1 = torch.nn.BatchNorm3d(self.width)
        self.bn2 = torch.nn.BatchNorm3d(self.width)
        self.bn3 = torch.nn.BatchNorm3d(self.width)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
    
        x = self.fc0(x)
        x = x.permute(0, 4, 1, 2, 3)
        
        x = F.pad(x, [0, self.padding]) # pad the domain if input is non-periodic

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        x = x[..., :-self.padding]
        x = x.permute(0, 2, 3, 4, 1) # pad the domain if input is non-periodic
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x.unsqueeze(-2)
    
def get_grid3d(S, T, time_scale=10):
    gridx = torch.tensor(np.linspace(0, 1, S + 1)[:-1], dtype=torch.float)
    gridx = gridx.reshape(1, S, 1, 1, 1).repeat([1, 1, S, T, 1])
    gridy = torch.tensor(np.linspace(0, 1, S + 1)[:-1], dtype=torch.float)
    gridy = gridy.reshape(1, 1, S, 1, 1).repeat([1, S, 1, T, 1])
    gridt = torch.tensor(np.linspace(0, 1 * time_scale, T), dtype=torch.float)
    gridt = gridt.reshape(1, 1, 1, T, 1).repeat([1, S, S, 1, 1])
    return gridx, gridy, gridt


def FNO_main():

    
    ################################################################
    # configs
    ################################################################
    PATH = '../data/ns_V1e-3_N5000_T50.mat'
    
    ntrain = 1024  
    ntest = 100     
    
    batch_size = 2
    learning_rate = 0.001
    
    epochs = 500
    step_size = 100
    gamma = 0.5
    
    modes = 8
    width = 16
    
   
    
    ################################################################
    # load data and data normalization
    ################################################################
    reader = MatReader(PATH)
    data = reader.read_field("u")  ## [5000, 64, 64, 50]
    a = data[..., :10]  # (N, n, n, T_0:T_1)
    u = data[...,10:20]
    
    
    x_train = a[:ntrain]   
    y_train = u[:ntrain]
    
    x_test = a[-ntest:]
    y_test = u[-ntest:]
    
    
    x_normalizer = UnitGaussianNormalizer(x_train)
    x_train = x_normalizer.encode(x_train)
    x_test = x_normalizer.encode(x_test)
    
    y_normalizer = UnitGaussianNormalizer(y_train)
    y_train = y_normalizer.encode(y_train)
    
    gridx, gridy, gridt = get_grid3d(64, 10)
    
    
    x_train = torch.cat((x_train.reshape(ntrain,64,64,10, 1),
        gridx.repeat([ntrain, 1, 1, 1, 1]),
        gridy.repeat([ntrain, 1, 1, 1, 1]),
        gridt.repeat([ntrain, 1, 1, 1, 1])
        ), dim=-1)
    
    x_test = torch.cat( (x_test.reshape(ntest,64,64,10, 1), gridx.repeat([ntest, 1, 1, 1, 1]), gridy.repeat([ntest, 1, 1, 1, 1]), gridt.repeat([ntest, 1, 1, 1, 1])), dim=-1)
    
    
    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False)
    
    

    
   
    
    
    # ################################################################
    # # training and evaluation
    # ################################################################
    model = FNO3d(modes1 = modes, modes2=modes, modes3= modes, width=width).cuda()
    print(get_num_params(model))
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    # scheduler = OneCycleLR(optimizer, max_lr= learning_rate, div_factor=1e4, final_div_factor=1e4,
    #                    steps_per_epoch=len(train_loader), epochs=epochs)
    
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
            out = model(x).reshape(batch_size, 64, 64, 10)
            
            
            mse = F.mse_loss(out.view(batch_size, -1), y.view(batch_size, -1), reduction='mean')
            mse.backward()
            
            out = y_normalizer.decode(out)
            y = y_normalizer.decode(y)
            loss = myloss(out.view(batch_size,-1), y.view(batch_size,-1))
    
            optimizer.step()
            train_mse += mse.item()
            train_l2 += loss.item()
    
        scheduler.step()
    
        model.eval()
        test_l2 = 0.0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.cuda(), y.cuda()
                out = model(x).reshape(batch_size, 64, 64, 10)
                out = y_normalizer.decode(out)
                test_l2 += myloss(out.view(batch_size,-1), y.view(batch_size,-1)).item()
    
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
    
    training_data_resolution = 141
    run_index = 0
    FNO_main()
