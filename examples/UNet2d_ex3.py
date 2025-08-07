
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

    def __init__(self, in_channels=1, out_channels=1, init_features=32):
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

        self.upconv4 = nn.ConvTranspose2d(features * 16, features * 8, kernel_size=2, stride=2, output_padding=1)
        self.decoder4 = UNet2d._block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2, output_padding=1)
        self.decoder3 = UNet2d._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = UNet2d._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2, output_padding=1)
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
    

def main(train_data_res):
    """
    Parameters
    ----------
    train_data_res : resolution of the training data
    save_index : index of the saving folder
    """

    
    ################################################################
    # configs
    ################################################################
    TRAIN_PATH = './data/piececonst_r421_N1024_smooth1.mat'
    TEST_PATH = './data/piececonst_r421_N1024_smooth2.mat'
    
    ntrain = 1000   # first 1000 of smooth1.mat 
    ntest = 100     # first 100 of smooth1.mat 
    
    batch_size = 2
    learning_rate = 0.001
    
    epochs = 100
    step_size = 100
    gamma = 0.5
    
    modes = 12
    width = 32
    
    s = train_data_res
    r = (421-1) // (s-1) 
   
    
    ################################################################
    # load data and data normalization
    ################################################################
    reader = MatReader(TRAIN_PATH)
    x_train = reader.read_field('coeff')[:ntrain,::r,::r][:,:s,:s]
    y_train = reader.read_field('sol')[:ntrain,::r,::r][:,:s,:s]
    x_train, y_train = y_train, x_train
    
    x_train += 0.01* np.random.rand(*x_train.shape)
    
    reader.load_file(TEST_PATH)
    x_test = reader.read_field('coeff')[:ntest,::r,::r][:,:s,:s]
    y_test = reader.read_field('sol')[:ntest,::r,::r][:,:s,:s]
    x_test, y_test = y_test, x_test
    x_test += 0.01* np.random.rand(*x_test.shape)
    
    

    x_normalizer = UnitGaussianNormalizer(x_train)
    x_train = x_normalizer.encode(x_train)
    x_test = x_normalizer.encode(x_test)
    
    y_normalizer = UnitGaussianNormalizer(y_train)
    y_train = y_normalizer.encode(y_train)
    
    grids = []
    grid_all = np.linspace(0, 1, 421).reshape(421, 1).astype(np.float64)
    grids.append(grid_all[::r,:])
    grids.append(grid_all[::r,:])
    grid = np.vstack([xx.ravel() for xx in np.meshgrid(*grids)]).T
    grid = grid.reshape(1,s,s,2)
    grid = torch.tensor(grid, dtype=torch.float)
    
    x_train = x_train.reshape(ntrain,s,s,1)
    # x_train = torch.cat([x_train.reshape(ntrain,s,s,1), grid.repeat(ntrain,1,1,1)], dim=3)
    x_train = x_train.permute(0,3, 1, 2 ) 
    x_test = x_test.reshape(ntest,s,s,1)
    # x_test = torch.cat([x_test.reshape(ntest,s,s,1), grid.repeat(ntest,1,1,1)], dim=3)
    x_test = x_test.permute(0,3, 1, 2)
    
    x_train = x_train.to(torch.float32)
    x_test = x_test.to(torch.float32)
    y_train = y_train.to(torch.float32)
    y_test = y_test.to(torch.float32)
    
    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False)
    
    sample = next(iter(train_loader))

    
    print("train x :", sample[0].shape)
    print("train y :", sample[1].shape)
    
    
    ################################################################
    # training and evaluation
    ################################################################
    model = UNet2d().cuda()
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
            out = model(x)

            
            
            mse = F.mse_loss(out.view(batch_size, -1), y.view(batch_size, -1), reduction='mean')
            mse.backward()
            
            out = y_normalizer.decode(out)
            y = y_normalizer.decode(y)
            loss = myloss(out.view(batch_size,-1), y.view(batch_size,-1))
            # loss.backward()
    
            optimizer.step()
            train_mse += mse.item()
            train_l2 += loss.item()
    
        scheduler.step()
    
        model.eval()
        test_l2 = 0.0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.cuda(), y.cuda()
                out = model(x).reshape(batch_size, s, s)
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
    main(training_data_resolution)
