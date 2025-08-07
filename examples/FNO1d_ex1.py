from libs_path import *
from libs import *
import time

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

class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super(SpectralConv1d, self).__init__()

        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  #Number of Fourier modes to multiply, at most floor(N/2) + 1

        self.scale = (1 / (in_channels*out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1)//2 + 1,  device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :self.modes1] = self.compl_mul1d(x_ft[:, :, :self.modes1], self.weights1)

        #Return to physical space
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x

class FNO1d(nn.Module):
    def __init__(self, modes, width):
        super(FNO1d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the initial condition and location (a(x), x)
        input shape: (batchsize, x=s, c=2)
        output: the solution of a later timestep
        output shape: (batchsize, x=s, c=1)
        """

        self.modes1 = modes
        self.width = width
        self.fc0 = nn.Linear(2, self.width) # input channel is 2: (a(x), x)

        self.conv0 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv1 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv2 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv3 = SpectralConv1d(self.width, self.width, self.modes1)
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)


        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):

        x = self.fc0(x)
        x = x.permute(0, 2, 1)

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.relu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.relu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.relu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

def main():
    
    args = get_args_1d()
    cuda = not args.no_cuda and torch.cuda.is_available()
    kwargs = {'pin_memory': True} if cuda else {}
    get_seed(args.seed, printout=False)

    data_path = os.path.join(DATA_PATH, 'burgers_data_R10.mat')
    train_dataset = BurgersDataset(subsample=args.subsample,
                                   train_data=True,
                                   train_portion=0.5,
                                   data_path=data_path,)

    valid_dataset = BurgersDataset(subsample=args.subsample,
                                   train_data=False,
                                   valid_portion=100,
                                   data_path=data_path,)
    
    
   
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              drop_last=True, **kwargs)
    valid_loader = DataLoader(valid_dataset, batch_size=args.val_batch_size, shuffle=False,
                              drop_last=False, **kwargs)

    ntrain = len(train_loader)*args.batch_size
   
    sample = next(iter(train_loader))

    print('='*20, 'Data loader batch', '='*20)
    for key in sample.keys():
        print(key, "\t", sample[key].shape)
    print('='*(40 + len('Data loader batch')+2))
    
    lr = 1e-3
    epochs = 100
    
    model = FNO1d(modes=32, width=128).cuda()
    
    print("num:", get_num_params(model))
    myloss =  LpLoss(size_average=False)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = OneCycleLR(optimizer, max_lr=lr, div_factor=1e4,
                           pct_start=0.2,
                           final_div_factor=1e4,
                           steps_per_epoch=len(train_loader), epochs=epochs)
    

    
    for ep in range(epochs):
        train_mse = 0
        train_l2 = 0
        for batch in train_loader:
            node = batch["node"].cuda()
            grid = batch["grid"].cuda()
            u = batch["target"][:,:,0].cuda()
            a = torch.concat([node, grid], dim= 2)
            
            optimizer.zero_grad()
            
            out = model(a)
            
            mse = F.mse_loss(out.view(args.batch_size, -1), u.view(args.batch_size, -1), reduction='mean')
            
            mse.backward()
            l2 = myloss(out.view(args.batch_size, -1), u.view(args.batch_size, -1))
           
            optimizer.step()
            scheduler.step()
            train_mse += mse.item()
            train_l2 += l2.item()
            
        model.eval()
        test_l2 = 0.0
        with torch.no_grad():
            for batch in valid_loader:
                node = batch["node"].cuda()
                grid = batch["grid"].cuda()
                u = batch["target"][:,:,0].cuda()
                a = torch.concat([node, grid], dim= 2)
                out = model(a)
                # out = y_normalizer.decode(out.view(batch_size, -1))
                test_l2 += myloss(out.view(args.val_batch_size, -1), u.view(args.val_batch_size, -1)).item()
                
        train_mse /= len(train_loader)
        train_l2 /= len(train_dataset)
        test_l2 /= len(valid_dataset)
        
        print("Epoch: %d,  Train Loss: %.3e, Train l2: %.3e, Test l2: %.3e" 
                  % ( ep, train_mse, train_l2, test_l2) )
       
    
                
            
    
if __name__=="__main__":
    main()