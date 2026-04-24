
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.io as sio

import os
from matplotlib import pyplot as plt
import seaborn as sns
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from timm.layers import drop
from timm.models.layers import DropPath, to_2tuple
import torchvision
import math
from scipy.spatial.distance import cdist


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, outdim=None, seq_len=256, dropout=0.):
        super().__init__()
        if outdim is not None:
            out_dim = outdim
        else:
            out_dim = dim
        self.net1 = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

        theta = np.linspace(0, 2 * np.pi - 2 * np.pi / seq_len, seq_len)
        xc = np.cos(theta).reshape([1, -1])
        yc = np.sin(theta).reshape([1, -1])
        data_ = np.concatenate((xc, yc), axis=0).T
        seq_l = xc.shape[1]

        dist = cdist(data_, data_, metric='euclidean')
        W = np.exp(-dist ** 2 /0.01) #- np.eye(seq_l)  #

        D = np.diag(W.sum(1) ** -0.5)
        L = D @ W @ D + np.eye(seq_l)
        
       
        dtype = self.net1[0].weight.dtype

        
        L = torch.from_numpy(L).to(dtype).unsqueeze(0)
         
        self.register_buffer('L', L)
         

    def forward(self, x):
        b = x.shape[0]
        x = self.net1(self.L @ x)
        return x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()


        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.):
        super().__init__()

        dim_head = dim // heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5



        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, dim * 3, bias=True)

        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):

        L = x.shape[1]
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)  
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, mlp_dim, seq_len=256, heads=8, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, seq_len=seq_len, dropout=dropout)),
            ]))

    def forward(self, x):
        for layer in self.layers:
            x = x + layer[0](x)
            x = x + layer[1](x)
        return x

class CirConv1dbk(nn.Module):
    def __init__(self, dim_in=1, dim=256):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(dim, dim_in, 3))
        self.bias = nn.Parameter(torch.Tensor(dim))
        nn.init.kaiming_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x):
 
        x = F.conv1d(x, self.weight, self.bias,padding=1)
        return x
class CirConv1d(nn.Module):
    def __init__(self, dim_in=1, dim=256):
        super().__init__()

        self.weight = nn.Parameter(torch.Tensor(dim, dim_in, 2))

        self.bias = nn.Parameter(torch.Tensor(dim))
        nn.init.kaiming_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        x = torch.cat([x[:, :, -1::], x, x[:, :, 0:1]], dim=2)

        weight = torch.cat([self.weight, self.weight[:, :, 0:1]], dim=2)
        x = F.conv1d(x, weight, self.bias)
        return x

class InvEqu(torch.nn.Module):
    def __init__(self, dim=256):
        super().__init__()

        self.dim = dim
        self.outsize = dim

        self.cc = CirConv1d(dim_in=1, dim=dim)

        self.pool = nn.AvgPool1d(16, 16)
        self.upsize = dim + 1

        upTimes = int(np.log2(dim // 16))

        self.upList = nn.ModuleList([])
        for i in range(upTimes):
            self.upList.append(nn.Sequential(
                nn.Upsample(size=2 ** (i + 5) + 1, mode='linear', align_corners=True),
                CirConv1d(dim_in=dim, dim=dim), 
                nn.SiLU(),
                
            ))



    def forward(self, v1):
        # v1   [b,16,16]
        # 3D   [b,Channels,Length]
        v1 = v1.squeeze(1)
        b, l = v1.shape[0], v1.shape[1]
        v1 = v1.reshape([b * v1.shape[1], 1, v1.shape[2]])#[b*16,1,16]

        o1 = self.cc(v1)       #[b*16,256,16]

        o1 = self.pool(o1)     #[b*16,256,1]######
        o1 = o1.reshape([b, l, self.dim])  # #[b,16,256]
        
        
        # align the angle of EIT and angle on Polar space
        o1 = o1.flip(1)

        o1 = torch.cat([o1, o1[:, 0:1, :]], dim=1)#[b,17,256]
        o1 = o1.permute(0, 2, 1)#[b,256,17] 


        #[b,256,17]==>33--65--129--257
        for layer in self.upList:
            o1 = layer(o1)

        o1 = o1.permute(0, 2, 1)  # [b,257,256]
        o1 = o1[:, :-1, :]  # [b,256,256] [b,L,dim]
        l=o1.shape[1]
 

        o1 = torch.roll(o1, shifts=l//16//2+l//4, dims=1) 

        return o1





class ModelT(nn.Module):
    def __init__(self, dim=384, modelname='rre'):
        super().__init__()
        ## 1 InvEqu
        #  input  [b, 16, 16]
        in_size = 128
        resolution = 256
        self.outsize = in_size
        self.equBlkNum = 1

        self.catconv = nn.Conv2d(self.equBlkNum + 1, 1, kernel_size=1)

        depth =10
        mlp_dim = dim

        self.equivList = nn.ModuleList([])

        for i in range(self.equBlkNum):
            self.equivList.append(InvEqu(dim=resolution))

        self.modelname = modelname
        model = nn.ModuleList([])


        for i in range(depth):
            model.append(Transformer(dim, 1, mlp_dim, seq_len=resolution, heads=8, dropout=0.0))


        self.in_linear = nn.Linear(resolution, dim)
        self.out_linear = nn.Linear(dim, resolution)

        self.model = nn.Sequential(*model)
        self.p2c = PolarToCartesianImage()
        self.c2p = CartesianToPolarImage()
        self.initialize()

    def initialize(self):

        for m in self.named_modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                nn.init.kaiming_uniform_(m.weight.data)
                try:
                    nn.init.zeros_(m.bias.data)
                except:
                    pass

            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                try:
                    nn.init.zeros_(m.bias.data)
                except:
                    pass

            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1.0)
                m.bias.data.fill_(0)

            elif isinstance(m, nn.LayerNorm):
                m.weight.data.fill_(1.0)
                m.bias.data.fill_(0)

     
    def forward(self, x, TR):

        z = [self.equivList[i](x) for i in range(self.equBlkNum)]
        x = torch.stack(z, dim=1)

        b, N, L, C = x.shape

 
        TR = self.c2p(TR)
        x =  torch.cat((x, TR), dim=1)  
        if self.modelname == 'cyc':
            x = self.catconv(x)  # [b,1,L,C]=[b,1,256,256]
            x = self.in_linear(x)
            x = self.model(x)
            x = self.out_linear(x)

        ## attention
        if self.modelname == 'rre':
            x = self.catconv(x)
            x = x.squeeze(1)  # [b,L,C]=[b,256,256]
            x = self.in_linear(x)
            x = self.model(x)
            x = self.out_linear(x).unsqueeze(1)

        return x


class CartesianToPolarImage(nn.Module):
    def __init__(self, output_size=(256, 256)):

        super(CartesianToPolarImage, self).__init__()
        self.output_size = output_size

    def forward(self, cartesian_image):

        N, C, H, W = cartesian_image.shape
        radius, angle = self.output_size

        r = torch.linspace(0, 1, radius, device=cartesian_image.device)  
        r = torch.sqrt(r)

        theta = torch.linspace(0, 2 * torch.pi-2 * torch.pi/angle, angle, device=cartesian_image.device) 
        
        rv, thetav = torch.meshgrid(r, theta, indexing="xy")


        xv = rv * torch.cos(thetav)
        yv = rv * torch.sin(thetav)


        xv = (xv)  
        yv = (yv)  

        grid = torch.stack((xv, yv), dim=-1) 
        grid = grid.unsqueeze(0).repeat(N, 1, 1, 1) 

        polar_image = F.grid_sample(cartesian_image, grid, mode='bilinear', padding_mode='zeros', align_corners=True)

        return polar_image


class PolarToCartesianImage(nn.Module):
    def __init__(self, output_size=(128, 128)):

        super(PolarToCartesianImage, self).__init__()
        self.output_size = output_size

    def forward(self, polar_image):
        polar_image = polar_image.permute(0, 1, 3, 2)
        N, C, H, W = polar_image.shape
        H_out, W_out = self.output_size


        x = torch.linspace(-1, 1, W_out, device=polar_image.device) 
        y = torch.linspace(-1, 1, H_out, device=polar_image.device)  
        xv, yv = torch.meshgrid(x, y, indexing="xy")


        r = torch.sqrt(xv ** 2 + yv ** 2)  

        r = r ** 2
        

        theta = torch.atan2(yv, xv)  
        theta = (theta + 2 * torch.pi) % (2 * torch.pi) 

        

        theta = theta / (2 * torch.pi)  
         

        grid = torch.stack((theta * 2 - 1, r * 2 - 1), dim=-1)  
        grid = grid.unsqueeze(0).repeat(N, 1, 1, 1)  

        polar_image=torch.cat([polar_image,polar_image[:,:,:,0:1]],dim=3)

        cartesian_image = F.grid_sample(polar_image, grid, mode='bilinear', padding_mode='zeros', align_corners=True)

        return cartesian_image


 



