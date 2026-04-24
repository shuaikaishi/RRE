import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
import scipy.io as sio
import torch.nn.functional as F
from matplotlib import pyplot as plt

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class EITdataset(Dataset):

    def __init__(self, data_file_path, modelname=None, dataset='simulate'):
        modelnameList = ['ImprovedLeNet', 'CNNEIM', 'SADBnet', 'SAHFL', 'EcNet', 'DHUnet', 'DEIT']
        assert modelname in modelnameList, \
            'modelname should be in ' + str(modelnameList)
        self.filenames = os.listdir(data_file_path)

        self.filenames.sort()
        
        # self.filenames = self.filenames[0:2]
        self.data_file_path = data_file_path
        self.modelname = modelname
        # self.scattering = Scattering1D(J=3, Q=8, shape=208)
        if dataset == 'simulate':
            self.voltage = 1
            self.current = 1
        elif dataset == 'data2017':
            self.voltage = 1.040856e3  # V2/V1
            self.current = 2  # I2/I1
        elif dataset == 'data2023':
            self.voltage = 1978  # V2/V1
            self.current = 2  # I2/I1
        elif dataset == 'data2024':
            self.voltage = 3022e4  # V2/V1
            self.current = 8  # I2/I1 =8 mA/1 mA
        else:
            self.voltage = 1
            self.current = 1
        self.mean = torch.load(self.data_file_path + '../mean.pth', weights_only=True)
        self.std = torch.load(self.data_file_path + '../std.pth', weights_only=True)
    def toEIM(self, voltage):
        '''

        Args:
            voltage: [1,16,13]

        Returns:
            eim:[1,16,16] EIM map

        '''

        num = 16  # num of electrodes
        eim = torch.zeros(1, num, num)


        for i in range(num):

            zero_positions = [(i + j) % num for j in range(3)]

            row = zero_positions[1]


            non_zero_index = 0
            for j in range(num):
                idx = (j) % num

                if idx not in zero_positions:
                    eim[:, row, idx] = voltage[:, row, non_zero_index]
                    non_zero_index += 1

        return eim

    def EIMtoEIV(self, voltage):
        '''
        Args:
            voltage: [1,16,16]

        Returns:
            eiv: [104]
        '''
        num = 16
        idx = 0
        eiv = torch.zeros([num * (num - 3) // 2])
        for i in range(num):
            for j in range(num):
                if j > i + 1:
                    if i == 0 and j == num - 1:
                        # print(i, j)
                        continue
                    # print(i, j, idx)
                    eiv[idx] = voltage[0, i, j]
                    idx += 1
        return eiv

    def norm(self, ys):
        assert self.modelname in ['ImprovedLeNet', 'CNNEIM', 'SADBnet', 'SAHFL', 'DEIT']
        return (ys - self.mean) / self.std

    def c2p(self, x):
        # for ground truth GT
        # for TR
        b = x.shape[0]
        R = 1  # img.shape[1] // 2
        C = 256
        L = 256

        theta = np.linspace(L - 1, 0, L)
        theta = theta / L * 2 * np.pi
        theta = np.roll(theta, L // 4 + 1)
        r = np.linspace(0, R, C)

        theta = torch.from_numpy(theta)
        r = torch.from_numpy(r)
        grid = torch.meshgrid(theta, r, indexing='ij')
        grid = [grid[1] * np.cos(grid[0]), grid[1] * np.sin(grid[0])]
        grid = torch.stack(grid, dim=2).unsqueeze(0).float().repeat(b, 1, 1, 1).to(x.device)

        output = F.grid_sample(x, grid, mode='bilinear', align_corners=True)
        return output

    def __getitem__(self, index):
        data_file = self.data_file_path + self.filenames[index]
        modelname = self.modelname


        ##
        d = sio.loadmat(data_file)
        xs_all = d['img']
        dv = d['vi'] - d['vh']
        xs_TR_all = d['TR']
        ##



        ##
        # d = np.load(data_file)
        # xs_all = d['xs']
        # xs_TR_all = d['TR']
        # dv = d['ys']
        ##
        # x_inv = d['xs_gn']


        ys_all = dv[:, 0] / self.voltage


        xs = torch.from_numpy(xs_all).float().unsqueeze(0)

        if modelname == 'ImprovedLeNet':
            ys = torch.from_numpy(ys_all).float()
            ys = ys.reshape([1, 16, 13])
            ys = self.norm(ys)
        elif modelname == 'CNNEIM' or modelname == 'DEIT':
            ys = torch.from_numpy(ys_all).float()
            ys = ys.reshape([1, 16, 13])
            ys = self.norm(ys)
            ys_st = torch.from_numpy(xs_TR_all).float().unsqueeze(axis=0) * self.voltage / self.current
            ys = self.toEIM(ys)




        elif modelname == 'SADBnet' or modelname == 'SAHFL':
            ys = torch.from_numpy(ys_all).float()
            ys = ys.reshape([1, 16, 13])
            ys = self.norm(ys)

            ys = self.toEIM(ys)
            ys = self.EIMtoEIV(ys)


        elif modelname == 'EcNet' or modelname == 'DHUnet':
            ys = torch.from_numpy(xs_TR_all).float().unsqueeze(axis=0) * self.voltage / self.current

        return ys, ys_st, xs  # x_inv, x, y, xs_TR

    def __len__(self):
        return len(self.filenames)


