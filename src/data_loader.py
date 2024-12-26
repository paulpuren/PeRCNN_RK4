'''
Data loader function for loading the datasets
'''

import numpy as np
from utils import *
import torch
import scipy.io as sio
from torch.distributions import normal

def add_noise(truth, pec=0.05):
    '''
    Adding noise to the ground truth data
    
    Args: 
    -----
    truth: torch tensor, shape: [t,c,h,w] or [t,c,h,w,d]
    '''

    # uv = [truth[:,0:1,...], truth[:,1:2,...]]
    uv_noi = []

    for uvidx in range(truth.shape[1]):
        tru = truth[:, uvidx:uvidx+1]
        n_distr = normal.Normal(0.0, 1.0)
        R = n_distr.sample(tru.shape)

        std_R = torch.std(R)          # std of samples
        std_T = torch.std(tru)

        # define the noise
        noise = R * std_T / std_R * pec

        # add noise channel-wisely
        uv_noi.append(tru + noise)

    return torch.cat(uv_noi, dim=1)


def gs2d(args, train = True):
    '''
    Data loader for 2D GS dataset

    Returns:
    X: the sparse
    '''

    # the steps we will use
    time_steps = 100

    # domain info
    dt = 1.0 / 2
    dx = 1.0 / 100
    dy = 1.0 / 100

    # true coeffs: [mu_u, mu_v, C_1, C_2, C_F, C_k]
    coeff = [2E-5, 5E-6, -1.00, 1.00, 0.04, 0.06]
    info = {"time_steps": time_steps, "dt": dt, "dx": dx, "dy": dy, "coeff": coeff}

    # load the dataset
    data = np.load(args.data_dir + 'gs2d_uv_2x401x100x100_dt=0.5.npy')
    data = np.transpose(data, (1, 0, 2, 3))

    # add noise
    data_noise = add_noise(torch.tensor(data), pec=args.noise)

    # get initial condition
    X = data_noise[0:1, :, ::args.s_upscale, ::args.s_upscale]

    # get the ground truth
    if train == True:
        # for training
        y = data_noise[:(time_steps+1):args.t_upscale, :, ::args.s_upscale, ::args.s_upscale]
    else:
        # for testing
        y = data[:time_steps+1,...]

    return X, y, info


def gs2d_para(args, train = True):
    '''
    Data loader for parametric 2D GS dataset

    Returns:
    X: the sparse
    '''

    # the steps we will use
    time_steps = 100 #400

    # domain info
    dt = 1.0 / 2
    dx = 1.0 / 100
    dy = 1.0 / 100

    # true coeffs: [mu_u, mu_v, C_1, C_2, C_F, C_k_A]
    coeff = [2E-5, 5E-6, -1.00, 1.00, 0.04, 0.063]
    info = {"time_steps": time_steps, "dt": dt, "dx": dx, "dy": dy, "coeff": coeff}

    # load the dataset
    data = np.load(args.data_dir + 'gs2d_para_uv_2x401x100x100_dt=0.5.npy')
    data = np.transpose(data, (1, 0, 2, 3))

    # add noise
    data_noise = add_noise(torch.tensor(data), pec=args.noise)

    # get initial condition
    X = data_noise[0:1, :, ::args.s_upscale, ::args.s_upscale]

    # get the ground truth
    if train == True:
        # for training
        y = data_noise[:(time_steps+1):args.t_upscale, :, ::args.s_upscale, ::args.s_upscale]
    else:
        # for testing
        y = data[:time_steps+1,...]

    return X, y, info


def burgers(args, train = True):
    '''
    Data loader for 2D Burgers dataset

    Returns:
    X: the sparse
    '''

    # the steps we will use
    time_steps = 400

    # domain info    
    dt = 0.00025
    dx = 1.0 / 100
    dy = 1.0 / 100 

    # true coeffs: [vis, UA, UB, VA, VB]
    coeff = [ 0.005, -1.00, -1.00, -1.00, -1.00]
    info = {"time_steps": time_steps, "dt": dt, "dx": dx, "dy": dy, "coeff": coeff}

    data = np.load(args.data_dir + 'burgers_uv_501x2x100x100_dt=0.00025_r=200.npy')

    # add noise
    data_noise = add_noise(torch.tensor(data), pec=args.noise)

    # get initial condition
    X = data_noise[0:1, :, ::args.s_upscale, ::args.s_upscale]

    # get the ground truth
    if train == True:
        # for training
        y = data_noise[:(time_steps+1):args.t_upscale, :, ::args.s_upscale, ::args.s_upscale]
    else:
        # for testing
        y = data[:time_steps+1,...]

    return X, y, info


def heat(args, train = True):
    '''
    Data loader for 2D Heat equation dataset

    Returns:
    X: the sparse
    '''

    # the steps we will use
    time_steps = 200

    # domain info
    dt = 0.0005
    dx = 5.0 / 100
    dy = 5.0 / 100

    # true coeffs: [C_k]
    coeff = [0.1]
    info = {"time_steps": time_steps, "dt": dt, "dx": dx, "dy": dy, "coeff": coeff}

    # load the dataset
    data = np.load(args.data_dir + 'heat_conduct01_dt.npy')

    data = torch.from_numpy(data)[100:,...].unsqueeze(1).double()
    
    # add noise
    data_noise = add_noise(torch.tensor(data), pec=args.noise)

    # get initial condition
    X = data_noise[0:1, :, ::args.s_upscale, ::args.s_upscale]

    # get the ground truth
    if train == True:
        # for training
        y = data_noise[:(time_steps+1):args.t_upscale, :, ::args.s_upscale, ::args.s_upscale]
    else:
        # for testing
        y = data[:time_steps+1,...] # [101,2,100,100]

    return X, y, info

if __name__ == '__main__':
    import argparse
    args = argparse.Namespace()
    args.data_dir = './burgers/'
    args.s_upscale = 4
    args.t_upscale = 2
    args.noise = 0.05
    X, y, info = burgers(args, True)
    print(X.shape, y.shape)
    print(info)