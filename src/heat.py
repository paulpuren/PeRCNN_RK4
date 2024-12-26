'''
    1. Physics-encoded Recurrent Conv Neural Network for scalar coefficients discovery.
    2. Different from the ref paper that is based on Forward Euler scheme, this network is based on RK4 scheme.
    3. 2D Heat RD equation is used as an example here.
'''

import torch
import torch.nn as nn
import numpy as np
from src.upsample import upscalerHeat
from utils import *
from prettytable import PrettyTable

torch.set_default_dtype(torch.float64)

# Discrete 2D Laplacian operator
lap_2d_op = {5: [[[[    0,   0, -1/12,   0,     0],
                   [    0,   0,   4/3,   0,     0],
                   [-1/12, 4/3,    -5, 4/3, -1/12],
                   [    0,   0,   4/3,   0,     0],
                   [    0,   0, -1/12,   0,     0]]]],
             3: [[[[0, 1, 0],
                   [1, -4, 1],
                   [0, 1, 0]]]],
             7: [[[[0, 0, 0, 2/180, 0, 0, 0],
                   [0, 0, 0, -27/180, 0, 0, 0],
                   [0, 0, 0, 270/180, 0, 0, 0],
                   [2/180, -27/180, 270/180, -980/180, 270/180, -27/180, 2/180],
                   [0, 0, 0, 270/180, 0, 0, 0],
                   [0, 0, 0, -27/180, 0, 0, 0],
                   [0, 0, 0, 2/180, 0, 0, 0]]]]}


class RCNNCell(nn.Module):
    '''
        Physics-encoded Recurrent Conv Neural Network Cell,
        Completely based on physics (only scalar coefficients are unknown) and RK4 scheme
    '''
   
    def __init__(self, dt=0.5, dx=0.01, input_kernel_size=5):

        super(RCNNCell, self).__init__()

        # the initial parameters
        self.input_kernel_size = input_kernel_size
        self.input_stride = 1
        self.input_padding = self.input_kernel_size//2

        # Ground truth
        # self.coeff = 0.1
        self.k  = torch.nn.Parameter(torch.tensor(np.random.rand(), dtype=torch.float64), requires_grad=True)
    
        self.dx = dx # 0.05
        self.dt = dt # 0.0005

        self.W_laplace = nn.Conv2d(1, 1, self.input_kernel_size, self.input_stride, padding=self.input_padding, bias=False)
        self.W_laplace.weight.data = 1/self.dx**2*torch.tensor(lap_2d_op[self.input_kernel_size], dtype=torch.float64)
        self.W_laplace.weight.requires_grad = False

    def padMethod(self, h):
        
        # [t,c,h,w]
        # Neumann BC, du / dn = 0
        h[:, :, :, -1] = h[:, :, :, -2] # top
        h[:, :, :, 0] = h[:, :, :, 1] # bot
        h[:, :, 0, :] = h[:, :, 1, :] # left
        h[:, :, -1, :] = h[:, :, -2, :] # right
        
        return h
        
        
    def f_rhs(self, u):
        # Compute the right hand side (RHS) of the governing PDE, e.g., f(u) in u_t = f(u)
        f_u = self.k * self.W_laplace(self.padMethod(u))

        return f_u
    
    
    def forward(self, h, method='rk4'):
        '''
        Calculate the updated gates forward.
        '''
        if method == 'rk2':
            u0 = h[:, 0:1, ...]
            f_u = self.f_rhs(u0)
            u_next = u0 + self.dt * f_u
            ch = u_next

        elif method == 'rk4':
            # initial condition
            u0 = h[:, 0:1, ...]

            # Stage 1
            k1_u = self.f_rhs(u0)

            u1 = u0 + k1_u * self.dt / 2.0

            # Stage 2
            k2_u = self.f_rhs(u1)

            u2 = u0 + k2_u * self.dt / 2.0

            # Stage 3
            k3_u = self.f_rhs(u2)

            u3 = u0 + k3_u * self.dt

            # Final stage
            k4_u = self.f_rhs(u3)

            u_next = u0 + self.dt * (k1_u + 2 * k2_u + 2 * k3_u + k4_u) / 6.0

            ch = u_next

        return ch

    def show_coef(self):
        # Print the current coefficients from the network
        table = PrettyTable()
        table.field_names = ['\\', r"$C_k$"]
        C_k = self.k.item()
        table.add_row(["True",       0.1])
        table.add_row(["Identified", C_k])
        print(table)
        
        return C_k, 


class RCNN(nn.Module):
    ''' PeRCNNCell '''
    def __init__(self, dt=0.5, dx=0.01, steps=1, input_kernel_size=5):

        super(RCNN, self).__init__()

        self.steps = steps
        # self.upscaler = upscaler()
        self.upscaler = upscalerHeat()
        self.rcnn_cell = RCNNCell(dt, dx, input_kernel_size)

    def forward(self, init_state_low, method='rk4'):

        # upscale intial state
        h = self.upscaler(init_state_low)
        outputs = [h]
        
        # roll-out
        for _ in range(self.steps):

            h = self.rcnn_cell(h, method=method)
            outputs.append(h)

        outputs = torch.cat(tuple(outputs), dim=0) # [t,c,h,w]

        return outputs

