'''
    1. Physics-encoded Recurrent Conv Neural Network for scalar coefficients discovery.
    2. Different from the ref paper that is based on Forward Euler scheme, this network is based on RK4 scheme.
    3. 2D Gray-scott RD equation is used as an example here.
'''

import torch
import torch.nn as nn
import numpy as np
from src.upsample import upscaler
from utils import *
from prettytable import PrettyTable

torch.set_default_dtype(torch.float64)

# Discrete 2D Laplacian operator
# lap_2d_op = [[[[    0,   0, -1/12,   0,     0],
#                [    0,   0,   4/3,   0,     0],
#                [-1/12, 4/3,    -5, 4/3, -1/12],
#                [    0,   0,   4/3,   0,     0],
#                [    0,   0, -1/12,   0,     0]]]]

# lap_2d_op =  [[[[0, 0, 0, -1, 0, 0, 0],
#                 [0, 0, 0, 9, 0, 0, 0],
#                 [0, 0, 0, -45, 0, 0, 0],
#                 [-1, 9, -45, 0, 45, -9, 1],
#                 [0, 0, 0, 45, 0, 0, 0],
#                 [0, 0, 0, -9, 0, 0, 0],
#                 [0, 0, 0, 1, 0, 0, 0]]]]
# #(1/60) *

lap_2d_op =  [[[[0, 0, 0, 2, 0, 0, 0],
                [0, 0, 0, -27, 0, 0, 0],
                [0, 0, 0, 270, 0, 0, 0],
                [2, -27, 270, -980, 270, -27, 2],
                [0, 0, 0, 270, 0, 0, 0],
                [0, 0, 0, -27, 0, 0, 0],
                [0, 0, 0, 2, 0, 0, 0]]]]
#(1/180) *





class RCNNCell(nn.Module):
    '''
        Physics-encoded Recurrent Conv Neural Network Cell,
        Completely based on physics (only scalar coefficients are unknown) and RK4 scheme
    '''
   
    def __init__(self, dt=0.5, dx=0.01):

        super(RCNNCell, self).__init__()

        # the initial parameters
        self.input_kernel_size = 7
        self.input_stride = 1
        self.input_padding = self.input_kernel_size//2

        # Ground truth
        # self.DA = 2 * 10 ** -5
        # self.DB = self.DA / 4
        # self.f = 1 / 25
        # self.k = 3 / 50

        self.mu_up = 4.0e-5  # upper bound for the diffusion coefficient
        self.CA = torch.nn.Parameter(torch.tensor(-np.random.rand()+0.5, dtype=torch.float64), requires_grad=True)
        self.CB = torch.nn.Parameter(torch.tensor(-np.random.rand()+0.5, dtype=torch.float64), requires_grad=True)
        self.NA = torch.nn.Parameter(torch.tensor(-np.random.rand(), dtype=torch.float64), requires_grad=True)
        self.NB = torch.nn.Parameter(torch.tensor( np.random.rand(), dtype=torch.float64), requires_grad=True)
        self.f  = torch.nn.Parameter(torch.tensor(0.1*np.random.rand(), dtype=torch.float64), requires_grad=True)
        self.k  = torch.nn.Parameter(torch.tensor(0.1*np.random.rand(), dtype=torch.float64), requires_grad=True)

        self.dx = dx # 0.01
        self.dt = dt # 1.0 / 2

        # circular padding mode due to periodic boundary condition
        self.W_laplace = nn.Conv2d(1, 1, self.input_kernel_size, self.input_stride, padding=self.input_padding, padding_mode='circular', bias=False)
        self.W_laplace.weight.data = 1/self.dx**2*torch.tensor(lap_2d_op, dtype=torch.float64)/180
        self.W_laplace.weight.requires_grad = False

    def f_rhs(self, u, v):
        # Compute the right hand side (RHS) of the governing PDE, e.g., f(u) in u_t = f(u)
        f_u = self.mu_up*torch.sigmoid(self.CA)*self.W_laplace(u) + self.NA* u*v**2 + self.f * (1 - u)
        f_v = self.mu_up*torch.sigmoid(self.CB)*self.W_laplace(v) + self.NB* u*v**2 - (self.k+self.f)*v

        return f_u, f_v
    
    def forward_rk2(self, h):

        u0 = h[:, 0:1, ...]
        v0 = h[:, 1:2, ...]

        f_u, f_v = self.f_rhs(u0, v0)

        u_next = u0 + self.dt * f_u
        v_next = v0 + self.dt * f_v

        ch = torch.cat((u_next, v_next), dim=1)

        return ch

    def forward(self, h):
        '''
        Calculate the updated gates forward.
        '''

        # initial condition
        u0 = h[:, 0:1, ...]
        v0 = h[:, 1:2, ...]

        # Stage 1
        k1_u, k1_v = self.f_rhs(u0, v0)

        u1 = u0 + k1_u * self.dt / 2.0
        v1 = v0 + k1_v * self.dt / 2.0

        # Stage 2
        k2_u, k2_v = self.f_rhs(u1, v1)

        u2 = u0 + k2_u * self.dt / 2.0
        v2 = v0 + k2_v * self.dt / 2.0

        # Stage 3
        k3_u, k3_v = self.f_rhs(u2, v2)

        u3 = u0 + k3_u * self.dt
        v3 = v0 + k3_v * self.dt

        # Final stage
        k4_u, k4_v = self.f_rhs(u3, v3)

        u_next = u0 + self.dt * (k1_u + 2 * k2_u + 2 * k3_u + k4_u) / 6.0
        v_next = v0 + self.dt * (k1_v + 2 * k2_v + 2 * k3_v + k4_v) / 6.0

        ch = torch.cat((u_next, v_next), dim=1)

        return ch

    def show_coef(self):
        # Print the current coefficients from the network
        table = PrettyTable()
        table.field_names = ['\\', r"$\mu_u$", r"$\mu_v$", r"$C_1$", r"$C_2$", r"$C_F$", r"$C_k$", ]
        mu_u, mu_v, C_1, C_2, C_F, C_k = self.mu_up*torch.sigmoid(self.CA).item(), \
                                         self.mu_up * torch.sigmoid(self.CB).item(), \
                                         self.NA.item(), self.NB.item(), self.f.item(), self.k.item()
        table.add_row(["True",       2E-5, 5E-6, -1.00, 1.00, 0.04, 0.06])
        table.add_row(["Identified", mu_u, mu_v, C_1, C_2, C_F, C_k])
        print(table)
        return mu_u, mu_v, C_1, C_2, C_F, C_k


class RCNN(nn.Module):
    ''' PeRCNNCell '''
    def __init__(self, dt=0.5, dx=0.01, steps=1):

        super(RCNN, self).__init__()

        self.steps = steps
        self.upscaler = upscaler()
        self.rcnn_cell = RCNNCell(dt, dx)

    def forward(self, init_state_low):

        # upscale intial state
        h = self.upscaler(init_state_low)
        outputs = [h]
        
        # roll-out
        for _ in range(self.steps):

            h = self.rcnn_cell(h)
            outputs.append(h)

        outputs = torch.cat(tuple(outputs), dim=0) # [t,c,h,w]

        return outputs

