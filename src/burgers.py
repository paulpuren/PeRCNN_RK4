'''
    1. Physics-encoded Recurrent Conv Neural Network for scalar coefficients discovery.
    2. Different from the ref paper that is based on Forward Euler scheme, this network is based on RK4 scheme.
    3. 2D Burgers equation is used as an example here.
'''

import torch
import torch.nn as nn
import numpy as np
# from src.upsample import upscalerBurgers
from src.upsample import upscaler
from utils import *
from prettytable import PrettyTable

torch.set_default_dtype(torch.float64)

# Discrete 2D Laplacian operator
lap_2d_op = [[[[    0,   0, -1/12,   0,     0],
               [    0,   0,   4/3,   0,     0],
               [-1/12, 4/3,    -5, 4/3, -1/12],
               [    0,   0,   4/3,   0,     0],
               [    0,   0, -1/12,   0,     0]]]]
# Discrete dx operator
dx_op  = [[[[    0,   0,  1/12,   0,    0],
               [    0,   0, -8/12,   0,    0],
               [    0,   0,     0,   0,    0],
               [    0,   0,  8/12,   0,    0],
               [    0,   0, -1/12,   0,    0]]]]
# Discrete dy operator
dy_op  = [[[[    0,     0,    0,     0,      0],
               [    0,     0,    0,     0,      0],
               [ 1/12, -8/12,    0,  8/12,  -1/12],
               [    0,     0,    0,     0,      0],
               [    0,     0,    0,     0,      0]]]]


class RCNNCell(nn.Module):
    '''
        Physics-encoded Recurrent Conv Neural Network Cell,
        Completely based on physics (only scalar coefficients are unknown) and RK4 scheme
    '''
   
    def __init__(self, dt=0.5, dx=0.01):

        super(RCNNCell, self).__init__()

        # the initial parameters
        self.input_kernel_size = 5
        self.input_stride = 1
        self.input_padding = 0 #self.input_kernel_size//2

        self.mu_up = 1e-3  # upper bound for the diffusion coefficient
        self.Re = torch.nn.Parameter(torch.tensor(np.random.rand(), dtype=torch.float64), requires_grad=True)
        self.UA = torch.nn.Parameter(torch.tensor(np.random.rand(), dtype=torch.float64), requires_grad=True)
        self.UB = torch.nn.Parameter(torch.tensor(np.random.rand(), dtype=torch.float64), requires_grad=True)
        self.VA = torch.nn.Parameter(torch.tensor(np.random.rand(), dtype=torch.float64), requires_grad=True)
        self.VB = torch.nn.Parameter(torch.tensor(np.random.rand(), dtype=torch.float64), requires_grad=True)
        
        self.dx = dx 
        self.dt = dt 

        self.W_laplace = nn.Conv2d(1, 1, self.input_kernel_size, self.input_stride, padding=self.input_padding, bias=False)
        self.W_laplace.weight.data = 1/self.dx**2*torch.tensor(lap_2d_op, dtype=torch.float64)
        self.W_laplace.weight.requires_grad = False
        
        self.dxOp =  nn.Conv2d(1, 1, self.input_kernel_size, self.input_stride, padding=0, bias=False)
        self.dxOp.weight.data = 1 / self.dx  * torch.tensor(dx_op, dtype=torch.float64)
        self.dxOp.weight.requires_grad = False

    def padMethod(self, h):
        # periodic padding
        h_pad_2 = torch.cat((h[:, :, :, -2:], h, h[:, :, :, 0:2]), dim=3)
        h_pad_2 = torch.cat((h_pad_2[:, :, -2:, :], h_pad_2, h_pad_2[:, :, 0:2, :]), dim=2)
        return h_pad_2
    
    def cal_deriv(self,x):
        x_pad = self.padMethod(x)
        res_dx = self.dxOp(x_pad)
        # res_dy = self.dyOp(x_pad)
        res_dy = self.dxOp(x_pad.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
        der = torch.cat((res_dx,res_dy),dim=1)
        return der
 
    def f_rhs(self, u, v):

        # Compute the right hand side (RHS) of the governing PDE, e.g., f(u) in u_t = f(u)
        #cal derivative of u and v
        u_deriv = self.cal_deriv(u)
        v_deriv = self.cal_deriv(v)
        u_x,u_y = u_deriv[:,0:1,...],u_deriv[:,1:2,...]
        v_x,v_y = v_deriv[:,0:1,...],v_deriv[:,1:2,...]

        f_u = self.mu_up * (1 / self.Re) * self.W_laplace(self.padMethod(u)) + self.UA * u * u_x + self.UB * v * u_y
        f_v = self.mu_up * (1 / self.Re) * self.W_laplace(self.padMethod(v)) + self.VA * u * v_x  + self.VB * v * v_y

        return f_u, f_v
    
    def forward(self, h, method='rk4'):

        if method == 'rk2':
            u0 = h[:, 0:1, ...]
            v0 = h[:, 1:2, ...]

            f_u, f_v = self.f_rhs(u0, v0)

            u_next = u0 + self.dt * f_u
            v_next = v0 + self.dt * f_v

            ch = torch.cat((u_next, v_next), dim=1)
        else:
            ch = self.forward_rk4(h)

        return ch

    def forward_rk4(self, h):
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
        from prettytable import PrettyTable
        table = PrettyTable()
        table.field_names = ['\\', r"$\vis$", r"$\UA$", r"$UB$", r"$VA$", r"$VB$", ]
        vis, UA, UB, VA, VB = self.mu_up * (1 / self.Re.item()), \
                                         self.UA.item(), self.UB.item(), self.VA.item(), self.VB.item()
        table.add_row(["True",       0.005, -1.00, -1.00, -1.00, -1.00])
        table.add_row(["Identified", vis, UA, UB, VA, VB])
        print(table)
        return vis, UA, UB, VA, VB


class RCNN(nn.Module):
    ''' PeRCNNCell '''
    def __init__(self, dt=0.5, dx=0.01, steps=1):

        super(RCNN, self).__init__()

        self.steps = steps
        self.upscaler = upscaler()
        self.rcnn_cell = RCNNCell(dt, dx)

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

