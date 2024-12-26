'''training function'''

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import argparse
import time
import matplotlib.pyplot as plt
from utils import *
from src.data_loader import *
import neptune
import os


COEFF_BY_IC = {'gs2d': ["mu_u", "mu_v", "C_1", "C_2", "C_F", "C_k"],
               'gs2d_para': ["mu_u", "mu_v", "C_1", "C_2", "C_F", "C_k_A"],
               'burgers': ["vis", "UA", "UB", "VA", "VB"],
               'heat': ["C_k"]}


def pretrain_upscaler(args, model_upscaler, init_state_low, run, plotFlag=True):
    '''
    pretraining the upscaler
    '''

    t, c, h_lr, w_lr = init_state_low.shape
    h_hr, w_hr = int(h_lr * args.s_upscale), int(w_lr * args.s_upscale)

    init_state_low = init_state_low.to(args.device)
    
    # using bicubic upscale
    init_state_bicubic = F.interpolate(init_state_low, (h_hr, w_hr), mode='bicubic')

    optimizer = optim.Adam(model_upscaler.parameters(), lr = 0.02)
    scheduler = StepLR(optimizer, step_size=100, gamma=0.99)
    mse_loss = nn.MSELoss()

    for epoch in range(3000): #3000
        optimizer.zero_grad()
        init_state_pred = model_upscaler(init_state_low)

        # use bicubic-interpolated results as a constraint
        loss = mse_loss(init_state_pred, init_state_bicubic)
        loss.backward(retain_graph=True)

        print('[%d] loss: %.9f' % ((epoch+1), loss.item()))
        optimizer.step()
        scheduler.step()

    if plotFlag:
        init_state_upconv = model_upscaler(init_state_low)
        x = np.linspace(-0.5, 0.5, 100)
        y = np.linspace(-0.5, 0.5, 100)
        x_star, y_star = np.meshgrid(x, y)
        u_bicubic = init_state_bicubic[0, 0, ...].detach().cpu().numpy()
        v_bicubic = init_state_bicubic[0, 1, ...].detach().cpu().numpy()
        u_upconv  = init_state_upconv[0, 0, :, :].detach().cpu().numpy()
        v_upconv  = init_state_upconv[0, 1, :, :].detach().cpu().numpy()
        size = int(u_bicubic.shape[-1]/init_state_low.shape[-1])
        u_low_res = np.kron(init_state_low[0, 0, ...].detach().cpu().numpy(), np.ones((size, size)))
        v_low_res = np.kron(init_state_low[0, 1, ...].detach().cpu().numpy(), np.ones((size, size)))
        fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(10, 6))
        fig.subplots_adjust(hspace=0.2, wspace=0.2)
        #
        cf = ax[0, 0].scatter(x_star, y_star, c=u_bicubic, alpha=0.9, edgecolors='none', cmap='hot', marker='s', s=4, vmin=0, vmax=1)
        ax[0, 0].axis('square')
        ax[0, 0].set_xlim([-0.5, 0.5])
        ax[0, 0].set_ylim([-0.5, 0.5])
        ax[0, 0].set_xticks([])
        ax[0, 0].set_yticks([])
        ax[0, 0].set_title('u-Bicubic')
        fig.colorbar(cf, ax=ax[0, 0], fraction=0.046, pad=0.04)
        #
        cf = ax[0, 1].scatter(x_star, y_star, c=u_upconv, alpha=0.9, edgecolors='none', cmap='hot', marker='s', s=4, vmin=0, vmax=1)
        ax[0, 1].axis('square')
        ax[0, 1].set_xlim([-0.5, 0.5])
        ax[0, 1].set_ylim([-0.5, 0.5])
        ax[0, 1].set_xticks([])
        ax[0, 1].set_yticks([])
        ax[0, 1].set_title('u-UpConv')
        fig.colorbar(cf, ax=ax[0, 1], fraction=0.046, pad=0.04)
        #
        cf = ax[0, 2].scatter(x_star, y_star, c=u_low_res, alpha=0.9, edgecolors='none', cmap='hot', marker='s', s=4, vmin=0, vmax=1)
        ax[0, 2].axis('square')
        ax[0, 2].set_xlim([-0.5, 0.5])
        ax[0, 2].set_ylim([-0.5, 0.5])
        ax[0, 2].set_xticks([])
        ax[0, 2].set_yticks([])
        ax[0, 2].set_title('u-LowRes')
        fig.colorbar(cf, ax=ax[0, 2], fraction=0.046, pad=0.04)
        #
        cf = ax[1, 0].scatter(x_star, y_star, c=v_bicubic, alpha=0.9, edgecolors='none', cmap='hot', marker='s', s=4, vmin=0, vmax=1)
        ax[1, 0].axis('square')
        ax[1, 0].set_xlim([-0.5, 0.5])
        ax[1, 0].set_ylim([-0.5, 0.5])
        ax[1, 0].set_xticks([])
        ax[1, 0].set_yticks([])
        ax[1, 0].set_title('v-Bicubic')
        fig.colorbar(cf, ax=ax[1, 0], fraction=0.046, pad=0.04)
        # #
        cf = ax[1, 1].scatter(x_star, y_star, c=v_upconv, alpha=0.9, edgecolors='none', cmap='hot', marker='s', s=4, vmin=0, vmax=1)
        ax[1, 1].axis('square')
        ax[1, 1].set_xlim([-0.5, 0.5])
        ax[1, 1].set_ylim([-0.5, 0.5])
        ax[1, 1].set_xticks([])
        ax[1, 1].set_yticks([])
        ax[1, 1].set_title('v-UpConv')
        fig.colorbar(cf, ax=ax[1, 1], fraction=0.046, pad=0.04)
        #
        cf = ax[1, 2].scatter(x_star, y_star, c=v_low_res, alpha=0.9, edgecolors='none', cmap='hot', marker='s', s=4, vmin=0, vmax=1)
        ax[1, 2].axis('square')
        ax[1, 2].set_xlim([-0.5, 0.5])
        ax[1, 2].set_ylim([-0.5, 0.5])
        ax[1, 2].set_xticks([])
        ax[1, 2].set_yticks([])
        ax[1, 2].set_title('v-LowRes')
        fig.colorbar(cf, ax=ax[1, 2], fraction=0.046, pad=0.04)
        #
        if args.neptune_status == 'async':
            run['train/pretrained_upscalar'].upload(fig)
        else:
            plt.savefig('./pretrained_upscalar.png', dpi=200)
        plt.close('all')

def pretrain_upscaler_3d(args, Upconv, init_state_low, epoch=10000):
    '''
    :param Upconv: upscalar model
    :param init_state_low: low resolution measurement
    :return:
    '''
    init_state_low = init_state_low.double().to(args.device)

    init_state_trilinear = F.interpolate(init_state_low, (48, 48, 48), mode='trilinear')
    optimizer = optim.Adam(Upconv.parameters(), lr = 0.02)
    scheduler = StepLR(optimizer, step_size=100, gamma=0.98)
    mse_loss = nn.MSELoss()
    
    for epoch in range(epoch):
        optimizer.zero_grad()
        init_state_pred = Upconv(init_state_low)
        loss = mse_loss(init_state_pred, init_state_trilinear)
        loss.backward(retain_graph=True)
        print('[%d] loss: %.9f' % ((epoch+1), loss.item()))
        optimizer.step()
        scheduler.step()

    torch.save({
            'model_state_dict': Upconv.state_dict(),
        }, args.data_dir + 'model/upconv.pt')
    

def train(args, model, X, y, model_name, run, optimizer=None, scheduler=None):

    '''
    Args:
    ----
    X: torch tensor, shape: [1, c, h/s_upscale, w/s_upscale]
        sparse and noisy initial condition
    y: torch tensor, shape: [t/t_upscale, c, h/s_upscale, w/s_upscale]
        sparse and noisy measurement data
    model_name: str
    '''
    
    train_loss_list = []
    best_loss = np.inf

    if optimizer is None:
        optimizer = optim.Adam(model.parameters(), lr = args.lr)
    if scheduler is None:
        scheduler = StepLR(optimizer, step_size=200, gamma=0.98) # step_size=500, gamma=0.99) # StepLR(optimizer, step_size=400, gamma=0.8) #step_size=200, gamma=0.97)
    mse_loss = nn.MSELoss().cuda()

    # transfer to gpu
    X = X.double().to(args.device)
    y = y.double().to(args.device)

    total_time = 0

    for epoch in range(args.start_epoch, args.epochs):
        optimizer.zero_grad()

        # curr_time = time.time()
        
        output = model(X, method=args.method) #, method='rk4')  # [t,c,h,w] in high-res
        
        # per_epoch_time = time.time() - curr_time
        # total_time += per_epoch_time
        # print(total_time/(epoch+1))

        if len(output.shape) == 5:
            loss = mse_loss(output[::args.t_upscale, :, ::args.s_upscale, ::args.s_upscale, ::args.s_upscale], y)
        else:
            loss = mse_loss(output[::args.t_upscale, :, ::args.s_upscale, ::args.s_upscale], y)

        # print(output[::args.t_upscale, :, ::args.s_upscale, ::args.s_upscale, ::args.s_upscale].shape)
        # print(torch.any(torch.isnan(loss)))
        loss.backward()
        model.to(args.device)

        optimizer.step()
        scheduler.step()

        run['total_time'].append(total_time)

        # print loss in each epoch
        if args.neptune_status == 'async':
            run['train/loss'].append(loss.item())
        else:
            print('[%d/%d %d%%] loss: %.12f' % ((epoch+1), args.epochs, ((epoch+1)/args.epochs*100.0), loss.item()))
        train_loss_list.append(loss.item())

        for param_group in optimizer.param_groups:
            print('lr=', param_group['lr'])
            run["train/lr"].append(param_group['lr'])

        # save model
        if (epoch+1)%1 == 0 and loss.item() < best_loss:
            # show learning rate
            best_loss = loss.item()

            # Save checkpoint
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': epoch,
            }, args.data_dir + 'model/' + model_name[:-3] + '_val.pt')
                
        # Save checkpoint
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'epoch': epoch,
        }, args.data_dir + 'model/' + model_name)

        # Record coef. history
        # model.rcnn_cell.show_coef()
        for ic, coeff in enumerate(model.rcnn_cell.show_coef()):
            run[f'train/{COEFF_BY_IC[args.data_name][ic]}'].append(coeff)
            
    return train_loss_list


def main():
    parser = argparse.ArgumentParser(description='training parameters')
    # arguments for data
    parser.add_argument('--data_name', type=str, default='gs2d', help='dataset')
    parser.add_argument('--data_dir', type=str, default='./gs2d/', help='the path to data, models, and figures')
    parser.add_argument('--pretrained', action='store_true', help='load the pretrained model')
    parser.add_argument('--noise', type=float, default=0.0, help='noise ratio')
    parser.add_argument('--s_upscale', type=int, default=4, help='upscale factor for spatial domain')
    parser.add_argument('--t_upscale', type=int, default=2, help='upscale factor for time domain')
    parser.add_argument('--pretrain_modelpath', type=str, default='', help='include model path if the model is restored')
    
    # arguments for training
    parser.add_argument('--epochs', type=int, default=5000, help='max epochs')
    parser.add_argument('--device', type=str, default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), help='computing device')
    parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
    parser.add_argument('--seed', type=int, default=66, help='random seed')
    parser.add_argument('--method', type=str, default='rk2', help='forward function method: rk2/rk4')
    parser.add_argument('--input_kernel_size', type=int, default=5, help='input kernel size for gs2d')
    
    parser.add_argument('--neptune_status', type=str, default='async')
    parser.add_argument('--neptune_tag', type=str)
    args = parser.parse_args()
    print(args)

    # % --- %
    # Set random seed to reproduce the work
    # % --- %
    set_seed(args.seed)

    # % --- %
    # Load data
    # % --- %
    if args.data_name == 'gs2d':
        X, y, info = gs2d(args, train=True)
        from src.gs import RCNN

    elif args.data_name == 'gs2d_para':
        X, y, info = gs2d_para(args, train = True)
        from src.gs_para import RCNN

    elif args.data_name == 'burgers':
        X, y, info = burgers(args, train=True)
        from src.burgers import RCNN
    
    elif args.data_name == 'burgers_traditional':
        X, y, info = burgers(args, train=True)
        from src.burgers_tradition import RCNN
        args.data_name = 'burgers'

    elif args.data_name == 'heat':
        # only one channel
        X, y, info = heat(args, train=True)
        from src.heat import RCNN
        
    else:
        raise ValueError('Dataset {} not recognized'.format(args.data_name))

    # define model
    if args.data_name == 'gs2d' or args.data_name == 'heat':
        model = RCNN(
            dt = info["dt"],
            dx = info["dx"],
            steps = info["time_steps"],
            input_kernel_size=args.input_kernel_size).to(args.device)
    else:
        model = RCNN(
            dt = info["dt"],
            dx = info["dx"],
            steps = info["time_steps"]).to(args.device)
    optimizer = None
    scheduler = None
    neptune_tags = [args.method, args.data_name]
    if hasattr(args, 'neptune_tag') and args.neptune_tag is not None:
        neptune_tags.append(args.neptune_tag)
  
    run = neptune.init_run(
    project="",
    api_token="",
    mode=args.neptune_status,
    tags=neptune_tags,
    )
    
    run_id = run['sys/id'].fetch()
    params = vars(args)
    
    for k, v in info.items():
        if k == 'coeff':
            for ic, cof in enumerate(v):
                params[COEFF_BY_IC[args.data_name][ic]] = cof
        elif k == 'device':
            del params[k]
        else:
            params[k] = v
    run['parameters'] = params
    
    # model = torch.nn.DataParallel(model)
    if not os.path.exists(os.path.join(args.data_dir, 'model')):
        os.makedirs(os.path.join(args.data_dir, 'model'))
    
    # if pretrain and posttune
    model_name = f'{run_id}.pt' #"checkpoint_" + args.data_name + "_noise" + str(args.noise) + ".pt"
    args.start_epoch = 0
    
    if args.pretrained and os.path.exists(args.pretrain_modelpath):
        model, optimizer, scheduler, start_epoch = load_model(model, args.pretrain_modelpath)
        model = model.to(args.device)
        args.start_epoch = start_epoch

    # Model summary
    print(model)    
    print('**** Setup ****')
    print('Total params Generator: %.3fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
    print('************')    

    # pretrain for upscaler
    print('Start pretraining...')
    start = time.time()
    
    if args.data_name == 'gs3d':
        pretrain_upscaler_3d(args, model.upscaler, X, epoch=10000)
    elif args.data_name == 'heat':
        pretrain_upscaler(args, model.upscaler, X, run, plotFlag=False)
    else:
        pretrain_upscaler(args, model.upscaler, X, run, plotFlag=True)
        
    end = time.time()
    print('Pretrain finished. The running time is: ', (end - start))

    print('Start end-to-end training...')
    start = time.time()
    train_loss_list = train(args, model, X, y, model_name, run, optimizer=optimizer, scheduler=scheduler)
    end = time.time()
    print('Train finished. The running time is: ', (end - start))

    # % --- %
    # Post-process: plot train loss
    # % --- %
    if args.neptune_status != 'async':
        x_axis = np.arange(0, args.epochs)
        fig, ax = plt.subplots()
        ax.plot(x_axis, train_loss_list, label = 'train loss')
        ax.set_yscale('log')
        ax.legend()
        plt.savefig(args.data_dir + 'figures/train_loss_' + str(args.data_name) + '_' + str(args.lr) + '_' + str(args.seed) + '.png', dpi = 300)
        plt.close() 
    
    run.stop()

if __name__ =='__main__':
    main()