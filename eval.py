'''training function'''

import numpy as np
import torch
import argparse
import matplotlib.pyplot as plt
from utils import *
from src.data_loader import *


def compute_relative_error(pred, truth):
    nume = torch.norm(pred - truth)
    deno = torch.norm(truth) 
    epsino = nume / deno
    return epsino


def relative_recons_error(truth, pred):
    '''
    Compute the relative reconstruction error

    Args:
    -----
    truth: torch tensor, shape: [t,c,h,w]
        ground truth
    pred: torch tensor, shape: [t,c,h,w]
        reconstructed dynamics
    '''

    recons_err_u = compute_relative_error(pred[:, 0:1,...], truth[:, 0:1,...])
    recons_err_v = compute_relative_error(pred[:, 1:2,...], truth[:, 1:2,...])
    recons_err = compute_relative_error(pred, truth)

    print("Relative full field l2 error: ", recons_err)
    print("Relative l2 error of u: ", recons_err_u)
    print("Relative l2 error of v: ", recons_err_v)

    return recons_err, recons_err_u, recons_err_v


def relative_coeff_error(truth_coeff, pred_coeff):
    '''
    Compute the relative error for identified coefficients
    
    Args:
    -----
    truth_coeff: numpy array, shape: [# coefficients]
    pred_coeff: numpy array, shape: [# coefficients]
    '''

    coeff_err = np.absolute((truth_coeff - pred_coeff) / truth_coeff)
    coeff_err_mean = np.mean(coeff_err) * 100
    coeff_err_std = np.std(coeff_err) * 100 
    
    return coeff_err, coeff_err_mean, coeff_err_std


def viz(pred, truth, coord_info, n_grids, num, fig_save_path):
    ''' 
    Visualize the snapshots

    Args:
    -----
    pred: numpy array, shape: [t,c,h,w]
    truth: numpy array, shape: [t,c,h,w]
    coord_info: list, [xmin, xmax, ymin, ymax] 
    n_grids: int, the number of mesh grids
    num: int, the time step to plot
    fig_save_path: str, the figure path
    '''

    xmin, xmax, ymin, ymax = coord_info
    x = np.linspace(xmin, xmax, n_grids) # [xmin,xmax] = [-0.5, 0.5]
    y = np.linspace(ymin, ymax, n_grids)
    x_star, y_star = np.meshgrid(x, y)
    
    # get the snapshot 
    u_star = truth[num, 0, ...]
    u_pred = pred[num, 0, ...]
    v_star = truth[num, 1, ...]
    v_pred = pred[num, 1, ...]
    
    # plot
    fig, ax = plt.subplots(nrows=2, ncols=2, layout='constrained', figsize=(7, 7))
    # fig.subplots_adjust(hspace=0.3, wspace=0.3)
    #
    cf = ax[0, 0].scatter(x_star, y_star, c=u_pred, alpha=0.9, edgecolors='none', cmap='hot', marker='s', s=5, vmin=0, vmax=1)
    ax[0, 0].axis('square')
    ax[0, 0].set_xlim([xmin, xmax])
    ax[0, 0].set_ylim([ymin, ymax])
    ax[0, 0].set_title('u-RCNN')
    fig.colorbar(cf, ax=ax[0, 0])
    #
    cf = ax[0, 1].scatter(x_star, y_star, c=u_star, alpha=0.9, edgecolors='none', cmap='hot', marker='s', s=5, vmin=0, vmax=1)
    ax[0, 1].axis('square')
    ax[0, 1].set_xlim([xmin, xmax])
    ax[0, 1].set_ylim([ymin, ymax])
    ax[0, 1].set_title('u-Ref.')
    fig.colorbar(cf, ax=ax[0, 1])
    #
    cf = ax[1, 0].scatter(x_star, y_star, c=v_pred, alpha=1.0, edgecolors='none', cmap='hot', marker='s', s=5, vmin=0, vmax=1)
    ax[1, 0].axis('square')
    ax[1, 0].set_xlim([xmin, xmax])
    ax[1, 0].set_ylim([ymin, ymax])
    ax[1, 0].set_title('v-RCNN')
    fig.colorbar(cf, ax=ax[1, 0])
    #
    cf = ax[1, 1].scatter(x_star, y_star, c=v_star, alpha=1.0, edgecolors='none', cmap='hot', marker='s', s=5, vmin=0, vmax=1)
    ax[1, 1].axis('square')
    ax[1, 1].set_xlim([xmin, xmax])
    ax[1, 1].set_ylim([ymin, ymax])
    ax[1, 1].set_title('v-Ref.')
    fig.colorbar(cf, ax=ax[1, 1])
    #
    plt.savefig(fig_save_path + 'uv_comparison_'+str(num).zfill(3)+'.png')
    plt.close('all')


def main():
    parser = argparse.ArgumentParser(description='training parameters')
    # arguments for data
    parser.add_argument('--data_name', type=str, default='gs2d', help='dataset')
    parser.add_argument('--data_dir', type=str, default='./burgers/', help='the path to data, models, and figures')
    parser.add_argument('--noise', type=float, default=0.0, help='noise ratio')
    parser.add_argument('--s_upscale', type=int, default=2, help='upscale factor for spatial domain')
    parser.add_argument('--t_upscale', type=int, default=2, help='upscale factor for time domain')
    parser.add_argument('--input_kernel_size', type=int, default=3, help='input kernel size')
    parser.add_argument('--model_path', type=str, default='', help='include model path')
    
    # arguments for training
    parser.add_argument('--device', type=str, default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), help='computing device')
    parser.add_argument('--seed', type=int, default=66, help='random seed')
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
        X, truth, info = gs2d(args, train=False)
        from src.gs import RCNN
    
    elif args.data_name == 'gs2d_para':
        X, truth, info = gs2d_para(args, train=False)
        from src.gs_para import RCNN
    
    elif args.data_name == 'burgers':
        X, truth, info = burgers(args, train=False)
        from src.burgers import RCNN
    
    elif args.data_name == 'burgers_traditional':
        X, truth, info = burgers(args, train=False)
        from src.burgers_tradition import RCNN
        args.data_name = 'burgers'

    elif args.data_name == 'heat':
        # only one channel
        X, truth, info = heat(args, train=False)
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

    # model = torch.nn.DataParallel(model)
    model_path = args.model_path
    model, _, _, _ = load_model(model, model_path)
    model = model.to(args.device)

    # Model summary
    print(model)    
    print('**** Setup ****')
    print('Total params Generator: %.3fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
    print('************')    

    # test the model
    with torch.no_grad():
        X = X.double().to(args.device)
        pred = model(X) # [t,c,h,w], [101,2,100,100]

    # Padding x, y axis due to periodic boundary condition
    pred = torch.cat((pred, pred[:, :, :, 0:1]), dim=3)
    pred = torch.cat((pred, pred[:, :, 0:1, :]), dim=2)
    pred = pred.cpu().numpy() # [101,2,101,101]

    # [t, 2, 101, 101]
    truth = np.concatenate((truth, truth[:, :, :, 0:1]), axis=3)
    truth = np.concatenate((truth, truth[:, :, 0:1, :]), axis=2) # [101,2,101,101]

    print(pred.shape, truth.shape)

    # compute relative errors of reconstruction
    recons_err, recons_err_u, recons_err_v = relative_recons_error(torch.from_numpy(truth), torch.from_numpy(pred))

    # compute relative errors of coefficients
    pred_coeff = np.asarray(list(model.rcnn_cell.show_coef()))
    true_coeff = info["coeff"]
    coeff_err, coeff_err_mean, coeff_err_std = relative_coeff_error(true_coeff, pred_coeff)
    
    print('rel err')
    print(recons_err, recons_err_u, recons_err_v)
    print('coeff err')
    print(coeff_err, coeff_err_mean, coeff_err_std)


if __name__ =='__main__':
    main()