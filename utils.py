import numpy as np
import torch

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def save_model(model, model_name, save_path):
    ''' save the model '''
    torch.save(model.state_dict(), save_path + model_name + '.pt')


def load_model(model, model_path):
    # Load model and optimizer state to continue training from a checkpoint
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.97)
    if 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    start_epoch = 0
    if 'epoch' in checkpoint:
        start_epoch = checkpoint['epoch']
    return model, optimizer, scheduler, start_epoch


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def summary_parameters(model):
    for i in model.parameters():
        print(i.shape)


def show_trainable(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.data)