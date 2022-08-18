from torch.autograd import Variable
import torch
import numpy as np
from torchvision.utils import save_image

def orthogonal_regularization(model, device, beta=1e-4):
    
    loss_orth = torch.tensor(0., dtype=torch.float32, device=device)
    
    for name, param in model.named_parameters():
        if 'weight' in name and param.requires_grad and len(param.shape)==4:
            
            N, C, H, W = param.shape
            
            weight = param.view(N * C, H, W)
            
            weight_squared = torch.bmm(weight, weight.permute(0, 2, 1)) # (N * C) * H * H
            
            ones = torch.ones(N * C, H, H, dtype=torch.float32) # (N * C) * H * H
            
            diag = torch.eye(H, dtype=torch.float32) # (N * C) * H * H
            
            loss_orth += ((weight_squared * (ones - diag).to(device)) ** 2).sum()
            
    return loss_orth * beta

def taxi_input(b_size = 128, test=False, device='cuda'):
    FloatTensor = torch.cuda.FloatTensor if device=='cuda' else torch.FloatTensor
    if test:
        z_noise = Variable(FloatTensor(np.random.uniform(-.8, .8, size=(b_size, 2))))
    else:
        z_noise = Variable(FloatTensor(np.random.uniform(-1.0, 1.0, size=(b_size, 2))))

    y_noise = Variable(FloatTensor(np.concatenate([np.random.uniform(-1.73, 1.73, size=(b_size, 1)), 
                                                    np.random.uniform(-1.74, 1.74, size=(b_size, 1))], axis=1)))
    return z_noise, y_noise

def save_generator_image(image, path, normalized=False):
    """
    Function to save torch image batches
    :param image: image tensor batch
    :param path: path name to save image
    """
    if normalized:
        save_image(image, path)
    else:
        save_image((image+1.0)/2.0, path)