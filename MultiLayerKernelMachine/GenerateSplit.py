import pandas as pd
import numpy as np
import time
from tqdm import tqdm

import torch
from torch.utils.data import Dataset,DataLoader
from torch import optim

from .Mydataset import mydataset


def GenerateSplit(split_fit,device_fit,net_fit,lr_fit,momentum_fit,weight_decay_fit,
                  train_x,train_y, batch,init_weights):
    """Generate data splitting subset
    Parameters
    ----------
    split_fit : int
        the number of splits 
        = the number of hidden layer
    device_fit : "cpu" or "cuda"
    net_fit : chosen network structure
    lr_fit : float
        learning rate
    momentum_fit : float
        momentum
    weight_decay_fit : float
        penalty parameter
        
    train_x : DataFrame
    train_y : Series
        global data
    batch : int
        batch size
    init_weights : function
        initialization 
            
    Returns
    -------
    train_loaderset : list (shape: split_fit x split_fit)
        training dataloader set
    netset :  list (shape: split_fit)
        network list
    optimizerset : list (shape: split_fit x split_fit)
        chosen optimizer set
    """
    
    train_loaderset=[] # there are "split" elements
    netset=[]
    optimizerset=[]

    np.random.seed(0)
    row_rand_array = np.arange(train_x.shape[0])
    np.random.shuffle(row_rand_array)
    split=split_fit
    layer=split_fit
    length=int(len(train_x)/layer)

    train_loaderset1=[]
    for l in range(layer): #split into different dataset
        curx=train_x.values[row_rand_array[l*length:(l+1)*length]]
        cury=train_y.values[row_rand_array[l*length:(l+1)*length]]
        nnx = torch.from_numpy(curx).float()
        nny = torch.squeeze(torch.from_numpy(cury).float()) 
        train_loader = DataLoader(mydataset(nnx, nny),batch_size=batch, shuffle=True)
        train_loaderset1.append(train_loader)
        
    for i in range(split):
        train_loaderset1=train_loaderset1[1:]+train_loaderset1[0:1]
        train_loaderset.append(train_loaderset1)
        net = net_fit
        net = net.to(device_fit)
        torch.manual_seed(1)
        net.apply(init_weights)
        netset.append(net)      
    
        optimizer1=[]
        for j in range(split):
            optimizer1.append(optim.SGD([{'params': net.layers[j].parameters()}],
                                        lr=lr_fit,momentum=momentum_fit,weight_decay=weight_decay_fit) )
        optimizerset.append(optimizer1)
        
    return (train_loaderset,
            netset,
            optimizerset)