import torch.nn as nn
import torch.nn.functional as F
from .RandomFeature import RandomFourierFeature

##### DNN
class Net(nn.Module):
    """Neural Network
    Parameters
    ----------
    layer_sizes : list (length>=3)
        Neuron numbers in each layer, 
        including input and output layer
    device : "cpu" or "cuda"
    """
    
    def __init__(self, layer_sizes,device): 
        super(Net, self).__init__()
        if len(layer_sizes) <3:
            raise Exception('Invalid Input')
        self.layers = nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            if i < len(layer_sizes) - 2:
                self.layers.append(nn.ReLU())  # activation 
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


##### ResNet
class ResidualBlock1(nn.Module):
    def __init__(self,infeatures,outfeatures):
        super(ResidualBlock1,self).__init__()
        self.infeatures = infeatures
        self.outfeatures = outfeatures
        self.fc1 = nn.Linear(infeatures,outfeatures)
        self.fc2 = nn.Linear(infeatures,outfeatures)
    
    def forward(self, x):
        y = self.fc1(x)
        y= F.relu(y)
        x = self.fc2(x)
        return F.relu(x+y)


class ResNet(nn.Module): 
    """ResNet
    Parameters
    ----------
    layer_sizes : list (length>=3)
        Neuron numbers in each layer, 
        including input and output layer
    device : "cpu" or "cuda"
    """
    
    def __init__(self, layer_sizes,device):
        super(ResNet, self).__init__()
        if len(layer_sizes) <3:
            raise Exception('Invalid Input')
        self.layers = nn.ModuleList()
        for i in range(len(layer_sizes) - 2):
            self.layers.append(ResidualBlock1(layer_sizes[i], layer_sizes[i+1]))
        self.layers.append(nn.Linear(layer_sizes[-2], layer_sizes[-1]))
 
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    

##### MTK
class KernelNet(nn.Module):
    """Multi-layer Kernel
    Parameters
    ----------
    layer_sizes : list (length>=3)
        Neuron numbers in each layer, 
        including input and output layer
    kernel_types : list (length>=1)
        Kernel function types in hidden layers
    kernel_scales : list (length>=1)
        Kernel function scales in hidden layers
    device : "cpu" or "cuda"
    """ 
    
    def __init__(self, layer_sizes,kernel_types,kernel_scales,device):
        super(KernelNet, self).__init__()
        if (len(layer_sizes) <3) or (len(kernel_types)!=len(kernel_scales)):
            raise Exception('Invalid Input')
        self.layers = nn.ModuleList()
        self.rffs = []
        for i in range(len(layer_sizes) - 2):
            if i==0:
                self.rffs.append(RandomFourierFeature(layer_sizes[i], layer_sizes[i+1],kernel=kernel_types[i],gamma=kernel_scales[i],device=device))
            else:
                self.rffs.append(RandomFourierFeature(layer_sizes[i+1], layer_sizes[i+1],kernel=kernel_types[i],gamma=kernel_scales[i],device=device))
        for i in range(1,len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
        
    def forward(self, x):
        for layer,rff in zip(self.layers,self.rffs):
            x = rff.transform(x)
            x = layer(x)
        return x
    

##### RK
class ResidualBlock2(nn.Module):
    def __init__(self,infeatures,outfeatures,rff):
        super(ResidualBlock2,self).__init__()
        self.infeatures = infeatures
        self.outfeatures = outfeatures
        self.rff=rff
        
        self.fc1 = nn.Linear(infeatures,outfeatures)
        self.fc2 = nn.Linear(outfeatures,outfeatures)
    
    def forward(self, x):
        rff=self.rff
        x = self.fc1(x)
        y = rff.transform(x)
        y = self.fc2(y)
        return x+y

class ResKernelNet(nn.Module): 
    """Multi-layer Kernel
    Parameters
    ----------
    layer_sizes : list (length>=3)
        Neuron numbers in each layer, 
        including input and output layer
    kernel_types : list (length>=1)
        Kernel function types in hidden layers
    kernel_scales : list (length>=1)
        Kernel function scales in hidden layers
    device : "cpu" or "cuda"
    """ 
    
    def __init__(self, layer_sizes,kernel_types,kernel_scales,device):
        super(ResKernelNet, self).__init__()
        if (len(layer_sizes) <3) or (len(kernel_types)!=len(kernel_scales)):
            raise Exception('Invalid Input')
        self.layers = nn.ModuleList()
        self.rffs = []
        for i in range(len(layer_sizes) - 2):
            if i==0:
                self.rffs.append(RandomFourierFeature(layer_sizes[i], layer_sizes[i+1],kernel=kernel_types[i],gamma=kernel_scales[i],device=device))
            else:
                self.rffs.append(RandomFourierFeature(layer_sizes[i+1], layer_sizes[i+1],kernel=kernel_types[i],gamma=kernel_scales[i],device=device))
        for i in range(1,len(layer_sizes) - 2):
            self.layers.append(ResidualBlock2(layer_sizes[i], layer_sizes[i+1],self.rffs[i]))
        self.layers.append(nn.Linear(layer_sizes[-2],layer_sizes[-1]))
 
    def forward(self, x):
        x = self.rffs[0].transform(x)
        for layer in self.layers:
            x = layer(x)
        return x