# Multi-Layer Kernel Machine (MLKM)

This is a package for the implementation of the Multi-Layer Kernel Machines (MLKM), which are used for multi-scale nonparametric regression and confidence bands estimation. The method integrates random feature projections with multi-layer structure.

## Installation

`pip install Multi-Layer-Kernel-Machine`


## License

Multi-Layer Kernel Machine (MLKM) is released under the MIT License. 


## Usage 

### Mydataset

`class.mydataset(x, y)`

**Description:** Prepare the dataset for model fitting. This module is often used in pytorch programs that specify the data set objects to be loaded.

**Example:**
```python
from Mydataset import dataset

nntrain_x = torch.from_numpy(train_x.to_numpy()).float()
nntrain_y = torch.squeeze(torch.from_numpy(train_y.to_numpy()).float())

train_loader = DataLoader(mydataset(nntrain_x, nntrain_y),batch_size=batch, shuffle=True)
```

### RandomFeature
`class.RandomFourierFeature(d,D,kernel,gamma,device)`

**Description:** Random Fourier feature.


**Parameters:** 
- d : int, Input space dimension
- D : int, Feature space dimension
- kernel : char, Kernel to use; 'G', 'L', or 'C'
- gamma : float, kernel scale
- device : "cpu" or "cuda"

**Methods:**

- `transform(x)` Transform data to random features.
    - Parameters:
        - x : tensor, shape=(n,d), data to be transformed
    - Returns:
        - x' : tensor, shape=(n,D), random features

**Example:**
```python
from RandomFeature import RandomFourierFeature

rff=RandomFourierFeature(90,100,kernel='G',gamma=0.1,device="cpu")
feature=rff.transform(nntrain_x)
```

### Structure
`class.Net(layer_sizes,device)`

`class.ResNet(layer_sizes,device)`

`class.KernelNet(layer_sizes,kernel_types,kernel_scales,device)`

`class.ResKernelNet(layer_sizes,kernel_types,kernel_scales,device)`

**Description:** Custom multi-layer structures for: common neural network, ResNet, multi-layer kernel machine (MTKM), and residual kernel machine (RKM).

**Parameters:** 
- layer_sizes : list (length>=3), Neuron numbers in each layer, including input and output layer
- kernel_types : list (length>=1), Kernel function types in hidden layers
- kernel_scales : list (length>=1), Kernel function scales in hidden layers
- device : "cpu" or "cuda"

**Example:**

```python
from Structure import Net,ResNet,KernelNet,ResKernelNet

net1 = Net([90,32,8,1],device) 
net2 = KernelNet([90,32,8,1],["C","G"],[0.01,0.1],device)
```

### Fitting
`MultilayerFitting(model_fit,device_fit,train_loader_fit, test_loader_fit,epochs_fit, criterion_fit, optimizer_fit, terminate_fit=10, print_fit=10,printchoice=True)`

**Description:** The procedure for prediction and estimation.

**Parameters:** 
- model_fit : chosen network structure
- device_fit : "cpu" or "cuda"
- train_loader_fit : dataloader for training data
- test_loader_fit : dataloader for test data
- epochs_fit : int, maximum epoch number
- criterion_fit : chosen criterion
- optimizer_fit : chosen optimizer
- terminate_fit : int, terminate parameter
- print_fit : print parameter
- printchoice : bool, choice of print or not

**Methods:**
- `fitting(train_x,train_y,test_x,test_y,batch)` The procedure for training and testing.
    - Parameters:
        - train_x, train_y, test_x, test_y : DataFrame & Series, data to fit
        - batch : int, batch size
    - Returns:
        - trainloss : list, trainning loss
        - testloss : list, test loss
        - prediction : list, model prediction for test data
- `Bootstrap(time_boot,bootbase,train_x,train_y,test_x,test_y, batch,init_weights)` Bootstrap confidence interval.
    - Parameters:
        - time_boot : int, Boostrap times
        - bootbase : list, basic prediction
        - train_x, train_y, test_x, test_y : DataFrame & Series, data to fit
        - batch : int, batch size
        -init_weights : function, initialization 
    - Returns:
        - length : float, confidence interval length
        - coverage : %, confidence interval 95% coverage
- `GradientBand(loss,train_x,train_y,test_x,test_y, batch,pen=0)` Gradient-based confidence interval.
    - Parameters:
        - loss : list, training loss (used for RSS)
        - train_x, train_y, test_x, test_y : DataFrame & Series, data to fit
        - batch : int, batch size
        - pen : float, penalty parameter
    - Returns:
        - length : float, confidence interval length
        - coverage : %, confidence interval 95% coverage
- `HomoConformalBand(train_x,train_y,test_x,test_y, batch)` Conformal confidence interval
    - Parameters:
        - train_x, train_y, test_x, test_y : DataFrame & Series, data to fit
        - batch : int, batch size
    - Returns:
        - length : float, confidence interval length
        - coverage : %, confidence interval 95% coverage
    - References: 
        [1] Lei, Jing, et al. "Distribution-free predictive inference for regression." Journal of the American Statistical Association 113.523 (2018): 1094-1111.
- `HeteConformalBand(loss,train_x,train_y,test_x,test_y, batch)` PLUS Conformal confidence interval.
    - Parameters:
        - loss : list, training loss (used for RSS)
        - train_x, train_y, test_x, test_y : DataFrame & Series, data to fit
        - batch : int, batch size
    - Returns:
        - length : float, confidence interval length
        - coverage : %, confidence interval 95% coverage

**Example:**
```python
from Structure import KernelNet
from Fitting import MultilayerFitting
def init_weights(m):
    if type(m) == nn.Conv2d:
        torch.nn.init.normal_(m.weight,mean=0,std=0.5)
    if type(m) == nn.Linear:
        torch.nn.init.uniform_(m.weight,a=0,b=1)
        m.bias.data.fill_(0.01)
        
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = KernelNet([90,32,8,1],["C","G"],[0.01,0.1],device)
torch.manual_seed(1)
net.apply(init_weights)
criterion=nn.MSELoss() 
optimizer=optim.SGD(net.parameters(),lr=8e-4,momentum=0.9,weight_decay=1e-4) 

mlmodel=MultilayerFitting(net,device,train_loader, test_loader, 2000, criterion, optimizer,100,100)
kernelnn_trainloss,kernelnn_testloss,kernelnn_bootbase=mlmodel.fitting(train_x,train_y,test_x,test_y, batch)

mlmodel.Bootstrap(400,kernelnn_bootbase,train_x,train_y,test_x,test_y, batch,init_weights)
mlmodel.GradientBand(kernelnn_trainloss,train_x,train_y,test_x,test_y, batch)
mlmodel.HomoConformalBand(train_x,train_y,test_x,test_y, batch)
mlmodel.HeteConformalBand(kernelnn_trainloss,train_x,train_y,test_x,test_y, batch)
```

### GenerateSplit
`GenerateSplit(split_fit,device_fit,net_fit,lr_fit,momentum_fit,weight_decay_fit,train_x,train_y, batch,init_weights)`

**Description:** Generation of data splitting subsets.

**Parameters:** 
- split_fit : int, the number of splits (= the number of hidden layer)
- device_fit : "cpu" or "cuda"
- net_fit : chosen network structure
- lr_fit : float, learning rate
- momentum_fit : float, momentum
- weight_decay_fit : float, penalty parameter
- train_x,train_y  : DataFrame & Series, global data
- batch : int, batch size
- init_weights : function, initialization 

**Returns:**
- train_loaderset : list (shape: split_fit x split_fit), training dataloader set
- netset : list (shape: split_fit), network list
- optimizerset : list (shape: split_fit x split_fit), chosen optimizer set
**Example:**
```python
from Structure import KernelNet
from GenerateSplit import GenerateSplit
def init_weights(m):
    if type(m) == nn.Conv2d:
        torch.nn.init.normal_(m.weight,mean=0,std=0.5)
    if type(m) == nn.Linear:
        torch.nn.init.uniform_(m.weight,a=0,b=1)
        m.bias.data.fill_(0.01)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

net = KernelNet([90,32,8,1],["C","G"],[0.01,0.1],device)

train_loaderset,netset,optimizerset=GenerateSplit(2,device,net,8e-4,0.9,1e-4,train_x,train_y, batch,init_weights)
```

### DataSplitting
`DataSplitting(split_fit,modelset_fit,device_fit,train_loaderset_fit, test_loader_fit, epochs_fit, criterion_fit, optimizerset_fit, terminate_fit=10, print_fit=10,printchoice=True`

**Description:** The procedure for prediction and estimation with data-splitting.

**Parameters:** 
- split_fit : int, the number of splits (= the number of hidden layer)
- modelset_fit : list (shape: split_fit), machine list
- device_fit : "cpu" or "cuda"
- train_loaderset_fit : list (shape: split_fit x split_fit), training dataloader set
- test_loader_fit : dataloader for test data
- epochs_fit : int, maximum epoch number
- criterion_fit : chosen criterion
- optimizerset_fit : list (shape: split_fit x split_fit), chosen optimizer set
- terminate_fit : int, terminate parameter
- print_fit : print parameter
- printchoice : bool, choice of print or not

**Methods:**
- `fitting(train_x,train_y,test_x,test_y,batch)` The procedure for training and testing.
    - Parameters:
        - train_x, train_y, test_x, test_y : DataFrame & Series, data to fit
        - batch : int, batch size
    - Returns:
        - trainloss : list, trainning loss (on average)
        - testloss : list, test loss (on average)
        - prediction : list, model prediction for test data (on average)
- `GradientBand(loss,train_x,train_y,test_x,test_y, batch,pen=0)` Gradient-based confidence interval.
    - Parameters:
        - loss : list, training loss (used for RSS)
        - train_x, train_y, test_x, test_y : DataFrame & Series, data to fit
        - batch : int, batch size
        - pen : float, penalty parameter
    - Returns:
        - length : float, confidence interval length
        - coverage : %, confidence interval 95% coverage
**Example:**
```python
#(continue the above)
from DataSplitting import DataSplitting
train_loaderset,netset,optimizerset=GenerateSplit(2,device,net,8e-4,0.9,1e-4,train_x,train_y, batch,init_weights)

splker_trainloss,splker_testloss,splker_prediction = splkermodel.fitting(train_x,train_y,test_x,test_y, batch)

splkermodel.GradientBand(splker_trainloss,train_x,train_y,test_x,test_y, batch)
```