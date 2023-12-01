## Fitting
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