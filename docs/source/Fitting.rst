Fitting
======

``MultilayerFitting(model_fit,device_fit,train_loader_fit, test_loader_fit,epochs_fit, criterion_fit, optimizer_fit, terminate_fit=100, print_fit=100,printchoice=True)``

**Description:** The procedure for estimation and inference.

**Parameters:** 

- model_fit : chosen network structure (see module "Structure")
- device_fit : char, device to use, "cpu" or "cuda"
- train_loader_fit : dataloader, training data
- test_loader_fit : dataloader, testing data
- epochs_fit : int, maximum epoch number
- criterion_fit : chosen criterion
- optimizer_fit : chosen optimizer
- terminate_fit : int, terminate parameter, terminate if the loss has no significant improvement over T consecutive epochs, default=100
- print_fit : int, print parameter for outputting results, print the results after every T epochs, default=100
- printchoice : bool, choice of print or not, default=True

**Methods:**

- ``fitting(train_x,train_y,test_x,test_y)`` The procedure for model training and testing.
    - Parameters:
        - train_x, test_x : DataFrame, predictors 
        - train_y, test_y : Series, responses
    - Returns:
        - trainloss : list, trainning loss
        - testloss : list, testing loss
        - prediction : list, model prediction for testing data
- ``Bootstrap(time_boot,bootbase,train_x,train_y,test_x,test_y, batch,init_weights)`` 95% Bootstrap confidence band.
    - Parameters:
        - time_boot : int, Boostrap times
        - bootbase : list, the original prediction for testing data (see returns of module "fitting")
        - train_x, test_x : DataFrame, predictors 
        - train_y, test_y : Series, responses
        - batch : int, batch size
        - init_weights : function, initialization method
    - Returns:
        - length : float, interval length
        - coverage : %, 95% coverage probability
- ``HomoConformalBand(train_x,train_y,test_x,test_y,calibration_x,calibration_y)`` 95% Conformal confidence band
    - Parameters:
        - train_x, test_x, calibration_x : DataFrame, predictors 
        - train_y, test_y, calibration_y : Series, responses
    - Returns:
        - length : float, interval length
        - coverage : %, 95% coverage probability
    - References: 
        [1] Lei, Jing, et al. "Distribution-free predictive inference for regression." Journal of the American Statistical Association 113.523 (2018): 1094-1111.
- ``HeteConformalBand(loss,train_x,train_y,test_x,test_y,calibration_x,calibration_y)`` Our proposed 95% MLKM conformal confidence band.
    - Parameters:
        - loss : list, training loss (used for RSS)
        - train_x, test_x, calibration_x : DataFrame, predictors 
        - train_y, test_y, calibration_y : Series, responses
    - Returns:
        - length : float, interval length
        - coverage : %, 95% coverage probability

**Example:**

.. code:: python
   :number-lines:

   from Multi_Layer_Kernel_Machine.Structure import KernelNet
   from Multi_Layer_Kernel_Machine.Fitting import MultilayerFitting
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
   optimizer=optim.SGD(net.parameters(),lr=1e-3,momentum=0.9,weight_decay=1e-4)
   ## Model Fitting
   mlmodel=MultilayerFitting(net,device,train_loader, test_loader, 2000, criterion, optimizer,100,100,printchoice=False)
   kernelnn_trainloss,kernelnn_testloss,kernelnn_bootbase=mlmodel.fitting(train_x,train_y,test_x,test_y)
   ## Confidence Bands
   mlmodel.HomoConformalBand(train_x,train_y,test_x,test_y, calibration_x,calibration_y)
   mlmodel.HeteConformalBand(kernelnn_trainloss,train_x,train_y,test_x,test_y, calibration_x,calibration_y)

