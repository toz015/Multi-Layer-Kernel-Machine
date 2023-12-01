DataSplitting
======

.. autoclass:: Multi-Layer-Kernel-Machine.DataSplitting.DataSplitting

``DataSplitting(split_fit,modelset_fit,device_fit,train_loaderset_fit, test_loader_fit, epochs_fit, criterion_fit, optimizerset_fit, terminate_fit=10, print_fit=10,printchoice=True``

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

- ``fitting(train_x,train_y,test_x,test_y,batch)`` The procedure for training and testing.
    - Parameters:
        - train_x, train_y, test_x, test_y : DataFrame & Series, data to fit
        - batch : int, batch size
    - Returns:
        - trainloss : list, trainning loss (on average)
        - testloss : list, test loss (on average)
        - prediction : list, model prediction for test data (on average)
- ``GradientBand(loss,train_x,train_y,test_x,test_y, batch,pen=0)`` Gradient-based confidence interval.
    - Parameters:
        - loss : list, training loss (used for RSS)
        - train_x, train_y, test_x, test_y : DataFrame & Series, data to fit
        - batch : int, batch size
        - pen : float, penalty parameter
    - Returns:
        - length : float, confidence interval length
        - coverage : %, confidence interval 95% coverage
        
**Example:**

.. code:: python   
   :number-lines:
   
   from GenerateSplit import GenerateSplit
   from DataSplitting import DataSplitting
   train_loaderset,netset,optimizerset=GenerateSplit(2,device,net,8e-4,0.9,1e-4,train_x,train_y, batch,init_weights)
   splker_trainloss,splker_testloss,splker_prediction = splkermodel.fitting(train_x,train_y,test_x,test_y, batch)
   splkermodel.GradientBand(splker_trainloss,train_x,train_y,test_x,test_y, batch)
   

