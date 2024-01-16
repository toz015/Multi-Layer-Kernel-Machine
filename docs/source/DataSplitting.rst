DataSplitting
======

``DataSplitting(split_fit,modelset_fit,device_fit,train_loaderset_fit, test_loader_fit, epochs_fit, criterion_fit, optimizerset_fit, terminate_fit=100, print_fit=100,printchoice=True)``

**Description:** The procedure for prediction and estimation with data-splitting.

**Parameters:** 

- split_fit : int, the number of splits (= the number of hidden layer)
- modelset_fit : list, shape = (split_fit), network list
- device_fit : char, device to use, "cpu" or "cuda"
- train_loaderset_fit : list, shape = (split_fit x split_fit), training dataloader set
- test_loader_fit : dataloader for testing data
- epochs_fit : int, maximum epoch number
- criterion_fit : chosen criterion
- optimizerset_fit : list, shape = (split_fit x split_fit), chosen optimizer set
- terminate_fit : int, terminate parameter, terminate if the loss has no significant improvement over T consecutive epochs, default=100
- print_fit : int, print parameter for outputting results, print the results after every T epochs, default=100
- printchoice : bool, choice of print or not, default=True

**Methods:**

- ``fitting(train_x,train_y,test_x,test_y)`` The procedure for training and testing.
    - Parameters:
        - train_x, test_x : DataFrame, predictors 
        - train_y, test_y : Series, responses
    - Returns:
        - trainloss : list, trainning loss (on average)
        - testloss : list, testing loss (on average)
        - prediction : list, model prediction for testing data (on average)
        
**Example:**

.. code:: python   
   :number-lines:
   
   from Multi_Layer_Kernel_Machine.GenerateSplit import GenerateSplit
   from Multi_Layer_Kernel_Machine.DataSplitting import DataSplitting
   ## Generate Subsamples
   train_loaderset,netset,optimizerset=GenerateSplit(2,device,net,8e-4,0.9,1e-4,train_x,train_y, batch,init_weights)
   ## Model Fitting
   splker_trainloss,splker_testloss,splker_prediction = splkermodel.fitting(train_x,train_y,test_x,test_y)
   

