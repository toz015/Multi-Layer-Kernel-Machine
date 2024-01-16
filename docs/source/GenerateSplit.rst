GenerateSplit
======

``GenerateSplit(split_fit,device_fit,net_fit,lr_fit,momentum_fit,weight_decay_fit,train_x,train_y, batch,init_weights)``

**Description:** Generation of data splitting subsets.

**Parameters:** 

- split_fit : int, the number of splits (= the number of hidden layer)
- device_fit : char, device to use, "cpu" or "cuda"
- net_fit : chosen network structure
- lr_fit : float, learning rate
- momentum_fit : float, momentum
- weight_decay_fit : float, penalty parameter
- train_x : DataFrame, predictors
- train_y : Series, responses
- batch : int, batch size
- init_weights : function, initialization method

**Returns:**

- train_loaderset : list, shape = (split_fit x split_fit), training dataloader set, there are L orders, each order contains L subsamples
- netset : list, shape = (split_fit), network set, each network is specific to one order
- optimizerset : list, shape = (split_fit x split_fit), chosen optimizer set, each optimizer is specific to one subsamples of one order

**Example:**

.. code:: python
   :number-lines:
   
   from Multi_Layer_Kernel_Machine.Structure import KernelNet
   from Multi_Layer_Kernel_Machine.GenerateSplit import GenerateSplit
   def init_weights(m):
      if type(m) == nn.Conv2d:
         torch.nn.init.normal_(m.weight,mean=0,std=0.5)
      if type(m) == nn.Linear:
         torch.nn.init.uniform_(m.weight,a=0,b=1)
         m.bias.data.fill_(0.01)
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   net = KernelNet([90,32,8,1],["C","G"],[0.01,0.1],device)
   ## Generate Subsamples
   train_loaderset,netset,optimizerset=GenerateSplit(2,device,net,8e-4,0.9,1e-4,train_x,train_y, batch,init_weights)



