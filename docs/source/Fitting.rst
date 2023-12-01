Fitting
======
.. automodule:: Multi-Layer-Kernel-Machine.Fitting
   :members:


For example:

.. code-block:: python
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

