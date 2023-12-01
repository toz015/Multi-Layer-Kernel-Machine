Mydataset
======
.. automodule:: Multi-Layer-Kernel-Machine.GenerateSplit
   :members:


For example:

.. code-block:: python
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


