Mydataset
======
.. automodule:: Multi-Layer-Kernel-Machine.DataSplitting
   :members:


For example:

.. code-block:: python
   from GenerateSplit import GenerateSplit
   from DataSplitting import DataSplitting
   train_loaderset,netset,optimizerset=GenerateSplit(2,device,net,8e-4,0.9,1e-4,train_x,train_y, batch,init_weights)
   splker_trainloss,splker_testloss,splker_prediction = splkermodel.fitting(train_x,train_y,test_x,test_y, batch)
   splkermodel.GradientBand(splker_trainloss,train_x,train_y,test_x,test_y, batch)


