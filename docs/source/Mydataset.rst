Mydataset
======

``class.mydataset(x, y)``

**Description:** Prepare the dataset for model fitting. 
This module is often used in pytorch programs that specify the dataset objects to be loaded.

**Parameters:** 

- x : Tensor, predictors
- y : Tensor, responses

**Example:**

.. code:: python
   :number-lines:
   
   from Multi_Layer_Kernel_Machine.Mydataset import dataset
   nntrain_x = torch.from_numpy(train_x.to_numpy()).float()
   nntrain_y = torch.squeeze(torch.from_numpy(train_y.to_numpy()).float())
   train_loader = DataLoader(mydataset(nntrain_x, nntrain_y),batch_size=batch, shuffle=True)


