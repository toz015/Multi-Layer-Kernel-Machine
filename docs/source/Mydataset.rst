Mydataset
======

.. autoclass:: Multi-Layer-Kernel-Machine.Mydataset.mydataset
   :members:
For example:
>>> from Mydataset import dataset
>>> nntrain_x = torch.from_numpy(train_x.to_numpy()).float()
>>> nntrain_y = torch.squeeze(torch.from_numpy(train_y.to_numpy()).float())
>>> train_loader = DataLoader(mydataset(nntrain_x, nntrain_y),batch_size=batch, shuffle=True)


