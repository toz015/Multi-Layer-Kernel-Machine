Mydataset
======
.. automodule:: Multi-Layer-Kernel-Machine.RandomFeature
   :members:


For example:

.. code-block:: python
   from RandomFeature import RandomFourierFeature
   rff=RandomFourierFeature(90,100,kernel='G',gamma=0.1,device="cpu")
   feature=rff.transform(nntrain_x)


