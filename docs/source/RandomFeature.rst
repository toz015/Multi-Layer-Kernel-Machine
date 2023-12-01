RandomFeature
======

.. autoclass:: Multi-Layer-Kernel-Machine.RandomFeature.RandomFourierFeature

For example:

.. code:: python
   from RandomFeature import RandomFourierFeature
   rff=RandomFourierFeature(90,100,kernel='G',gamma=0.1,device="cpu")
   feature=rff.transform(nntrain_x)


