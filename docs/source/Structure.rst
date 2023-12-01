Structure
======
.. automodule:: Multi-Layer-Kernel-Machine.Structure
   :members:


For example:

.. code-block:: python
   from Structure import Net,ResNet,KernelNet,ResKernelNet
   net1 = Net([90,32,8,1],device) 
   net2 = KernelNet([90,32,8,1],["C","G"],[0.01,0.1],device)


