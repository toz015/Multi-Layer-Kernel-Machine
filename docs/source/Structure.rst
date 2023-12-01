Structure
======

.. autoclass:: Multi-Layer-Kernel-Machine.Structure.Net

.. autoclass:: Multi-Layer-Kernel-Machine.Structure.ResNet

.. autoclass:: Multi-Layer-Kernel-Machine.Structure.KernelNet

.. autoclass:: Multi-Layer-Kernel-Machine.Structure.ResKernelNet


For example:

.. code:: python
   from Structure import Net,ResNet,KernelNet,ResKernelNet
   net1 = Net([90,32,8,1],device) 
   net2 = KernelNet([90,32,8,1],["C","G"],[0.01,0.1],device)


