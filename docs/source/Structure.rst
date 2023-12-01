Structure
======

.. autoclass:: Multi-Layer-Kernel-Machine.Structure.Net
   :members:
.. autoclass:: Multi-Layer-Kernel-Machine.Structure.ResNet
   :members:
.. autoclass:: Multi-Layer-Kernel-Machine.Structure.KernelNet
   :members:
.. autoclass:: Multi-Layer-Kernel-Machine.Structure.ResKernelNet
   :members:
For example:
>>> from Structure import Net,ResNet,KernelNet,ResKernelNet
>>> net1 = Net([90,32,8,1],device) 
>>> net2 = KernelNet([90,32,8,1],["C","G"],[0.01,0.1],device)


