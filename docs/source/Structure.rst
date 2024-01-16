Structure
======

``class.Net(layer_sizes,device)``

``class.ResNet(layer_sizes,device)``

``class.KernelNet(layer_sizes,kernel_types,kernel_scales,device)``

``class.ResKernelNet(layer_sizes,kernel_types,kernel_scales,device)``

**Description:** Custom multi-layer structures for: common neural network (NN), ResNet, multi-layer kernel machine (MLKM), and residual kernel machine (RKM).

**Parameters:** 

- layer_sizes : list (length>=3), neuron numbers in each layer, including input and output layer
- kernel_types : list (length>=1), kernel function types in hidden layers (for MLKM and RKM)
- kernel_scales : list (length>=1), kernel function scales in hidden layers (for MLKM and RKM)
- device : char, device to use, "cpu" or "cuda"

**Example:**

.. code:: python
   :number-lines:
   
   from Multi_Layer_Kernel_Machine.Structure import Net,ResNet,KernelNet,ResKernelNet
   net1 = Net([90,32,8,1],device) 
   net2 = KernelNet([90,32,8,1],["C","G"],[0.01,0.1],device)


