## Structure
`class.Net(layer_sizes,device)`

`class.ResNet(layer_sizes,device)`

`class.KernelNet(layer_sizes,kernel_types,kernel_scales,device)`

`class.ResKernelNet(layer_sizes,kernel_types,kernel_scales,device)`

**Description:** Custom multi-layer structures for: common neural network, ResNet, multi-layer kernel machine (MTKM), and residual kernel machine (RKM).

**Parameters:** 
- layer_sizes : list (length>=3), Neuron numbers in each layer, including input and output layer
- kernel_types : list (length>=1), Kernel function types in hidden layers
- kernel_scales : list (length>=1), Kernel function scales in hidden layers
- device : "cpu" or "cuda"

**Example:**

```python
from Structure import Net,ResNet,KernelNet,ResKernelNet

net1 = Net([90,32,8,1],device) 
net2 = KernelNet([90,32,8,1],["C","G"],[0.01,0.1],device)
```