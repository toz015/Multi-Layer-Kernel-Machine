RandomFeature
======

``class.RandomFourierFeature(d,D,kernel,gamma,device)``

**Description:** Random Fourier feature.


**Parameters:** 

- d : int, Input space dimension
- D : int, Feature space dimension
- W : tensor, shape=(D,d), random feature parameter for cos(2Wx+b), default=None
- b : tensor, shape=(D), random feature parameter for cos(2Wx+b), default=None
- kernel : char, kernel to use; 'G', 'L', or 'C', default='G'
- gamma : float, kernel scale, default=1
- device : char, device to use, "cpu" or "cuda", default="cpu"

**Methods:**

- ``transform(x)`` Transform original data to random features.
    - Parameters:
        - x : tensor, shape=(n,d), data to be transformed
    - Returns:
        - result : tensor, shape=(n,D), random features

**Example:**

.. code:: python
   :number-lines:
   
   from Multi_Layer_Kernel_Machine.RandomFeature import RandomFourierFeature
   rff=RandomFourierFeature(90,100,kernel='G',gamma=0.1,device="cpu")
   feature=rff.transform(nntrain_x)


