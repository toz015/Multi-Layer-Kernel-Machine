## RandomFeature
`class.RandomFourierFeature(d,D,kernel,gamma,device)`

**Description:** Random Fourier feature.


**Parameters:** 
- d : int, Input space dimension
- D : int, Feature space dimension
- kernel : char, Kernel to use; 'G', 'L', or 'C'
- gamma : float, kernel scale
- device : "cpu" or "cuda"

**Methods:**

- `transform(x)` Transform data to random features.
    - Parameters:
        - x : tensor, shape=(n,d), data to be transformed
    - Returns:
        - x' : tensor, shape=(n,D), random features

**Example:**
```python
from RandomFeature import RandomFourierFeature

rff=RandomFourierFeature(90,100,kernel='G',gamma=0.1,device="cpu")
feature=rff.transform(nntrain_x)
```