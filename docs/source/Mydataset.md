## Mydataset

`class.mydataset(x, y)`

**Description:** Prepare the dataset for model fitting. This module is often used in pytorch programs that specify the data set objects to be loaded.

**Parameters:** 
- x : Tensor, predictors
- y : Tensor, responses


**Example:**
```python
from Mydataset import dataset

nntrain_x = torch.from_numpy(train_x.to_numpy()).float()
nntrain_y = torch.squeeze(torch.from_numpy(train_y.to_numpy()).float())

train_loader = DataLoader(mydataset(nntrain_x, nntrain_y),batch_size=batch, shuffle=True)
```