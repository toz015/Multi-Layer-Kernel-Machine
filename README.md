# Multi-Layer Kernel Machine (MLKM)

This is a package for the implementation of the Multi-Layer Kernel Machines (MLKM), which are used for multi-scale nonparametric regression and confidence bands estimation. The method integrates random feature projections with multi-layer structure.

## Installation

`pip install Multi-Layer-Kernel-Machine`

See [the package in PyPi](https://pypi.org/project/Multi-Layer-Kernel-Machine/).


## Dependencies
- numpy, pandas, matplotlib, tqdm, scikit-learn
- pytorch


## License

Multi-Layer Kernel Machine (MLKM) is released under the MIT License. 


## Usage 

### Mydataset

### RandomFeature

### Structure

### Fitting

### GenerateSplit

### DataSplitting



## Reproducing Experiments

### Data 

| Datasets | Instances |  Attributes | Source |
| --- | --- | --- | --- |
| YearPredictionMSD | 515,345 | 90 | http://archive.ics.uci.edu/dataset/203/yearpredictionmsd |
| SML2010 | 4,137 | 20 | http://archive.ics.uci.edu/dataset/274/sml2010 |
| California Housing | 20,640 | 8 | from sklearn.datasets import fetch_california_housing   |
| Dry Bean | 13,611 | 16 | http://archive.ics.uci.edu/dataset/602/dry+bean+dataset |

We carry out the full YearPredictionMSD dataset experiment using the [AutoDL platform](https://www.autodl.com).


### Raw examples

Direct implementation of the raw experiments is available. See experiments in [Simulation & Real Data Analysis](https://github.com/ZZZhyEva/Multi-Layer-Kernel-Machines/tree/main/Simulations%20%26%20Real%20Data%20Analysis) folder. Run them directly.

### Package examples

Install our package and follow the module instruction. A complete example is in [package example usage(MSD).ipynb](https://github.com/ZZZhyEva/Multi-Layer-Kernel-Machines/blob/main/package%20example%20usage(MSD).ipynb). You can run it with either cpu or gpu.


