# Multi-Layer Kernel Machine (MLKM)

[![Documentation Status](https://readthedocs.org/projects/multi-layer-kernel-machine/badge/?version=latest)](https://multi-layer-kernel-machine.readthedocs.io/en/latest/?badge=latest)
[![PyPI](https://img.shields.io/pypi/v/Multi-Layer-Kernel-Machine.svg?style=plastic&PyPI)](https://pypi.org/project/Multi-Layer-Kernel-Machine/)


This is a package for the implementation of the Multi-Layer Kernel Machines (MLKM), which are used for multi-scale nonparametric regression and confidence bands estimation. The method integrates random feature projections with a multi-layer structure.

## Installation

`pip install Multi-Layer-Kernel-Machine`

See [the package in PyPi](https://pypi.org/project/Multi-Layer-Kernel-Machine/).


## Dependencies
- numpy, pandas, matplotlib, tqdm, scikit-learn
- pytorch


## Usage 

See [the documentation](https://multi-layer-kernel-machine.readthedocs.io/en/latest/).


## License

Multi-Layer Kernel Machine (MLKM) is released under the MIT License. 


## Reproducing Experiments

### Data 

| Datasets | Instances |  Attributes | Source |
| --- | --- | --- | --- |
| YearPredictionMSD | 515,345 | 90 | Download through [YearPredictionMSD](http://archive.ics.uci.edu/dataset/203/yearpredictionmsd) |
| SML2010 | 4,137 | 20 | Download through [SML2010](http://archive.ics.uci.edu/dataset/274/sml2010) |
| CaliforniaHousing | 20,640 | 8 | `from sklearn.datasets import fetch_california_housing`   |
| DryBean | 13,611 | 16 | Download through [Dry Bean Dataset](http://archive.ics.uci.edu/dataset/602/dry+bean+dataset) |


### Simulations

Run [codes for simulations](https://github.com/ZZZhyEva/Multi-Layer-Kernel-Machine/tree/main/Numerical%20Results/Simulation) directly.

`Gaussian Mixture.ipynb` and `Gaussian Mixture Plus.ipynb` are for two-dimensional Gaussian mixture models.

`Additive (function) (distribution).ipynb` are for additive models with `function` $\in$ {`Trigonometric`, `sin-ration`, `mix`} and `distribution` $\in$ {`1` (uniform), `2` (multivariate normal)}.


### Real Data Analysis

Run [codes for real data analysis](https://github.com/ZZZhyEva/Multi-Layer-Kernel-Machine/tree/main/Numerical%20Results/Real_Data_Analysis) after downloading and unzipping the data. Store the data and codes in the same path and don't change the filenames of them.

`RealData_(dataset).ipynb` are for CPU/GPU training with `dataset` $\in$ {`SML`, `MSD`, `House`, `DryBean`}. We sample a subset for `MSD`.

`Autodl_MSD.ipynb` is for CPU/GPU training for large-scale `MSD`. We recommend specifying `cuda:0/1` for GPU training.


### Package Usage
Our experiments can also be carried out using our Multi-Layer-Kernel-Machine package. We provide [an example for dataset MSD](https://github.com/ZZZhyEva/Multi-Layer-Kernel-Machine/blob/main/tests/package%20example%20usage(MSD).ipynb) as a quick start.
