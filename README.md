# Multi-Layer Kernel Machine (MLKM)

[![Documentation Status](https://readthedocs.org/projects/multi-layer-kernel-machine/badge/?version=latest)](https://multi-layer-kernel-machine.readthedocs.io/en/latest/?badge=latest)
[![PyPI](https://img.shields.io/pypi/v/Multi-Layer-Kernel-Machine.svg?style=plastic&PyPI)](https://pypi.org/project/Multi-Layer-Kernel-Machine/)


This is a package for the implementation of the Multi-Layer Kernel Machine (MLKM), which is a framework for multi-scale nonparametric regression and confidence bands. The method integrates random feature projections with a multi-layer structure.

## Installation

`pip install Multi-Layer-Kernel-Machine`

See [MLKM package in PyPi](https://pypi.org/project/Multi-Layer-Kernel-Machine/).


## Dependencies
- Python 3
- Pytorch
- numpy, pandas, matplotlib, tqdm, scikit-learn


## Usage 

See [MLKM documentation](https://multi-layer-kernel-machine.readthedocs.io/en/latest/).


## License

Multi-Layer Kernel Machine (MLKM) is released under the MIT License. 


## Reproducing Experiments

### Data 

| Datasets | Instances |  Attributes | Source |
| --- | --- | --- | --- |
| YearPredictionMSD | 515,345 | 90 | Download through [YearPredictionMSD](http://archive.ics.uci.edu/dataset/203/yearpredictionmsd) |
| SML2010 | 4,137 | 20 | Download through [SML2010](http://archive.ics.uci.edu/dataset/274/sml2010) |
| DryBean | 13,611 | 16 | Download through [Dry Bean Dataset](http://archive.ics.uci.edu/dataset/602/dry+bean+dataset) |


### Simulations

Run [Simulation Examples](https://github.com/ZZZhyEva/Multi-Layer-Kernel-Machine/tree/main/Numerical%20Examples/Simulation_Examples) directly.

To obtain the results in Example 1, run `Sparse Additive - d=x.ipynb` for `x` $\in\{4,8,16,32,64,128\}$.

To obtain the results in Example 2, run `ATLAS model - d=x.ipynb` for `x` $\in\{4,8,16,32,64,128\}$.

To obtain the results in Example 3, run `Additive (function) (distribution).ipynb` for additive models with `function` $\in$ {Trigonometric, sin-ration, mix} and `distribution` $\in$ {1 (uniform), 2 (multivariate normal)}.

To obtain the results in Example 4, run `Example4.ipynb`.

To obtain the results in Appendix, run codes in `Additional Examples`.


### Real Data Analysis

Run [Real Data Examples](https://github.com/ZZZhyEva/Multi-Layer-Kernel-Machine/tree/main/Numerical%20Examples/Real_Data_Examples) after downloading and unzipping the data. Store the data and codes in the same path and don't change the filenames of them.

To obtain the results of temperature forecasting data, run `RealData_SML.ipynb`. 

To obtain the results of audio feature data, run `RealData_MSD.ipynb` for small-scale CPU training and run `Autodl_MSD.ipynb` for large-scale GPU training. Please  specify `cuda:0/1` for GPU training.

To obtain the results in Appendix, run `RealData_DryBean.ipynb`.


### Package Usage
Our experiments can also be carried out using our Multi-Layer-Kernel-Machine package. We provide [an example for dataset MSD](https://github.com/ZZZhyEva/Multi-Layer-Kernel-Machine/blob/main/tests/package%20example%20usage(MSD).ipynb) as a quick start.
