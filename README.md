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

The notebooks below reproduce every numerical result in the paper. Notebooks corresponding to earlier draft experiments are kept under `archive/` folders for reference.

### Data

| Datasets | Instances | Attributes | Source |
| --- | --- | --- | --- |
| YearPredictionMSD | 515,345 | 90 | Download from [YearPredictionMSD](http://archive.ics.uci.edu/dataset/203/yearpredictionmsd) |
| SML2010 | 4,137 | 20 | Download from [SML2010](http://archive.ics.uci.edu/dataset/274/sml2010) |


### Real Data Analysis (Section 5)

Notebooks live under [`Numerical Examples/Real_Data_Examples/`](https://github.com/toz015/Multi-Layer-Kernel-Machine/tree/main/Numerical%20Examples/Real_Data_Examples). Place the downloaded data alongside the notebook and keep the original filenames.

| Paper section | Dataset | Notebook |
| --- | --- | --- |
| §5.1 — Temperature forecasting | SML2010 | [`RealData_SML_final.ipynb`](https://github.com/toz015/Multi-Layer-Kernel-Machine/blob/main/Numerical%20Examples/Real_Data_Examples/RealData_SML_final.ipynb) |
| §5.2 — Audio feature data | MillionSongs (10% subset) | [`RealData_MSD_final.ipynb`](https://github.com/toz015/Multi-Layer-Kernel-Machine/blob/main/Numerical%20Examples/Real_Data_Examples/RealData_MSD_final.ipynb) |


### Simulations (Appendix A.4)

Notebooks live under [`Numerical Examples/Simulation_Examples/Additional Examples/`](https://github.com/toz015/Multi-Layer-Kernel-Machine/tree/main/Numerical%20Examples/Simulation_Examples/Additional%20Examples).

| Paper section | Setting | Notebook |
| --- | --- | --- |
| A.4.1 — Examples A.25, A.26 | 2D Gaussian-mixture target, Gaussian RFF, varying $n$, $\sigma$, $D$ | [`Two-Dimensional_Example_1_final.ipynb`](https://github.com/toz015/Multi-Layer-Kernel-Machine/blob/main/Numerical%20Examples/Simulation_Examples/Additional%20Examples/Two-Dimensional_Example_1_final.ipynb) |
| Example A.27 | Same target, Matérn kernel, $\nu \in \{15, 20, 30\}$ | [`Two-Dimensional_Example_1_matern_final.ipynb`](https://github.com/toz015/Multi-Layer-Kernel-Machine/blob/main/Numerical%20Examples/Simulation_Examples/Additional%20Examples/Two-Dimensional_Example_1_matern_final.ipynb) |


### Package Usage

Experiments can also be reproduced through the `Multi-Layer-Kernel-Machine` package. See [an MSD quick-start example](https://github.com/toz015/Multi-Layer-Kernel-Machine/blob/main/tests/package%20example%20usage(MSD).ipynb).
