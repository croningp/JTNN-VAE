# Junction Tree Variational Autoencoder for Molecular Graph Generation

This is the implementation of [junction tree variational autoencoders][JTNN paper] by Jin *et al.* (original implementation [here][ICML18 repo]) ported to Python 3 and slightly improved.

# Requirements
The code in this repository has only been tested on Linux. We recommend using the cross-platform [miniforge] Python distribution/package manager to install the dependencies (here in a conda environment called `jtnn_env`)

```sh
conda create --name jtnn_env --file conda_list.txt
```

# Quick Start
The following directories contains the most up-to-date implementations:
* `fast_jtnn/` contains codes for model implementation.
* `fast_molvae/` contains codes for VAE training. Please refer to `fast_molvae/README.md` for details.

The following directories provides scripts for the experiments in our original ICML paper:
* `molvae/` includes scripts for training our VAE model only. Please read `molvae/README.md` for training our VAE model.
* `jtnn/` contains codes for model formulation.


# Contact
Repository authors: Hessam Mehr (Hessam.Mehr@glasgow.ac.uk), Dario Caramelli (Dario.Caramelli@glasgow.ac.uk)
Original author: Wengong Jin (wengong@csail.mit.edu)

[ICML18 repo]: https://github.com/wengong-jin/icml18-jtnn
[JTNN paper]: https://arxiv.org/abs/1802.04364
[miniforge]: https://github.com/conda-forge/miniforge