# szlan

Code used to reproduce the experiments from the paper **"Langevin for Nonconvex Optimization: Exact, Inexact and Zeroth-Order."**

A preprint of the paper is available on [ArXiv](https://arxiv.org/abs/2607.22353).

The experiment scripts and their instructions are provided in the [`paper_experiments`](./paper_experiments/) folder.



# :package: Installation

Follow the steps below to set up the environment and install the required dependencies.

We recommend creating a new Python environment using [Conda](https://docs.conda.io/projects/conda/en/23.11.x/user-guide/install/download.html) to avoid dependency conflicts. The code has been implemented using **Python 3.11**, so we recommend creating the Conda environment with this version as follows:

```bash
conda create -n langevin python=3.11
conda activate langevin
```

The code is implemented using [PyTorch](https://pytorch.org/get-started/locally/) version **2.6**, which therefore has to be installed.

You can install PyTorch using Conda:

```bash
conda install pytorch
```

or using pip:

```bash
pip install torch
```

For a correct PyTorch installation with CUDA support, install the PyTorch version appropriate for your CUDA version. Please refer to the [official PyTorch installation instructions](https://pytorch.org/get-started/locally/) for the appropriate command.


# :books: Citation

If you use this code or the results of this work in your research, please cite the paper

```bibtex
@misc{langevin_exact_inexact,
      title={Langevin for Nonconvex Optimization: Exact, Inexact and Zeroth-Order}, 
      author={Emanuele Naldi and Marco Rando and Lorenzo Rosasco and Silvia Villa},
      year={2026},
      eprint={2607.22353},
      archivePrefix={arXiv},
      primaryClass={math.OC},
      url={https://arxiv.org/abs/2607.22353}, 
}
```