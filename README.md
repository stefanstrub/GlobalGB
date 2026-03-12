<<<<<<< HEAD
# GlobalGB

```
conda create -n global_gb python=3.12
conda activate global_gb
=======
# Accelerated global Galactic binary search algorithm

The parameter estimate of Galactic binaries (GBs) is split into two parts. The first part extracts the best fitting GBs in the provided data set based on a differential evolution search algorithm. The second part uses the GBGPU, https://github.com/mikekatz04/GBGPU, to quickly sample across a reduced parameter space around the maximum likelihood estimate from the first steps. That way we are able to obtain a posterior distribution of a found GB within 2 seconds on a laptop grade GPU. 

The first part only requires the pip installable lisa-data-challenge. The second part requires GBGPU with GPU access. The following information about `GBGPU` and its installement is a close version of https://github.com/mikekatz04/GBGPU.

`GBGPU` is a GPU-accelerated version of the `FastGB` waveform which has been developed by Neil Cornish, Tyson Littenberg, Travis Robson, and Stas Babak. It computes gravitational waveforms for Galactic binary systems observable by LISA using a fast/slow-type decomposition. For more details on the original construction of `FastGB` see [arXiv:0704.1808](https://arxiv.org/abs/0704.1808).

The current version of the code is very closely related to the implementation of `FastGB` in the LISA Data Challenges' Python code package. The waveform code is entirely Python-based. It is about 1/2 the speed of the full C version, but much simpler in Python for right now. There are also many additional functions including fast likelihood computations for individual Galactic binaries, as well as fast C-based methods to combine waveforms into global fitting templates. 

The code is CPU/GPU agnostic. CUDA and NVIDIA GPUs are required to run these codes for GPUs.

See the [documentation](https://mikekatz04.github.io/GBGPU/html/index.html) for more details. This code was designed for (# TODO: add new arxiv number). If you use any part of this code, please cite (# TODO: add new arxiv number), its [Zenodo page](https://zenodo.org/record/6500434#.YmpofxNBzlw), [arXiv:0704.1808](https://arxiv.org/abs/0704.1808), and [arXiv:1806.00500](https://arxiv.org/abs/1806.00500). 

### Prerequisites

To install this software for CPU usage, you need [gsl >2.0](https://www.gnu.org/software/gsl/), Python >3.4, and NumPy. We generally recommend installing everything, including gcc and g++ compilers, in the conda environment as is shown in the examples here. This generally helps avoid compilation and linking issues. If you use your own chosen compiler, you may need to add information to the `setup.py` file.

To install this software for use with NVIDIA GPUs (compute capability >2.0), you need the [CUDA toolkit](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html) and [CuPy](https://cupy.chainer.org/). The CUDA toolkit must have cuda version >8.0. Be sure to properly install CuPy within the correct CUDA toolkit version. Make sure the nvcc binary is on `$PATH` or set it as the `CUDAHOME` environment variable.

### Installing


0) [Install Anaconda](https://docs.anaconda.com/anaconda/install/) if you do not have it.

1) Create a virtual environment. **Note**: There is no available `conda` compiler for Windows. If you want to install for Windows, you will probably need to add libraries and include paths to the `setup.py` file.

```
conda create -n ldc-gpu -c conda-forge gcc_linux-64 gxx_linux-64 gsl numpy Cython scipy h5py matplotlib python=3.9
conda activate ldc-gpu
>>>>>>> master
```

```
pip install -r requirements.txt
uv pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ mojito-processor
```



# Install LISAanalysistools
```
# For CPU-only version
pip install lisaanalysistools

# For GPU-enabled versions with CUDA 11.Y.Z
pip install lisaanalysistools-cuda11x

# For GPU-enabled versions with CUDA 12.Y.Z
pip install lisaanalysistools-cuda12x
```



# Install LISAonGPU

1. Change the folder.
```
cd ..
cd lisa-on-gpu
```
2. Run setup.py
```
python setup.py install
```

# Install Eryn
```
pip install eryn
```

# Install LDC

'''
conda install gsl fftw
pip install lisa-data-challenge
'''

# Install BBhx

'''
pip install bbhx-cuda12x
'''

