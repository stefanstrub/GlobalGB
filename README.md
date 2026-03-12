# GlobalGB

```
conda create -n global_gb python=3.12
conda activate global_gb
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

