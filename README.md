# LQVAE-separation
Code for "Unsupervised Source Separation via Bayesian inference in the latent domain"

[Paper](https://arxiv.org/abs/putarxivhere) 

# Install

Install the conda package manager from https://docs.conda.io/en/latest/miniconda.html

``` 
conda create --name lqvae-separation python=3.7.5
conda activate lqvae-separation
pip install mpi4py==3.0.3
conda install pytorch=1.4 torchvision=0.5 torchaudio=0.5 cudatoolkit=10.0 -c pytorch
pip install -r requirements.txt
pip install -e .
```
# Checkpoints

- Enter inside `script/` folder and create the folder `checkpoints/`. 
- Download the checkpoints contained in this [Google Drive](https://drive.google.com/drive/folders/1LWhzfUMDg0fnSzPOgMNDgfjbEfF8ARO6?usp=sharing) folder and put them inside `checkpoints/`

# Separation

- Call the following in order to perform `bs` separations of 3 seconds starting from second `shift` of the mixture created with the sources in `path_1` and `path_2`. The sources must be sampled at 22kHz.
  ```
  PYTHONPATH=.. python bayesian_inference.py --shift=shift --path_1=path_1 --path_2=path_2 --bs=bs
  ```
- The default value for `bs` is `64`, and can be handled by an RTX3080 with 16 GB of VRAM. Lower the value if you get `CUDA: out of memory`.