# LQVAE-separation
Code for "Unsupervised Source Separation via Bayesian inference in the latent domain"

[Paper](https://arxiv.org/abs/2110.05313) 

# Install

Install the conda package manager from https://docs.conda.io/en/latest/miniconda.html

``` 
conda create --name lqvae-separation python=3.7.5
conda activate lqvae-separation
pip install mpi4py==3.0.3
pip install ffmpeg-python==0.2.0
pip install torch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2
pip install -r requirements.txt
pip install -e .
```
# Checkpoints

- Enter inside `script/` folder and create the folder `checkpoints/` and the folder `results/`.  
- Download the checkpoints contained in this [Google Drive](https://drive.google.com/drive/folders/1LWhzfUMDg0fnSzPOgMNDgfjbEfF8ARO6?usp=sharing) folder and put them inside `checkpoints/`

# Separation

- Call the following in order to perform `bs` separations of 3 seconds starting from second `shift` of the mixture created with the sources in `path_1` and `path_2`. The sources must be sampled at 22kHz.
  ```
  PYTHONPATH=.. python bayesian_inference.py --shift=shift --path_1=path_1 --path_2=path_2 --bs=bs
  ```
- The default value for `bs` is `64`, and can be handled by an RTX3080 with 16 GB of VRAM. Lower the value if you get `CUDA: out of memory`.

# Training

- The `vqvae/vqvae.py`file of Jukebox has been modified in order to include the linearization loss of the LQ-VAE (it is computed at all levels of the hierarchical VQ-VAE but
we only care of the top-most level given that we perform separation there). One can train a new LQ-VAE on custom
data by running (the trained model uses the `vqvae` hyperparameters in `hparams.py` so if you want to change the separation level)
