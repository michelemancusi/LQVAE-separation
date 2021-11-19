# LQVAE-separation
Code for "Unsupervised Source Separation via Bayesian inference in the latent domain"

[Paper](https://arxiv.org/abs/2110.05313)

# Samples

|     |  GT Compressed  | Separated |
| ----------- | ----------- | ----------- |
| Drums      | [GT Compressed Drums ](/samples/real_drums.wav)       | [Separated Drums](/samples/rec_drums.wav)       |
| Bass   | [GT Compressed Bass](/samples/real_bass.wav)        | [Separated Bass](/samples/rec_bass.wav)        |
| Mix   | [GT Compressed Mix](/samples/real_mix.wav)        | [Separated Mix](/samples/rec_mix.wav)        |

The separation is performed on a x64 compressed latent domain. The results can be upsampled via Jukebox upsamplers in
order to increment perceptive quality (WIP). 


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

# Separation with checkpoints

- Call the following in order to perform `bs` separations of 3 seconds starting from second `shift` of the mixture created with the sources in `path_1` and `path_2`. The sources must be WAV files sampled at 22kHz.
  ```
  PYTHONPATH=.. python bayesian_inference.py --shift=shift --path_1=path_1 --path_2=path_2 --bs=bs
  ```
- The default value for `bs` is `64`, and can be handled by an RTX3080 with 16 GB of VRAM. Lower the value if you get `CUDA: out of memory`.

# Training

## LQ-VAE
- The `vqvae/vqvae.py`file of Jukebox has been modified in order to include the linearization loss of the LQ-VAE (it is computed at all levels of the hierarchical VQ-VAE but
we only care of the topmost level given that we perform separation there). One can train a new LQ-VAE on custom
data (here `data/train` for train and `data/test` for test) by running the following from the root of the project 
```
PYTHONPATH=. mpiexec -n 1 python jukebox/train.py --hps=vqvae --sample_length=131072 --bs=8 
--audio_files_dir=data/train/ --labels=False --train --test --aug_shift --aug_blend --name=lq_vae --test_audio_files_dir=data/test
```
- The trained model uses the `vqvae` hyperparameters in `hparams.py` so if you want to change the levels / downsampling factors you have to modify them there.
- The only constraint for training the LQ-VAE is to use an even number for the batch size, given its use of pairs in the loss.
- Given that `L_lin` enforces the sum operation on the latent domain, you can use the data of both sources together (or any other audio data).
- Checkpoints are save in `logs/lq_vae` (`lq_vae` is the `name` parameter).

## Priors
- After training the LQ-VAE, train two priors on two different classes by calling
```
PYTHONPATH=. mpiexec -n 1 python jukebox/train.py --hps=vqvae,small_prior,all_fp16,cpu_ema --name=pior_source
 --audio_files_dir=data/source/train --test_audio_files_dir=data/source/test --labels=False --train --test --aug_shift
  --aug_blend --prior --levels=3 --level=2 --weight_decay=0.01 --save_iters=1000 --min_duration=24 --sample_length=1048576 
  --bs=16 --n_ctx=8192 --sample=True --sample_iters=1000 --restore_vqvae=logs/lq_vae/checkpoint_lq_vae.pth.tar
```
- Here the data of the source is located in `data/source/train` and `data/source/test` and we assume
the LQ-VAE has 3 levels (topmost level = 2).
- The Transformer model is defined by the parameters of `small_prior` in `hparams.py` and uses a context of `n_ctx=8192` codes.
- The checkpoint path of the LQ-VAE trained in the previous step must be passed to `--restore_vqvae`
- Checkpoints are save in `logs/pior_source` (`pior_source` is the `name` parameter).

## Codebook sums
- Before separation, the sums between all codes must be computed using the LQ-VAE. This can be done using the `codebook_precalc.py` in the `script` folder:
```
PYTHONPATH=.. python codebook_precalc.py --save_path=checkpoints/codebook_sum_precalc.pt 
--restore_vqvae=../logs/lq_vae/checkpoint_lq_vae.pth.tar` --raw_to_tokens=64 --l_bins=2048
--sample_rate=22050 --alpha=[0.5, 0.5] --downs_t=(2, 2, 2) --commit=1.0 --emb_width=64
```


## Separation with trained checkpoints

- Trained checkpoints can be given to `bayesian_inference.py` as following:
  ```
  PYTHONPATH=.. python bayesian_inference.py --shift=shift --path_1=path_1 --path_2=path_2 --bs=bs --restore_vqvae=checkpoints/checkpoint_step_60001_latent.pth.tar
  --restore_priors 'checkpoints/checkpoint_drums_22050_latent_78_19k.pth.tar' checkpoints/checkpoint_latest.pth.tar' --sum_codebook=checkpoints/codebook_precalc_22050_latent.pt
  ```
- `restore_priors` accepts two paths to the first and second prior checkpoints.

# Evaluation

- In order to evaluate the pre-trained checkpoints, run `bayesian_test.py` after you have put the full `Slakh` drums and bass
validation split inside `data/bass/validation` and `data/drums/validation`.

# Future work

- [ ] training of upsamplers for increasing the quality of the separations
- [ ] better rejection sampling method (maybe use verifiers as in https://arxiv.org/abs/2110.14168)

# Citations
If you find the code useful for your research, please consider citing
```
@article{mancusi2021unsupervised,
  title={Unsupervised Source Separation via Bayesian Inference in the Latent Domain},
  author={Mancusi, Michele and Postolache, Emilian and Fumero, Marco and Santilli, Andrea and Cosmo, Luca and Rodol{\`a}, Emanuele},
  journal={arXiv preprint arXiv:2110.05313},
  year={2021}
}
```
as well as the Jukebox baseline:
- Dhariwal, P., Jun, H., Payne, C., Kim, J. W., Radford, A., & Sutskever, I. (2020). Jukebox: A generative model for music. arXiv preprint arXiv:2005.00341.
