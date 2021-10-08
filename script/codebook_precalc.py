import torch

from jukebox.hparams import setup_hparams
from jukebox.make_models import make_vqvae
from jukebox.utils.dist_utils import setup_dist_from_mpi
from tqdm import tqdm

rank, local_rank, device = setup_dist_from_mpi(port=29530)

raw_to_tokens = 64          # 128
l_bins        = 2048
sample_rate   = 22050       # 44100
alpha         = [0.5, 0.5]  #[1., 1.]    # [0.5, 0.5]
downs_t       = (2, 2, 2)   # (3, 2, 2)
commit        = 1.0         # 0.02
sample_length = raw_to_tokens * l_bins
# restore_vqvae ='../../../logs/checkpoint_step_60001_latent.pth.tar'
# restore_vqvae = 'https://openaipublic.azureedge.net/jukebox/models/5b/vqvae.pth.tar'
restore_vqvae = '../../../logs/checkpoint_vqvae_latent_validation_step_60001.pth.tar'
save_path     = 'results'
data_path     = 'data'
save_name     = 'codebook_precalc_22050_latent_validation.pt'

vqvae = make_vqvae(setup_hparams('vqvae', dict(sample_length=raw_to_tokens*l_bins,
                                               sr=sample_rate, downs_t=downs_t,
                                               commit=commit,
                                               restore_vqvae=restore_vqvae)), 'cuda')

tokens = torch.arange(l_bins).unsqueeze(0).to(device)         # (1, 2048)
tokens = vqvae.bottleneck.one_level_decode(tokens)            # (1, 64, 2048)
y_mean = (alpha[0]*tokens.permute(2, 1, 0) +
          alpha[1]*tokens.permute(0, 1, 2)).permute(0, 2, 1)  # (2048, 2048, 64)
y_mean = y_mean.reshape(l_bins*l_bins, 64)                    # (2048 * 2048, 64)

y_list = []
for i in tqdm(range(l_bins*l_bins)):
    y_list.append(vqvae.bottleneck.one_level_encode(y_mean[i, :].reshape(1, 64, 1)).cpu())

temp = torch.tensor(y_list)
temp = temp.unsqueeze(0)
torch.save(temp, save_name)
