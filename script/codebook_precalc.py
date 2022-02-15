import torch

from jukebox.hparams import setup_hparams
from jukebox.make_models import make_vqvae
from jukebox.utils.dist_utils import setup_dist_from_mpi
from tqdm import tqdm
import argparse


def compute_sums(args):
    rank, local_rank, device = setup_dist_from_mpi(port=29530)
    args.sample_length = args.raw_to_tokens * args.l_bins # sample_tokens

    vqvae = make_vqvae(setup_hparams('vqvae', dict(sample_length=args.sample_length,
                                                   sr=args.sample_rate, downs_t=args.downs_t,
                                                   commit=args.commit,
                                                   restore_vqvae=args.restore_vqvae)), 'cuda')

    tokens = torch.arange(args.l_bins).unsqueeze(0).to(device)         # (1, 2048)
    tokens = vqvae.bottleneck.one_level_decode(tokens)                 # (1, 64, 2048)
    y_mean = (args.alpha[0]*tokens.permute(2, 1, 0) +
              args.alpha[1]*tokens.permute(0, 1, 2)).permute(0, 2, 1)  # (2048, 2048, 64)
    y_mean = y_mean.reshape(args.l_bins*args.l_bins, args.emb_width)               # (2048 * 2048, 64)

    y_list = []
    for i in tqdm(range(args.l_bins*args.l_bins)):
        y_list.append(vqvae.bottleneck.one_level_encode(y_mean[i, :].reshape(1, args.emb_width, 1)).cpu())

    temp = torch.tensor(y_list)
    temp = temp.unsqueeze(0)
    torch.save(temp, args.save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Codebook sum pre-computation')

    parser.add_argument('--save_path', type=str, help='Pre-computed sum codebook save path',
                        default='checkpoints/codebook_sum_precalc.pt')
    parser.add_argument('--restore_vqvae', type=str, help='LQ-VAE checkpoint path',
                        default='checkpoints/checkpoint_lq_vae.pth.tar')

    parser.add_argument('--raw_to_tokens', type=int, help='Downsampling factor', default=64)
    parser.add_argument('--l_bins', type=int, help='Number of latent codes', default=2048)
    parser.add_argument('--sample_rate', type=int, help='Sample rate', default=22050)
    parser.add_argument('--alpha', type=float, nargs=2, help='Convex coefficients for the mixture',
                        default=[0.5, 0.5], metavar=('ALPHA_1', 'ALPHA_2'))
    parser.add_argument('--downs_t', type=int, nargs='+', help='Downsampling factors', default=(2, 2, 2))
    parser.add_argument('--commit', type=float, help='Commit scale', default=1.0)
    parser.add_argument('--emb_width', type=int, help='Embedding width', default=64)

    args = parser.parse_args()
    compute_sums(args)