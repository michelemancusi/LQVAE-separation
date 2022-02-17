import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import os.path
import torch
import torchaudio
import argparse
import museval
from matplotlib import pyplot as plt

from jukebox.prior.autoregressive import split_chunks

from jukebox.utils.sample_utils import get_starts

from jukebox.hparams import setup_hparams
from jukebox.make_models import make_vqvae, make_prior
from jukebox.transformer.ops import filter_logits
from jukebox.utils.torch_utils import empty_cache
from jukebox.utils.dist_utils import setup_dist_from_mpi
from jukebox.utils.logger import def_tqdm
from tqdm import tqdm
import numpy as np

# plt.set_cmap('viridis')

def sample_level(vqvae, priors, m, n_samples, n_ctx, hop_length, sample_tokens, sigma=0.01, context=8, fp16=False, temp=1.0,
                 alpha=None, top_k=0, top_p=0.0, bs_chunks=1, window_mode='constant', l_bins=2048, raw_to_tokens=128, device=None,
                 chunk_size=None, latent_loss=True, top_k_posterior=0, delta_likelihood=False, sum_codebook=None, emb_width=64):
    xs_0 = torch.zeros(n_samples, 0, dtype=torch.long, device=device)
    xs_1 = torch.zeros(n_samples, 0, dtype=torch.long, device=device)

    if sample_tokens >= n_ctx:
        for start in get_starts(sample_tokens, n_ctx, hop_length):
            xs_0, xs_1, log_p_0_sum, log_p_1_sum, log_likelihood_sum = sample_single_window(xs_0, xs_1, vqvae, priors, m, n_samples, n_ctx, start=start, sigma=sigma,
                                              context=context, fp16=fp16, temp=temp, alpha=alpha, top_k=top_k,
                                              top_p=top_p, bs_chunks=bs_chunks, window_mode=window_mode, l_bins=l_bins,
                                              raw_to_tokens=raw_to_tokens, device=device, chunk_size=chunk_size,
                                              latent_loss=latent_loss, top_k_posterior=top_k_posterior, delta_likelihood=delta_likelihood, sum_codebook=sum_codebook,
                                                                                            emb_width=emb_width)
    else:
        xs_0, xs_1, log_p_0_sum, log_p_1_sum, log_likelihood_sum = ancestral_sample(vqvae, priors, m, n_samples, sample_tokens=sample_tokens, sigma=sigma,
                                        context=context, fp16=fp16, temp=temp, alpha=alpha, top_k=top_k, top_p=top_p,
                                        bs_chunks=bs_chunks, window_mode=window_mode, l_bins=l_bins,
                                        raw_to_tokens=raw_to_tokens, device=device, latent_loss=latent_loss,
                                        top_k_posterior=top_k_posterior, delta_likelihood=delta_likelihood, sum_codebook=sum_codebook, emb_width=emb_width)
    return xs_0, xs_1, log_p_0_sum, log_p_1_sum, log_likelihood_sum


def sample_single_window(xs_0, xs_1, vqvae, priors, m, n_samples, n_ctx, start=0, sigma=0.01, context=8, fp16=False, temp=1.0,
                 alpha=None, top_k=0, top_p=0.0, bs_chunks=1, window_mode='constant', l_bins=2048, raw_to_tokens=128, device=None,
                 chunk_size=None, latent_loss=True, top_k_posterior=0, delta_likelihood=False, sum_codebook=None,
                         emb_width=64):
    end = start + n_ctx
    # get z already sampled at current level
    x_0 = xs_0[:, start:end]
    x_1 = xs_1[:, start:end]

    sample_tokens = end - start
    conditioning_tokens, new_tokens = x_0.shape[1], sample_tokens - x_0.shape[1]

    if new_tokens <= 0:
        return xs_0, xs_1

    empty_cache()

    x_0, x_1, log_p_0_sum, log_p_1_sum, log_likelihood_sum = sample(x_0, x_1, vqvae, priors, m, n_samples, sample_tokens=sample_tokens, sigma=sigma, context=context,
                      fp16=fp16, temp=temp, alpha=alpha, top_k=top_k, top_p=top_p, bs_chunks=bs_chunks,
                      window_mode=window_mode, l_bins=l_bins, raw_to_tokens=raw_to_tokens, device=device,
                      chunk_size=chunk_size, latent_loss=latent_loss, top_k_posterior=top_k_posterior, delta_likelihood=delta_likelihood,
                      sum_codebook=sum_codebook, emb_width=emb_width)

    # Update z with new sample
    x_0_new = x_0[:, -new_tokens:]
    x_1_new = x_1[:, -new_tokens:]

    xs_0 = torch.cat([xs_0, x_0_new], dim=1)
    xs_1 = torch.cat([xs_1, x_1_new], dim=1)

    return xs_0, xs_1, log_p_0_sum, log_p_1_sum, log_likelihood_sum


def sample(xs_0, xs_1, vqvae, priors, m, n_samples, sample_tokens, sigma=0.01, context=8, fp16=False, temp=1.0,
           alpha=None, top_k=0, top_p=0.0, bs_chunks=1, window_mode='constant', l_bins=2048, raw_to_tokens=128,
           device=None, chunk_size=None, latent_loss=True, top_k_posterior=0, delta_likelihood=False, sum_codebook=None,
           emb_width=64):
    no_past_context = (xs_0 is None or xs_0.shape[1] == 0 or xs_1 is None or xs_1.shape[1] == 0)
    with torch.no_grad():
        if no_past_context:
            x_0, x_1, log_p_0_sum, log_p_1_sum, log_likelihood_sum = ancestral_sample(vqvae, priors, m, n_samples, sample_tokens=sample_tokens, sigma=sigma,
                                        context=context, fp16=fp16, temp=temp, alpha=alpha, top_k=top_k, top_p=top_p,
                                        bs_chunks=bs_chunks, window_mode=window_mode, l_bins=l_bins,
                                        raw_to_tokens=raw_to_tokens, device=device, latent_loss=latent_loss,
                                        top_k_posterior=top_k_posterior, delta_likelihood=delta_likelihood, sum_codebook=sum_codebook,
                                        emb_width=emb_width)
        else:
            x_0, x_1 = primed_sample(xs_0, xs_1, vqvae, priors, m, n_samples, sample_tokens=sample_tokens,
                                     sigma=sigma, context=context, fp16=fp16, temp=temp, alpha=alpha, top_k=top_k,
                                     top_p=top_p, bs_chunks=bs_chunks, window_mode=window_mode, l_bins=l_bins,
                                     raw_to_tokens=raw_to_tokens, device=device, chunk_size=chunk_size,
                                     latent_loss=latent_loss, top_k_posterior=top_k_posterior, delta_likelihood=delta_likelihood,
                                     sum_codebook=sum_codebook, emb_width=emb_width)
            nll_sum_0 = None
            nll_sum_1 = None
    return x_0, x_1, nll_sum_0, nll_sum_1, None


def primed_sample(x_0, x_1, vqvae, priors, m, n_samples, sample_tokens, sigma, context, fp16, temp, alpha, top_k,
                  top_p, bs_chunks, window_mode, l_bins, raw_to_tokens, device, chunk_size=None, latent_loss=True,
                  top_k_posterior=0, delta_likelihood=False, sum_codebook=None, emb_width=64):

    x_cond = torch.zeros((n_samples, 1, priors[0].width), dtype=torch.float).to(device)


    if latent_loss:
        codebook = torch.load(sum_codebook)
        codebook = codebook.to(device)  # shape (1, 2048*2048)
        M = vqvae.bottleneck.one_level_decode(codebook)  # (1, 64, 2048*2048)
        M = M.squeeze(0)  # (64, 2048*2048)
        M = M.permute(1, 0)  # (2048*2048, 64)
        M = M.reshape(l_bins, l_bins, emb_width)  # (2048, 2048, 64)
        codebook = codebook.squeeze(0).reshape(l_bins, l_bins)
    else:
        tokens = torch.arange(l_bins).reshape(1, l_bins, 1).repeat(n_samples, 1, 1).to(device)

    xs_0 = torch.split(x_0, 1, dim=1)
    xs_0 = list(xs_0)
    xs_1 = torch.split(x_1, 1, dim=1)
    xs_1 = list(xs_1)

    # Fill up key/value cache for past context by runing forward pass.
    # We do so in chunks instead of doing the whole past in one forward pass to reduce max memory usage.
    if chunk_size is None:
        chunk_size = len(xs_0)
    #assert len(xs) % chunk_size == 0, f'expected {len(xs)} to be divisible by {chunk_size}'
    chunk_sizes = split_chunks(len(xs_0), chunk_size)
    start = 0
    x_0 = None
    x_1 = None

    for current_chunk_size in chunk_sizes:
        xs_0_prime = []
        xs_1_prime = []

        for sample_t in range(start, start + current_chunk_size):
            x_0_prime, _ = priors[0].get_emb(sample_t, n_samples, x_0, x_cond, y_cond=None)
            x_0 = xs_0[sample_t]
            xs_0_prime.append(x_0_prime)

            x_1_prime, _ = priors[1].get_emb(sample_t, n_samples, x_1, x_cond, y_cond=None)
            x_1 = xs_1[sample_t]
            xs_1_prime.append(x_1_prime)

        start = start + current_chunk_size

        x_0_prime = torch.cat(xs_0_prime, dim=1)
        x_1_prime = torch.cat(xs_1_prime, dim=1)

        del xs_0_prime
        del xs_1_prime

        x_0_prime = priors[0].transformer(x_0_prime, encoder_kv=None, sample=True, fp16=fp16) # Transformer
        x_1_prime = priors[1].transformer(x_1_prime, encoder_kv=None, sample=True, fp16=fp16) # Transformer

        del x_0_prime
        del x_1_prime

    empty_cache()
    priors[0].transformer.check_cache(n_samples, len(xs_0), fp16)
    priors[1].transformer.check_cache(n_samples, len(xs_1), fp16)

    x_0 = xs_0[-1]
    x_1 = xs_1[-1]

    empty_cache()
    for sample_t in range(len(xs_0), sample_tokens):
        x_0, cond_0 = priors[0].get_emb(sample_t, n_samples, x_0, x_cond, y_cond=None)
        priors[0].transformer.check_cache(n_samples, sample_t, fp16)
        x_0 = priors[0].transformer(x_0, encoder_kv=None, sample=True, fp16=fp16)  # Transformer
        if priors[0].add_cond_after_transformer:
            x_0 = x_0 + cond_0
        assert x_0.shape == (n_samples, 1, priors[0].width)
        x_0 = priors[0].x_out(x_0)  # Predictions

        x_0 = x_0 / temp
        x_0 = filter_logits(x_0, top_k=top_k, top_p=top_p)
        p_0 = torch.distributions.Categorical(logits=x_0).probs  # Sample and replace x
        log_p_0 = torch.log(p_0)  # n_samples, 1, 2048

        x_1, cond_1 = priors[1].get_emb(sample_t, n_samples, x_1, x_cond, y_cond=None)
        priors[1].transformer.check_cache(n_samples, sample_t, fp16)
        x_1 = priors[1].transformer(x_1, encoder_kv=None, sample=True, fp16=fp16)  # Transformer
        if priors[1].add_cond_after_transformer:
            x_1 = x_1 + cond_1
        x_1 = priors[1].x_out(x_1)  # Predictions
        x_1 = x_1 / temp
        x_1 = filter_logits(x_1, top_k=top_k, top_p=top_p)
        p_1 = torch.distributions.Categorical(logits=x_1).probs  # Sample and replace x
        log_p_1 = torch.log(p_1)  # n_samples, 1, 2048

        log_p = log_p_0 + log_p_1.permute(0, 2, 1)  # n_samples, 2048, 2048 (p_1 sulle righe, p_0 sulle colonne)
        # print(f"{log_p.shape = }")

        ### START LOG LIKELIHOOD
        if not latent_loss:

            if window_mode == 'increment':
                x_0 = xs_0[sample_t // context * context: sample_t // context * context + sample_t % context]
                x_1 = xs_1[sample_t // context * context: sample_t // context * context + sample_t % context]
            elif window_mode == 'constant':
                x_0 = xs_0[sample_t - context: sample_t]
                x_1 = xs_1[sample_t - context: sample_t]
            else:
                raise NotImplementedError(f'Window mode {window_mode} not found')

            x_0 = torch.zeros(n_samples, 0).long().to(device) if x_0 == [] else torch.cat(x_0, dim=1)  # n_samples, t-1
            x_1 = torch.zeros(n_samples, 0).long().to(device) if x_1 == [] else torch.cat(x_1, dim=1)  # n_samples, t-1

            x_0 = torch.cat((x_0.unsqueeze(1).repeat(1, l_bins, 1), tokens), dim=-1).reshape(n_samples * l_bins, -1)  # n_samples * 2048, t
            x_1 = torch.cat((x_1.unsqueeze(1).repeat(1, l_bins, 1), tokens), dim=-1).reshape(n_samples * l_bins, -1)
            # n_samples * 2048, t-1 * 128, 1
            # n_samples * 2048, t-1 * 128
            # n_samples * 2048, 128
            d1 = vqvae.decode([x_0], start_level=2, bs_chunks=bs_chunks).squeeze(-1)[:, -raw_to_tokens:]
            d2 = vqvae.decode([x_1], start_level=2, bs_chunks=bs_chunks).squeeze(-1)[:, -raw_to_tokens:]

            # n_samples, 2048, 2048, 128
            y_mean = alpha[0]*d2.reshape(n_samples, l_bins, 1, raw_to_tokens) + alpha[1]*d1.reshape(n_samples, 1, l_bins, raw_to_tokens)

            m_t = m[:, (sample_t)*raw_to_tokens:(sample_t+1)*raw_to_tokens].reshape(1, 1, 1, -1)
            # print(f"{m_t.shape = }")
            log_likelihood = -(1/(2.*(sigma**2)))*torch.linalg.norm(y_mean - m_t, dim = -1)**2  # n_samples, 2048, 2048
        else:
            # M: 2048, 2048, 64
            # m: 1, 1, 1, 64
            # log_likehood: 2048, 2048, 1
            # log_p: N, 2048, 2048, 1
            # print(f"{m = }")
            if delta_likelihood:
                factors = (codebook == m[:, sample_t]).nonzero()
                log_likelihood = -1/(torch.zeros((n_samples, 2048, 2048)) + 0.0000000000000001).cuda()
                if (factors.shape[0] != 0):
                    log_likelihood[:, factors[0], factors[1]] = torch.log(torch.tensor(1/factors.shape[0]).cuda())
            else:
                log_likelihood = -(1/(2.*(sigma**2)))*torch.linalg.norm(M - m[:, :, sample_t].unsqueeze(0), dim=-1)**2  # n_samples, 2048, 2048


    # log_likelihood = -(1/(2.*sigma))*torch.linalg.norm(y_mean - m_t, dim = -1)**2  # n_samples, 2048, 2048
        # print(f"{log_likelihood.shape = }")
        # print(f"{torch.min(log_likelihood) = }")
        # print(f"{torch.max(log_likelihood) = }")
        # print(f"{torch.min(log_p) = }")
        # print(f"{torch.max(log_p) = }")
        #### END LIKELIHOOD ####

        log_posterior = log_likelihood + log_p   # n_samples, 2048, 2048

        # print(f"{torch.min(log_posterior) = }")
        # print(f"{torch.max(log_posterior) = }")
        log_posterior = log_posterior.reshape(n_samples, l_bins * l_bins)  # n_samples, 2048 * 2048
        log_posterior = filter_logits(log_posterior.unsqueeze(1), top_k=top_k_posterior, top_p=0.0).squeeze(1)
        posterior = torch.distributions.Categorical(logits=log_posterior)
        # print(f"{torch.min(posterior.probs) = }")
        # print(f"{torch.max(posterior.probs) = }")

        x = posterior.sample()  # n_samples
        x_0 = (x % l_bins).reshape(n_samples, -1)  # n_samples, 1
        x_1 = (x // l_bins).reshape(n_samples, -1)  # n_samples, 1

        xs_0.append(x_0.clone())
        xs_1.append(x_1.clone())

    del x_0
    del x_1
    priors[0].transformer.del_cache()
    priors[1].transformer.del_cache()

    x_0 = torch.cat(xs_0, dim=1)  # n_samples, sample_tokens
    # print(f"{x_0.shape = }")
    x_0 = priors[0].postprocess(x_0, sample_tokens)  # n_samples, sample_tokens
    # print(f"{x_0.shape = }")
    x_1 = torch.cat(xs_1, dim=1)
    x_1 = priors[1].postprocess(x_1, sample_tokens)
    return x_0, x_1


def ancestral_sample(vqvae, priors, m, n_samples, sample_tokens, sigma=0.01, context=8, fp16=False, temp=1.0, alpha=None,
                     top_k=0, top_p=0.0, bs_chunks=1, window_mode='constant', l_bins=2048, raw_to_tokens=128, device=None,
                     latent_loss=True, top_k_posterior=0, delta_likelihood=False, sum_codebook=None, emb_width=64):
    x_cond = torch.zeros((n_samples, 1, priors[0].width), dtype=torch.float).to(device)
    xs_0, xs_1, x_0, x_1 = [], [], None, None

    log_p_0_sum = torch.zeros((n_samples,)).to(device)
    log_p_1_sum = torch.zeros((n_samples,)).to(device)
    log_likelihood_sum = torch.zeros((n_samples,)).to(device)

    if latent_loss:
        codebook = torch.load(sum_codebook)
        codebook = codebook.to(device)  # shape (1, 2048*2048)
        M = vqvae.bottleneck.one_level_decode(codebook)  # (1, 64, 2048*2048)
        M = M.squeeze(0)  # (64, 2048*2048)
        M = M.permute(1, 0)  # (2048*2048, 64)
        M = M.reshape(l_bins, l_bins, emb_width)  # (2048, 2048, 64)
        codebook = codebook.squeeze(0).reshape(l_bins, l_bins)
    else:
        tokens = torch.arange(l_bins).reshape(1, l_bins, 1).repeat(n_samples, 1, 1).to(device)

    for sample_t in def_tqdm(range(0, sample_tokens)):
        x_0, cond_0 = priors[0].get_emb(sample_t, n_samples, x_0, x_cond, y_cond=None)
        priors[0].transformer.check_cache(n_samples, sample_t, fp16)
        x_0 = priors[0].transformer(x_0, encoder_kv=None, sample=True, fp16=fp16) # Transformer
        if priors[0].add_cond_after_transformer:
            x_0 = x_0 + cond_0
        assert x_0.shape == (n_samples, 1, priors[0].width)
        x_0 = priors[0].x_out(x_0) # Predictions

        x_0 = x_0 / temp
        x_0 = filter_logits(x_0, top_k=top_k, top_p=top_p)
        p_0 = torch.distributions.Categorical(logits=x_0).probs # Sample and replace x
        log_p_0 = torch.log(p_0) # n_samples, 1, 2048

        x_1, cond_1 = priors[1].get_emb(sample_t, n_samples, x_1, x_cond, y_cond=None)
        priors[1].transformer.check_cache(n_samples, sample_t, fp16)
        x_1 = priors[1].transformer(x_1, encoder_kv=None, sample=True, fp16=fp16) # Transformer
        if priors[1].add_cond_after_transformer:
            x_1 = x_1 + cond_1
        x_1 = priors[1].x_out(x_1) # Predictions
        x_1 = x_1 / temp
        x_1 = filter_logits(x_1, top_k=top_k, top_p=top_p)
        p_1 = torch.distributions.Categorical(logits=x_1).probs # Sample and replace x
        log_p_1 = torch.log(p_1) # n_samples, 1, 2048

        log_p = log_p_0 + log_p_1.permute(0, 2, 1)  # n_samples, 2048, 2048 (p_1 sulle righe, p_0 sulle colonne)

        #plt.figure()
        #plt.matshow(log_p[0].cpu().numpy(), vmin=-30., vmax=0.)
        # print(f"log_p[0] = {log_p[0]}")
        # plt.colorbar()
        # plt.savefig(f'log_p_t={sample_t}', dpi=200)
        # plt.close()
        # plt.figure()

        #print(f"{log_p.shape = }")

        ### START LOG LIKELIHOOD
        if not latent_loss:
            if window_mode == 'increment':
                x_0 = xs_0[sample_t // context * context: sample_t // context * context + sample_t % context]
                x_1 = xs_1[sample_t // context * context: sample_t // context * context + sample_t % context]
            elif window_mode == 'constant':
                x_0 = xs_0[sample_t - context: sample_t]
                x_1 = xs_1[sample_t - context: sample_t]
            else:
                raise NotImplementedError(f'Window mode {window_mode} not found')

            x_0 = torch.zeros(n_samples, 0).long().to(device) if x_0 == [] else torch.cat(x_0, dim=1)  # n_samples, t-1
            x_1 = torch.zeros(n_samples, 0).long().to(device) if x_1 == [] else torch.cat(x_1, dim=1)  # n_samples, t-1

            x_0 = torch.cat((x_0.unsqueeze(1).repeat(1, l_bins, 1), tokens), dim=-1).reshape(n_samples * l_bins, -1)  # n_samples * 2048, t
            x_1 = torch.cat((x_1.unsqueeze(1).repeat(1, l_bins, 1), tokens), dim=-1).reshape(n_samples * l_bins, -1)

            # n_samples * 2048, t-1 * 128, 1
            # n_samples * 2048, t-1 * 128
            # n_samples * 2048, 128
            d1 = vqvae.decode([x_0], start_level=2, bs_chunks=bs_chunks).squeeze(-1)[:, -raw_to_tokens:]
            d2 = vqvae.decode([x_1], start_level=2, bs_chunks=bs_chunks).squeeze(-1)[:, -raw_to_tokens:]

            # n_samples, 2048, 2048, 128
            y_mean = alpha[0]*d2.reshape(n_samples, l_bins, 1, raw_to_tokens) + alpha[1]*d1.reshape(n_samples, 1, l_bins, raw_to_tokens)

            m_t = m[:, (sample_t)*raw_to_tokens:(sample_t+1)*raw_to_tokens].reshape(1, 1, 1, -1)
            log_likelihood = -(1/(2.*(sigma**2)))*torch.linalg.norm(y_mean - m_t, dim=-1)**2  # n_samples, 2048, 2048

        else:
            # M: 2048, 2048, 64
            # m: 1, 1, 1, 64
            # log_likehood: 2048, 2048, 1
            # log_p: N, 2048, 2048, 1

            if delta_likelihood:
                factors = (codebook == m[:, sample_t]).nonzero()
                #log_likelihood = -1/(torch.zeros((n_samples, 2048, 2048)) + 0.0000000000000001).cuda()
                prop_fact = torch.tensor(0.9999).cuda()
                n_factors = factors.shape[0]
                # print(n_factors)
                log_likelihood = torch.log((1-prop_fact)/(2048*2048 - n_factors))*torch.ones((n_samples, 2048, 2048)).cuda()
                log_likelihood[:, factors[0], factors[1]] = torch.log(prop_fact/(factors.shape[0])) #torch.log(torch.tensor(1/factors.shape[0]).cuda())
            else:
                # l2 = torch.linalg.norm(M - m[:, :, sample_t].unsqueeze(0), dim=-1)**2

                #mp = log_p.mean(-1).mean(-1)
                #ml = l2.mean(-1).mean(-1)
                #sigma_rejection_squared = - ml / (2*mp)
                # print(f"sigma_rejection = {torch.sqrt(sigma_rejection_squared)}")
                # log_likelihood = (-(1/(2.*(sigma.reshape(-1, 1, 1)))) * l2.unsqueeze(0).repeat(bs, 1, 1))
                log_likelihood = -(1/(2.*(sigma**2)))*torch.linalg.norm(M - m[:, :, sample_t].unsqueeze(0), dim=-1)**2  # n_samples, 2048, 2048
                #print(f"log_likelihood = {log_likelihood}")
                # plt.figure()
                # plt.matshow(log_likelihood.cpu().numpy(), vmin=-100., vmax=0.)#, vmin=-1000., vmax=0.)
                #  plt.colorbar()
                # plt.savefig(f'log_likelihood_t={sample_t}', dpi=200)
                # plt.close()
                # plt.figure()

        # print(f"{log_likelihood.shape = }")
        # print(f"{torch.min(log_likelihood) = }")
        # print(f"{torch.max(log_likelihood) = }")
        # print(f"{torch.min(log_p) = }")
        # print(f"{torch.max(log_p) = }")
        #### END LIKELIHOOD ####

        log_posterior = log_likelihood + log_p
        # log_posterior = log_likelihood.unsqueeze(-1).repeat(n_samples, 1, 1)   # n_samples, 2048, 2048 #.unsqueeze(-1).repeat(32, 1, 1)



        # plt.figure()

        # print(f"{torch.min(log_posterior) = }")
        # print(f"{torch.max(log_posterior) = }")
        log_posterior = log_posterior.reshape(n_samples, l_bins*l_bins) # n_samples, 2048 * 2048
        log_posterior = filter_logits(log_posterior.unsqueeze(1), top_k=top_k_posterior, top_p=0.0).squeeze(1)
        posterior = torch.distributions.Categorical(logits=log_posterior)

        #plt.figure()
        #plt.matshow(posterior.logits.reshape(-1, 2048, 2048)[0].cpu().numpy(), vmin=-100., vmax=0.)#, vmin=-100., vmax=0.)
        # print(f"log_posterior[0] = {log_posterior[0]}")
        # plt.colorbar()
        #plt.savefig(f'log_posterior_t={sample_t}', dpi=200)
        #plt.close()

        # print(f"{torch.min(posterior.probs) = }")
        # print(f"{torch.max(posterior.probs) = }")

        x = posterior.sample() # n_samples
        x_0 = (x % l_bins).reshape(n_samples, -1) # n_samples, 1
        x_1 = (x // l_bins).reshape(n_samples, -1) # n_samples, 1

        log_p_0_sum += log_p_0[range(x_0.shape[0]), :, x_0.squeeze(-1)].squeeze(-1)
        log_p_1_sum += log_p_1[range(x_1.shape[0]), :, x_1.squeeze(-1)].squeeze(-1)
        #  log_likelihood_sum += log_likelihood[x_1, x_0].squeeze(-1)

        xs_0.append(x_0.clone())
        xs_1.append(x_1.clone())

    del x_0
    del x_1
    priors[0].transformer.del_cache()
    priors[1].transformer.del_cache()

    x_0 = torch.cat(xs_0, dim=1) # n_samples, sample_tokens
    #print(f"{x_0.shape = }")
    x_0 = priors[0].postprocess(x_0, sample_tokens) # n_samples, sample_tokens
    #print(f"{x_0.shape = }")
    x_1 = torch.cat(xs_1, dim=1)
    x_1 = priors[1].postprocess(x_1, sample_tokens)
    return x_0, x_1, log_p_0_sum, log_p_1_sum, None #log_likelihood_sum


def save_samples(x_0, x_1, res_0, res_1, sample_rate, alpha, path='results'):

    if not os.path.exists(f"{path}/"):
        os.makedirs(f"{path}/")

    torch.save(x_0.cpu(), f"{path}/res_first_prior.pt")
    torch.save(x_1.cpu(), f"{path}/res_second_prior.pt")

    for i in range(res_0.shape[0]):
        torchaudio.save(f'{path}/res_{i}_first_prior.wav', res_0[i].unsqueeze(0).cpu(), sample_rate=sample_rate)
        torchaudio.save(f'{path}/res_{i}_second_prior.wav', res_1[i].unsqueeze(0).cpu(), sample_rate=sample_rate)
        torchaudio.save(f'{path}/res_{i}_mix.wav', alpha[0]*res_0[i].unsqueeze(0).cpu() + alpha[1]*res_1[i].
                        unsqueeze(0).cpu(), sample_rate=sample_rate)


def sdr(track1, track2):
    sdr_metric = museval.evaluate(track1, track2)
    sdr_metric[0][sdr_metric[0] == np.inf] = np.nan
    return np.nanmedian(sdr_metric[0])


def rejection_sampling(nll0, nll1, res0, res1, remaining0, remaining1, m, alpha, bs, rejection_sigma,
                       n_samples):
    nll_sum_0_sorted, indices_nll_sum_0_sorted = torch.sort(nll0)
    nll_sum_1_sorted, indices_nll_sum_1_sorted = torch.sort(nll1)

    global_likelihood = torch.zeros((bs, bs))
    global_posterior = torch.zeros((bs, bs))
    global_prior = torch.zeros((bs, bs))
    global_prior_reverse = torch.zeros((bs, bs))
    global_l2 = torch.zeros((bs, bs))

    # sdr(alpha[0]*res0[i, :].reshape(1, -1, 1).cpu().numpy() +
    #                             alpha[1]*res1[j, :].reshape(1, -1, 1).cpu().numpy(),
    #                            m.unsqueeze(-1).cpu().numpy())

    for i in tqdm(range(bs)):
        for j in range(bs):
            global_prior_reverse[i, j] = nll0[i] + nll1[j]

    global_prior_reverse = global_prior_reverse.reshape(bs*bs)
    global_prior_reverse = torch.distributions.Categorical(logits=global_prior_reverse)
    #plt.matshow(global_prior_reverse.logits.reshape(bs, bs))
    #plt.figure()
    global_prior_reverse = global_prior_reverse.logits / n_samples
    global_prior_reverse = torch.distributions.Categorical(logits=global_prior_reverse).logits.reshape(bs, bs)

    for i in tqdm(range(bs)):
        for j in range(bs):
            # drop = remaining0[i] and remaining1[j] #-np.inf if (not remaining0[i] or not remaining1[j]) else 0.

            global_prior[i, j] = (nll0[i] + nll1[j]) / n_samples # if drop else -np.inf
            global_l2[i, j] = torch.linalg.norm((alpha[0]*res0[i] +
                                                alpha[1]*res1[j] - m), dim=-1)**2

    mp = global_prior.mean()
    ml = global_l2.mean()
    sigma_rejection_squared = - ml / (2*mp)
    print(f"sigma_rejection = {torch.sqrt(sigma_rejection_squared)}")
    global_likelihood = (-(1/(2.*(sigma_rejection_squared)))* global_l2)
                                       #if drop else 0.)
            # global_posterior[i, j] = global_likelihood[i, j] + global_prior[i, j]

    global_prior = global_prior.reshape(bs*bs)
    global_prior = torch.distributions.Categorical(logits=global_prior)
    global_prior_p = global_prior.probs.reshape(bs, bs)
    global_prior = global_prior.logits.reshape(bs, bs)

    print(f"global prior test: {global_prior == global_prior_reverse}")
    global_posterior = global_prior + global_likelihood

    global_posterior = global_posterior.reshape(bs*bs) # n_samples, 2048 * 2048
    global_posterior = torch.distributions.Categorical(logits=global_posterior)
    global_posterior = global_posterior.probs.reshape(bs, bs)

    marginal_0 = global_posterior.sum(dim=-1)
    marginal_1 = global_posterior.sum(dim=0)
    marginal_1_sorted, marginal_1_idx_sorted = torch.sort(marginal_1)
    marginal_0_sorted, marginal_0_idx_sorted = torch.sort(marginal_0)

    print(f"marginal_0_sorted = {marginal_0_sorted}")
    print(f"marginal_1_sorted = {marginal_1_sorted}")
    print(f"marginal_0_idx_sorted = {marginal_0_idx_sorted}")
    print(f"marginal_1_idx_sorted = {marginal_1_idx_sorted}")

    global_l2_vectorized = global_l2.reshape(bs*bs)  # righe: prior_0, colonne: prior_1
    global_l2_topk_vectorized, global_l2_topk_idx_vectorized  = torch.topk(global_l2_vectorized, k=bs, largest=False)
    print(f"global_l2_topk = {global_l2_topk_vectorized}")
    print(f"global_l2_topk_idx = {[(idx // bs, idx % bs) for idx in global_l2_topk_idx_vectorized]}")

    rejection_index = torch.argmax(global_posterior)
    plt.figure()
    plt.matshow(global_prior.cpu().numpy())
    plt.figure()
    plt.matshow(global_prior_reverse.cpu().numpy())
    plt.figure()
    plt.matshow(global_likelihood.cpu().numpy())
    plt.figure()
    plt.matshow(global_posterior.cpu().numpy())


    # print(f"rejection_index = {rejection_index}")
    # print(f"rejection_indices = {(rejection_index // bs, rejection_index % bs)}")

    print(f"top nll0: {nll_sum_0_sorted}")
    print(f"top nll1: {nll_sum_1_sorted}")
    print(f"top nll0 indices: {indices_nll_sum_0_sorted}")
    print(f"top nll1 indices: {indices_nll_sum_1_sorted}")
    print(f"global_likelihood: {global_likelihood}")
    print(f"global_posterior: {global_posterior}")
    print(f"global_prior: {global_prior}")
    print(f"global_prior_p: {global_prior_p}")

    # print(f"rejection_vector_indices: {rejection_vector_indices}")
    # print(f"rejection_vector_best: {rejection_vector_indices[-1]}")


def evaluate_sdr_gt(gt0, gt1, res0, res1):
    sdr_0 = torch.zeros((res0.shape[0],))
    sdr_1 = torch.zeros((res1.shape[0],))
    for i in range(res0.shape[0]):
        sdr_0[i] = sdr(gt0.unsqueeze(-1).cpu().numpy(), res0[i, :].reshape(1, -1, 1).cpu().numpy())
        sdr_1[i] = sdr(gt1.unsqueeze(-1).cpu().numpy(), res1[i, :].reshape(1, -1, 1).cpu().numpy())

    sdr_0_sorted, sdr_0_sorted_idx = torch.sort(sdr_0)
    sdr_1_sorted, sdr_1_sorted_idx = torch.sort(sdr_1)

    print(f"sdr_0_sorted = {sdr_0_sorted}")
    print(f"sdr_1_sorted = {sdr_1_sorted}")

    print(f"sdr_0_sorted_idx = {sdr_0_sorted_idx}")
    print(f"sdr_1_sorted_idx = {sdr_1_sorted_idx}")


def evaluate_sdr_real(gt0, gt1, res0, res1):
    sdr_0 = torch.zeros((res0.shape[0],))
    sdr_1 = torch.zeros((res1.shape[0],))
    for i in range(res0.shape[0]):
        sdr_0[i] = sdr(gt0.unsqueeze(-1).cpu().numpy(), res0[i, :].reshape(1, -1, 1).cpu().numpy())
        sdr_1[i] = sdr(gt1.unsqueeze(-1).cpu().numpy(), res1[i, :].reshape(1, -1, 1).cpu().numpy())

    sdr_0_sorted, sdr_0_sorted_idx = torch.sort(sdr_0)
    sdr_1_sorted, sdr_1_sorted_idx = torch.sort(sdr_1)

    print(f"sdr_real_0_sorted = {sdr_0_sorted}")
    print(f"sdr_real_1_sorted = {sdr_1_sorted}")

    print(f"sdr_real_0_sorted_idx = {sdr_0_sorted_idx}")
    print(f"sdr_real_1_sorted_idx = {sdr_1_sorted_idx}")


def evaluate_l2_gt(gt_0, gt_1, res_0, res_1):
    l2_0 = torch.zeros((res_0.shape[0],))
    l2_1 = torch.zeros((res_1.shape[0],))
    for i in range(res_0.shape[0]):
        l2_0[i] = torch.linalg.norm(res_0[i, :] - gt_0.squeeze(0)) ** 2
        l2_1[i] = torch.linalg.norm(res_1[i, :] - gt_1.squeeze(0)) ** 2

    l2_0_sorted, l2_0_sorted_idx = torch.sort(l2_0)
    l2_1_sorted, l2_1_sorted_idx = torch.sort(l2_1)

    print(f"l2_0_sorted (first is best) = {l2_0_sorted}")
    print(f"l2_1_sorted (first is best) = {l2_1_sorted}")
    print(f"l2_0_sorted_idx (first is best) = {l2_0_sorted_idx}")
    print(f"l2_1_sorted_idx (first is best) = {l2_1_sorted_idx}")


#  sample only
def rejection_sampling_latent(log_p_0_sum, log_p_1_sum, log_likelihood_sum, bs):
    global_posterior = torch.zeros((bs,))
    global_prior = torch.zeros((bs,))

    for i in tqdm(range(bs)):
        global_prior[i]      = log_p_0_sum[i] + log_p_1_sum[i]
        global_posterior[i]  = log_likelihood_sum[i] + global_prior[i]

    # rejection_vector = global_posterior.reshape(bs*bs)  # righe: prior_0, colonne: prior_1

    rejection_vector, rejection_vector_idx = torch.sort(global_posterior)
    print(f"rejection_vector = {rejection_vector}")
    print(f"rejection_vector_idx = {rejection_vector_idx}")


# def cross_prior_rejection(x_0, x_1, log_p_0_sum, log_p_1_sum, priors, sample_tokens):
#     n_samples = x_0.shape[0]
#     x_cond = torch.zeros((n_samples, 1, priors[0].width), dtype=torch.float).to(device)
#
#     log_p_0_1_sum = torch.zeros((n_samples,)).to(device)
#     log_p_1_0_sum = torch.zeros((n_samples,)).to(device)
#
#     for sample_t in def_tqdm(range(0, sample_tokens)):
#         x_0_1, cond_0_1 = priors[0].get_emb(sample_t, n_samples, x_1[:, sample_t - 1].unsqueeze(-1), x_cond, y_cond=None)
#         priors[0].transformer.check_cache(n_samples, sample_t, fp16)
#         x_0_1 = priors[0].transformer(x_0_1, encoder_kv=None, sample=True, fp16=fp16) # Transformer
#         if priors[0].add_cond_after_transformer:
#             x_0_1 = x_0_1 + cond_0_1
#         assert x_0_1.shape == (n_samples, 1, priors[0].width)
#         x_0_1 = priors[0].x_out(x_0_1)  # Predictions
#         p_0_1 = torch.distributions.Categorical(logits=x_0_1).probs # Sample and replace x
#         log_p_0_1 = torch.log(p_0_1) # n_samples, 1, 2048
#
#         x_1_0, cond_1_0 = priors[1].get_emb(sample_t, n_samples, x_0[:, sample_t - 1].unsqueeze(-1), x_cond, y_cond=None)
#         priors[1].transformer.check_cache(n_samples, sample_t, fp16)
#         x_1_0 = priors[1].transformer(x_1_0, encoder_kv=None, sample=True, fp16=fp16) # Transformer
#         if priors[1].add_cond_after_transformer:
#             x_1_0 = x_1_0 + cond_1_0
#         assert x_1_0.shape == (n_samples, 1, priors[1].width)
#         x_1_0 = priors[1].x_out(x_1_0)  # Predictions
#         p_1_0 = torch.distributions.Categorical(logits=x_1_0).probs  # Sample and replace x
#         log_p_1_0 = torch.log(p_1_0)   # n_samples, 1, 1024??? #2048
#
#         log_p_0_1_sum += (log_p_0_1[range(x_0.shape[0]), :, x_1[:, sample_t]]).squeeze(-1)   # .squeeze(-1)
#         log_p_1_0_sum += (log_p_1_0[range(x_0.shape[0]), :, x_0[:, sample_t]]).squeeze(-1)   # x_0[:, sample_t]].squeeze(-1)
#
#     priors[0].transformer.del_cache()
#     priors[1].transformer.del_cache()
#
#     remaining_0 = (log_p_1_0_sum < log_p_0_sum)
#     remaining_1 = (log_p_0_1_sum < log_p_1_sum)
#
#     # log_p_0_sorted, log_p_0_sorted_idx = torch.sort(log_p_0_sum)
#     # log_p_1_sorted, log_p_1_sorted_idx = torch.sort(log_p_1_sum)
#     # log_p_0_1_sorted, log_p_0_1_sorted_idx = torch.sort(log_p_0_1_sum)
#     # log_p_1_0_sorted, log_p_1_0_sorted_idx = torch.sort(log_p_1_0_sum)
#
#     print(f"log_p_0_sum: {log_p_0_sum}")
#     print(f"log_p_0_1_sum: {log_p_0_1_sum}")
#     print(f"log_p_1_sum: {log_p_1_sum}")
#     print(f"log_p_1_0_sum: {log_p_1_0_sum}")
#     return remaining_0, remaining_1

def create_mixture_from_audio_files(path_audio_1, path_audio_2, raw_to_tokens, sample_tokens,
                                    vqvae, save_path, sample_rate, alpha, device='cuda', shift=0.):
    #sqrt(2) è grazie a Giorgio che si è accordo che durante il training le tracce vengono caricate con una funzione load_audio di jukebox ch normalizza con sqrt(2) le tracce audio
    m1, _ = torchaudio.load(path_audio_1)/np.sqrt(2) #deve essere di dimensioni (1, length) es (1, 5060608)
    m2, _ = torchaudio.load(path_audio_2)/np.sqrt(2)
    shift = int(shift * sample_rate)
    assert sample_tokens * raw_to_tokens <= min(m1.shape[-1], m2.shape[-1]), "Sources must be longer than sample_tokens"
    minin = sample_tokens * raw_to_tokens
    m1_real    = m1[:, shift:shift+minin]
    m2_real    = m2[:, shift:shift+minin]
    mix        = alpha[0]*m1_real + alpha[1]*m2_real
    torchaudio.save(f'{save_path}/real_mix.wav', mix.cpu().squeeze(-1), sample_rate=sample_rate)
    torchaudio.save(f'{save_path}/real_m1.wav',  m1_real.cpu().squeeze(-1), sample_rate=sample_rate)
    torchaudio.save(f'{save_path}/real_m2.wav',  m2_real.cpu().squeeze(-1), sample_rate=sample_rate)

    z_m1 = vqvae.encode(m1_real.unsqueeze(-1).to(device), start_level=2, bs_chunks=1)[0]
    z_m2 = vqvae.encode(m2_real.unsqueeze(-1).to(device), start_level=2, bs_chunks=1)[0]
    z_mixture = vqvae.encode(mix.unsqueeze(-1).to(device), start_level=2, bs_chunks=1)[0]
    latent_mix = vqvae.bottleneck.decode([z_mixture]*3)[-1]
    mix = vqvae.decode([z_mixture], start_level=2, bs_chunks=1).squeeze(-1)  # 1, 8192*128
    m1 = vqvae.decode([z_m1], start_level=2, bs_chunks=1).squeeze(-1)  # 1, 8192*128
    m2 = vqvae.decode([z_m2], start_level=2, bs_chunks=1).squeeze(-1)  # 1, 8192*128
    if not os.path.exists(f'{save_path}/'):
        os.makedirs(f'{save_path}/')
    torchaudio.save(f'{save_path}/mix.wav', mix.cpu().squeeze(-1), sample_rate=sample_rate)
    torchaudio.save(f'{save_path}/m1.wav', m1.cpu().squeeze(-1), sample_rate=sample_rate)
    torchaudio.save(f'{save_path}/m2.wav', m2.cpu().squeeze(-1), sample_rate=sample_rate)
    return mix, latent_mix, z_mixture, m1, m2, m1_real, m2_real


def make_models(vqvae_path, priors_list, sample_length, downs_t, sample_rate, commit,
                levels=3, level=2, fp16=True, device='cuda'):
    # construct openai vqvae and priors
    vqvae = make_vqvae(setup_hparams('vqvae', dict(sample_length=sample_length, downs_t=downs_t, sr=sample_rate,
                                                   commit=commit, restore_vqvae=vqvae_path)), device)
    prior_path_0 = priors_list[0]
    prior_path_1 = priors_list[1]

    prior_0 = make_prior(setup_hparams('small_prior', dict(levels=levels, level=level, labels=None,
                                                           restore_prior=prior_path_0, c_res=1, fp16_params=fp16,
                                                           n_ctx=8192)), vqvae, device)
    prior_1 = make_prior(setup_hparams('small_prior', dict(levels=levels, level=level, labels=None,
                                                           restore_prior=prior_path_1, c_res=1, fp16_params=fp16,
                                                           n_ctx=8192)), vqvae, device)
    priors = [prior_0, prior_1]
    return vqvae, priors

def separate(args):
    rank, local_rank, device = setup_dist_from_mpi(port=29531)

    args.sample_length = args.raw_to_tokens * args.l_bins #sample_tokens
    args.fp16 = True
    assert args.alpha[0] + args.alpha[1] == 1.

    vqvae, priors = make_models(args.restore_vqvae, args.restore_priors, args.sample_length, args.downs_t,
                                args.sample_rate, args.commit, levels=args.levels, level=args.level,
                                fp16=args.fp16, device=device)
    mix, latent_mix, z_mixture, m0, m1, m0_real, m1_real = create_mixture_from_audio_files(args.path_1, args.path_2,
                                                                                           args.raw_to_tokens, args.sample_tokens,
                                                                                           vqvae, args.save_path, args.sample_rate, args.alpha, shift=args.shift)
    n_ctx = min(priors[0].n_ctx, priors[1].n_ctx)
    hop_length = n_ctx // 2
    if not args.time_likelihood:
        if args.delta_likelihood:
            m = z_mixture
        else:
            m = latent_mix
    else:
        m = mix

    x_0, x_1, log_p_0_sum, log_p_1_sum, log_likelihood_sum = sample_level(vqvae, [priors[0].prior, priors[1].prior], m=m, n_ctx=n_ctx,
                                                                          hop_length=hop_length, alpha=args.alpha, n_samples=args.bs,
                                                                          sample_tokens=args.sample_tokens, sigma=args.sigma, context=args.context, fp16=args.fp16,
                                                                          bs_chunks=args.bs_chunks, window_mode=args.window_mode, l_bins=args.l_bins,
                                                                          raw_to_tokens=args.raw_to_tokens, device=device, chunk_size=args.chunk_size,
                                                                          latent_loss=not args.time_likelihood, top_k_posterior=args.top_k_posterior,
                                                                          delta_likelihood=args.delta_likelihood, sum_codebook=args.sum_codebook,
                                                                          emb_width=args.emb_width)

    res_0 = vqvae.decode([x_0], start_level=2, bs_chunks=1).squeeze(-1)  # n_samples, sample_tokens*128
    res_1 = vqvae.decode([x_1], start_level=2, bs_chunks=1).squeeze(-1)  # n_samples, sample_tokens*128

    # noinspection PyTypeChecker
    rejection_sampling(log_p_0_sum, log_p_1_sum, res_0, res_1, None, None, mix, args.alpha,
                       args.bs, rejection_sigma=None, n_samples=args.sample_tokens)

    evaluate_sdr_gt(m0, m1, res_0, res_1)
    evaluate_sdr_real(m0_real, m1_real, res_0, res_1)
    evaluate_l2_gt(m0, m1, res_0, res_1)
    save_samples(x_0, x_1, res_0, res_1, args.sample_rate, args.alpha, f'{args.save_path}/')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Bayesian inference with LQ-VAE')

    parser.add_argument('--path_1', type=str, help='Folder containing the first source',
                        default='data/drums/validation/Track01510.wav')
    parser.add_argument('--path_2', type=str, help='Folder containing the second source',
                    default='data/bass/validation/Track01510.wav')
    parser.add_argument('--shift', type=float, help='Where to start separating inside the song (in seconds)',
                        default=0.)
    parser.add_argument('--bs', type=int, help='Batch size', default=64)

    parser.add_argument('--sample_tokens', type=int, help='How many tokens to sample (344 tokens for ~1s). '
                                                          'Default: 1024 for ~ 3s.', default=1024)

    parser.add_argument('--sigma', type=float, help='Inference sigma', default=0.4)
    parser.add_argument('--alpha', type=float, nargs=2, help='Convex coefficients for the mixture',
                        default=[0.5, 0.5], metavar=('ALPHA_1', 'ALPHA_2'))
    parser.add_argument('--raw_to_tokens', type=int, help='Downsampling factor', default=64)
    parser.add_argument('--l_bins', type=int, help='Number of latent codes', default=2048)
    parser.add_argument('--levels', type=int, help='Number of levels', default=3)
    parser.add_argument('--level', type=int, help='VQ-VAE Level on which separation is performed', default=2)

    parser.add_argument('--sample_rate', type=int, help='Sample rate', default=22050)
    parser.add_argument('--bs_chunks', type=int, help='Batch size chunks', default=1)
    parser.add_argument('--chunk_size', type=int, help='Batch size chunk size', default=32)
    parser.add_argument('--window_mode', type=str, help='Local window type for time domain inference',
                        default='constant', choices=['constant', 'increment'])
    parser.add_argument('--downs_t', type=int, nargs='+', help='Downsampling factors', default=(2, 2, 2))
    parser.add_argument('--context', type=int, help='Time domain local context', default=50)
    parser.add_argument('--top_k_posterior', type=int, help="Filter top-k over the posterior (0 = don't filter)",
                        default=0)
    parser.add_argument('--time_likelihood', action='store_true',
                        help='Perform inference with likelihood function computed in time domain')
    parser.add_argument('--delta_likelihood', action='store_true',
                        help='Perform inference via delta functions instead of gaussians')
    parser.add_argument('--restore_priors', type=str, help='Time domain local context',
                        default=['checkpoints/checkpoint_drums_22050_latent_78_19k.pth.tar',
                                                              'checkpoints/checkpoint_latest.pth.tar'], nargs=2,
                        metavar=('PRIOR_1', 'PRIOR_2'))
    parser.add_argument('--restore_vqvae', type=str, help='Path to lq-vae checkpoint',
                        default='checkpoints/checkpoint_step_60001_latent.pth.tar')
    parser.add_argument('--sum_codebook', type=str, help='Pre-computed sum codebook path',
                        default='checkpoints/codebook_precalc_22050_latent.pt')
    parser.add_argument('--save_path', type=str, help='Folder containing the results', default='results')
    parser.add_argument('--commit', type=float, help='Commit scale', default=1.0)
    parser.add_argument('--emb_width', type=int, help='Embedding width', default=64)

    args = parser.parse_args()
    separate(args)
