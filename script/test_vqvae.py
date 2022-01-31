"""
RIQUANTIZZAZIONE VS SOMMA DELLE TRACCE NEL DOMINIO DEL TEMPO PRIMA DELL'ENCODING
"""



# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import os.path

import numpy as np
import torch as t
import torch
import torchaudio
import museval

from tqdm import tqdm
from jukebox.hparams import setup_hparams
from jukebox.make_models import make_vqvae, make_prior
from jukebox.transformer.ops import filter_logits
from jukebox.utils.dist_utils import setup_dist_from_mpi
from jukebox.utils.logger import def_tqdm
from jukebox.data.files_dataset import FilesAudioDataset
from torch.utils.data.distributed import DistributedSampler
from jukebox.utils.audio_utils import audio_preprocess, audio_postprocess
from torch.utils.data import DataLoader, Dataset, BatchSampler, RandomSampler
rank, local_rank, device = setup_dist_from_mpi(port=29524)
from jukebox.hparams import setup_hparams, Hyperparams

#%%
# define parameters
raw_to_tokens = 64
l_bins = 2048
sample_length = raw_to_tokens * l_bins
levels = 3
level = 2
sample_tokens = 10
n_samples = 1
sample_rate = 22050
fp16 = True
bs_chunks = 1
sigma = 0.03
min_duration = 5.95
#hps = setup_hparams("vqvae", {})
#print(hps)
labels = False
bass_audio_files_dir = '/media/michelemancusi/SSD Samsung/datasets/bass_22050/validation'
drums_audio_files_dir = '/media/michelemancusi/SSD Samsung/datasets/drums_22050/validation'
collate_fn = lambda batch: t.stack([t.from_numpy(b) for b in batch], 0)
levels = 3
channels = 1
#context = 8 # 8
aug_blend = True
restore_vqvae = '/home/michelemancusi/Scaricati/checkpoint_step_95001_latent_loss_reproduced.pth.tar'



hps = setup_hparams(
    "vqvae",
    dict(
        # vqvae hps
        restore_vqvae=restore_vqvae,
        # data hps
        #audio_files_dir=audio_files_dir,
        labels=False,
        aug_shift=True,
        aug_blend=True,
        sample_length=sample_length
    )
)

# hps_bass = setup_hparams(
#     "vqvae",
#     dict(
#         # vqvae hps
#         restore_vqvae=restore_vqvae,
#         # data hps
#         audio_files_dir=bass_audio_files_dir,
#         labels=False,
#         aug_shift=True,
#         aug_blend=True,
#         sample_length=sample_length
#     )
# )
#
# hps_drums = setup_hparams(
#     "vqvae",
#     dict(
#         # vqvae hps
#         restore_vqvae=restore_vqvae,
#         # data hps
#         audio_files_dir=drums_audio_files_dir,
#         labels=False,
#         aug_shift=True,
#         aug_blend=True,
#         sample_length=sample_length
#     )
# )



vqvae = make_vqvae(hps, device)

hps.audio_files_dir=bass_audio_files_dir

dataset_bass = FilesAudioDataset(hps)
hps.audio_files_dir=drums_audio_files_dir
dataset_drums = FilesAudioDataset(hps)

#vqvae = make_vqvae(setup_hparams('vqvae', dict(sample_length=sample_length, downs_t=(2, 2, 2), sr=22050,
#                                              restore_vqvae=restore_vqvae)), device)

#dataset_bass = FilesAudioDataset(setup_hparams('vqvae', dict(sample_length=sample_length, min_duration=min_duration,
#                                                             labels=labels, audio_files_dir=bass_audio_files_dir,
#                                                             downs_t=(2, 2, 2), sr=22050, restore_vqvae=restore_vqvae)))
#dataset_drums = FilesAudioDataset(setup_hparams('vqvae', dict(sample_length=sample_length, min_duration=min_duration,
#                                                              labels=labels, audio_files_dir=drums_audio_files_dir,
#                                                              downs_t=(2, 2, 2), sr=22050, restore_vqvae=restore_vqvae))
#                                  )

loader_bass = DataLoader(
    dataset_bass,
    batch_size=1,
    num_workers=8,
    pin_memory=False,
    shuffle=True,
    drop_last=True,
    collate_fn=lambda batch: torch.stack([torch.from_numpy(b) for b in batch], 0),
)

loader_drums = DataLoader(
    dataset_drums,
    batch_size=1,
    num_workers=8,
    pin_memory=False,
    shuffle=True,
    drop_last=True,
    collate_fn=lambda batch: torch.stack([torch.from_numpy(b) for b in batch], 0),
)

rnd1 = np.random.randint(1000)
rnd2 = np.random.randint(1000)



print('bass seed', rnd1)
print('drums seed', rnd2)

#loader_bass = DataLoader(dataset_bass, batch_size=1, num_workers=hps.nworkers, pin_memory=False, sampler=sampler_bass, drop_last=True, collate_fn=collate_fn)
#loader_drums = DataLoader(dataset_drums, batch_size=1, num_workers=hps.nworkers, pin_memory=False, sampler=sampler_drums, drop_last=True, collate_fn=collate_fn)
#m, _ = torchaudio.load('/home/michelemancusi/PycharmProjects/jukebox-langevin/data/vocals_44100/train/James May - If You Say/vocals.wav')print(m1.shape)

#z_mixture = vqvae.encode(m.unsqueeze(-1).cuda(), start_level=2, bs_chunks=1)[0]
#m = vqvae.decode([z_mixture], start_level=2, bs_chunks=1).squeeze(-1) # 1, 8192*128
#mixture_chunk = audio_preprocess(mixture_chunk.T.unsqueeze(0), aug_blend=False, channels=2)


#NUMBER_OF_PAIRS = len(test_loader2)
NUMBER_OF_PAIRS = 500

SILENCE_THRESHOLD = 1.5e-4
ELEMENT_CHECK = 7

save = True
it_num = 0
iter_counter = 0
l2_distances_list = []
l2_codes_distances_list = []
number_of_different_codes_list = []
mediam_sdr_list = []
#silence = t.zeros(1, int(262144), 1)
#sil_quant =vqvae.encode(silence.to('cuda'))
#for elem, elem2 in tqdm(zip(test_loader_bass, test_loader_drums), total=NUMBER_OF_PAIRS):

if len(loader_bass) < len(loader_drums):
    dataloader1 = loader_bass
    dataloader2 = loader_drums

else:
    dataloader1 = loader_drums
    dataloader2 = loader_bass

dataloader_iterator = iter(dataloader2)

for i, elem in enumerate(tqdm(dataloader1, total=NUMBER_OF_PAIRS)):
    elem2 = next(dataloader_iterator)
    if iter_counter == NUMBER_OF_PAIRS:
        break
    else:
        if torch.equal(elem, elem2):
            print("elements skipped")
            pass

        else:
            m1 = audio_preprocess(elem, hps)

            m2 = audio_preprocess(elem2, hps)
            #cost_z = vqvae.encode(cost.to('cuda'))
            zs1 = vqvae.encode(m1.to('cuda'))

            zs2 = vqvae.encode(m2.to('cuda'))


            xs1 = vqvae.bottleneck.decode(zs1)
            xs2 = vqvae.bottleneck.decode(zs2)
            #cost_x = vqvae.bottleneck.decode(cost_z)

            sum_at_latent_domain = [0.5*xs_1 + 0.5*xs_2 for xs_1, xs_2 in zip(xs1, xs2)]
            sum_quant = vqvae.bottleneck.encode(sum_at_latent_domain)
            sum_quant_dec = vqvae.bottleneck.decode(sum_quant)
            sum_quant_decoded = vqvae.decode([sum_quant[-1]], start_level=2, bs_chunks=1)

            sum_at_time_domain = 0.5*m1 + 0.5*m2
            encoded_sum_at_time_domain = vqvae.encode(sum_at_time_domain.to('cuda'))
            audio_sumdomain = vqvae.decode([encoded_sum_at_time_domain[-1]], start_level=2, bs_chunks=1)

            latent_sum = vqvae.bottleneck.decode(encoded_sum_at_time_domain)


            #diff = [xs_1 - xs_2 for xs_1, xs_2 in zip(latent_sum, sum_quant_dec)]
            #diff_quant = vqvae.bottleneck.encode(diff)
            #audio_3 = vqvae.decode([diff_quant[-1]], start_level=2, bs_chunks=1)

            if float(torch.sum(audio_sumdomain.squeeze(-1))/2048) <= SILENCE_THRESHOLD:
                pass
            else:
                number_of_different_codes = torch.sum(torch.eq(encoded_sum_at_time_domain[-1], sum_quant[-1]) == False)
                #a = vqvae.bottleneck.decode(encoded_summed_at_d)[-1]
                #b = vqvae.bottleneck.decode(sum_quant)[-1]

                l2_codes_distance = t.mean((latent_sum[-1] - sum_quant_dec[-1]) ** 2)
                l2_distance = torch.cdist(sum_quant_decoded.cpu().squeeze(-1), audio_sumdomain.cpu().squeeze(-1))
                number_of_different_codes_list.append(int(number_of_different_codes))
                l2_distances_list.append(float(l2_distance))
                l2_codes_distances_list.append(float(l2_codes_distance))


                sdr_metric = museval.evaluate(audio_sumdomain.cpu().numpy(), sum_quant_decoded.cpu().numpy())
                sdr_metric[0][sdr_metric[0] == np.inf] = np.nan

                #sdr_metric[0] = no_inf


                mediam_sdr_list.append(np.nanmedian(sdr_metric[0]))

                it_num += 1
                if save and iter_counter == ELEMENT_CHECK - 1:
                    torchaudio.save(f'results4/m1_{ELEMENT_CHECK}.wav',
                                    m1.cpu().squeeze(-1), sample_rate=sample_rate)
                    torchaudio.save(f'results4/m2_{ELEMENT_CHECK}.wav', m2.cpu().squeeze(-1),
                                    sample_rate=sample_rate)
                    torchaudio.save(f'results4/sum_at_timedom_element_{ELEMENT_CHECK}.wav', audio_sumdomain.cpu().squeeze(-1), sample_rate=sample_rate)
                    torchaudio.save(f'results4/sum_at_latent_element_{ELEMENT_CHECK}.wav', sum_quant_decoded.cpu().squeeze(-1), sample_rate=sample_rate)
                    #torchaudio.save(f'results4/diff_{ELEMENT_CHECK}.wav', audio_3.cpu().squeeze(-1), sample_rate=sample_rate)

                iter_counter += 1

    if i >= NUMBER_OF_PAIRS:
        break


print(f"Mean l2 distance: {sum(l2_distances_list)/it_num}")
print(f"Mean l2 codes distance: {sum(l2_codes_distances_list)/it_num}")
print(f"Mean different codes : {sum(number_of_different_codes_list)/it_num}")

with open('results4/number_of_different_codes.txt', 'w') as codes:
    with open('results4/l2_distances.txt', 'w') as distances:
        with open('results4/l2_codes_distances.txt', 'w') as codes_distances:
            with open('results4/sdr_median.txt', 'w') as sdr_file:
                for c, d, cd, sdr in zip(number_of_different_codes_list, l2_distances_list, l2_codes_distances_list, mediam_sdr_list):
                    codes.write("%s\n" % c)
                    distances.write("%s\n" % d)
                    codes_distances.write("%s\n" % d)
                    sdr_file.write("%s\n" % sdr)