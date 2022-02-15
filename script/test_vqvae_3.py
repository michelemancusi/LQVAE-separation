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
import random
#import cupy as cp
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
audio_files_dir = '/media/michelemancusi/SSD Samsung/datasets/validation_slakh_22050'
collate_fn = lambda batch: t.stack([t.from_numpy(b) for b in batch], 0)
levels = 3
channels = 1
#context = 8 # 8
aug_blend = True
restore_vqvae = '/home/michelemancusi/Scaricati/checkpoint_step_85001_lq_vae_test.pth.tar'
bs = 2

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(100)




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





vqvae = make_vqvae(hps, device)

hps.audio_files_dir = audio_files_dir

dataset = FilesAudioDataset(hps)


loader = DataLoader(
    dataset,
    batch_size=bs,
    num_workers=8,
    pin_memory=False,
    shuffle=True,
    drop_last=True,
    collate_fn=lambda batch: torch.stack([torch.from_numpy(b) for b in batch], 0),
    worker_init_fn=seed_worker,
    generator=g
)

#rnd1 = 200 #np.random.randint(1000)
#rnd2 = 654 #np.random.randint(1000)



#print('bass seed', rnd1)
#print('drums seed', rnd2)

#loader_bass = DataLoader(dataset_bass, batch_size=1, num_workers=hps.nworkers, pin_memory=False, sampler=sampler_bass, drop_last=True, collate_fn=collate_fn)
#loader_drums = DataLoader(dataset_drums, batch_size=1, num_workers=hps.nworkers, pin_memory=False, sampler=sampler_drums, drop_last=True, collate_fn=collate_fn)
#m, _ = torchaudio.load('/home/michelemancusi/PycharmProjects/jukebox-langevin/data/vocals_44100/train/James May - If You Say/vocals.wav')print(m1.shape)

#z_mixture = vqvae.encode(m.unsqueeze(-1).cuda(), start_level=2, bs_chunks=1)[0]
#m = vqvae.decode([z_mixture], start_level=2, bs_chunks=1).squeeze(-1) # 1, 8192*128
#mixture_chunk = audio_preprocess(mixture_chunk.T.unsqueeze(0), aug_blend=False, channels=2)


#NUMBER_OF_PAIRS = len(test_loader2)
NUMBER_OF_PAIRS = 50

#SILENCE_THRESHOLD = 1.5e-4
#ELEMENT_CHECK = 7

save = True
it_num = 0
#iter_counter = 0
l2_distances_list = []
l2_codes_distances_list = []
number_of_different_codes_list = []
median_sdr_list = []
median_sdr_ori_vs_sum_latent_domain_list = []
median_sdr_ori_vs_sum_time_domain_list =[]
#silence = t.zeros(1, int(262144), 1)
#sil_quant =vqvae.encode(silence.to('cuda'))
#for elem, elem2 in tqdm(zip(test_loader_bass, test_loader_drums), total=NUMBER_OF_PAIRS):


#dataloader_iterator = iter(loader)

for i, x in enumerate(tqdm(loader, total=NUMBER_OF_PAIRS)):
    #elem2 = next(dataloader_iterator)
    #if iter_counter == NUMBER_OF_PAIRS:
    #    break
    #else:
        #if torch.equal(elem, elem2):
        #    print("elements skipped")
        #    pass

    #    else:
    x_in = audio_preprocess(x, hps)

    #m2 = audio_preprocess(elem2, hps)
    #cost_z = vqvae.encode(cost.to('cuda'))
    zs = vqvae.encode(x_in.to('cuda'))

    #zs2 = vqvae.encode(m2.to('cuda'))


    xs = vqvae.bottleneck.decode(zs)
    #xs2 = vqvae.bottleneck.decode(zs2)
    #cost_x = vqvae.bottleneck.decode(cost_z)

    sum_at_latent_domain = 1/2 * xs[-1][:xs[-1].shape[0]//2, :] + 1/2*xs[-1][xs[-1].shape[0]//2:, :]
    sum_quant = vqvae.bottleneck.one_level_encode(sum_at_latent_domain)
    sum_quant_dec = vqvae.bottleneck.one_level_decode(sum_quant)
    sum_quant_decoded = vqvae.decode([sum_quant], start_level=2, bs_chunks=1)

    sum_at_time_domain = 1/2.*x[:x.shape[0]//2, :] + 1/2.*x[x.shape[0]//2:, :]
    sum_at_time_domain = audio_preprocess(sum_at_time_domain, hps)
    encoded_sum_at_time_domain = vqvae.encode(sum_at_time_domain.to('cuda'))
    audio_sumdomain = vqvae.decode([encoded_sum_at_time_domain[-1]], start_level=2, bs_chunks=1)

    latent_sum = vqvae.bottleneck.decode(encoded_sum_at_time_domain)


    #diff = [xs_1 - xs_2 for xs_1, xs_2 in zip(latent_sum, sum_quant_dec)]
    #diff_quant = vqvae.bottleneck.encode(diff)
    #audio_3 = vqvae.decode([diff_quant[-1]], start_level=2, bs_chunks=1)

    #if float(torch.sum(audio_sumdomain.squeeze(-1),-1)/2048) <= SILENCE_THRESHOLD:
        #print("silent track")
    #    pass
    #else:
    number_of_different_codes = torch.sum(torch.eq(encoded_sum_at_time_domain[-1], sum_quant) == False)
    #a = vqvae.bottleneck.decode(encoded_summed_at_d)[-1]
    #b = vqvae.bottleneck.decode(sum_quant)[-1]

    l2_codes_distance = t.mean((latent_sum[-1] - sum_quant_dec) ** 2)
    #l2_distance = torch.cdist(sum_quant_decoded.cpu().squeeze(-1), audio_sumdomain.cpu().squeeze(-1))
    l2_distance = t.mean((sum_quant_decoded - audio_sumdomain) ** 2)
    number_of_different_codes_list.append(int(number_of_different_codes))
    l2_distances_list.append(float(l2_distance))
    l2_codes_distances_list.append(float(l2_codes_distance))


    try:
        sdr_metric = museval.evaluate(audio_sumdomain.cpu().numpy(), sum_quant_decoded.cpu().numpy(), win=22050, hop=22050)
        sdr_metric[0][sdr_metric[0] == np.inf] = np.nan
        a = np.nanmedian(sdr_metric[0], axis=-1)
        median_sdr_list += a.tolist()

    except ValueError:
        print("non valid sdr")

    try:
        sdr_metric = museval.evaluate(audio_sumdomain.cpu().numpy(), sum_at_time_domain.cpu().numpy(), win=22050, hop=22050)
        sdr_metric[0][sdr_metric[0] == np.inf] = np.nan
        b = np.nanmedian(sdr_metric[0], axis=-1)
        median_sdr_ori_vs_sum_latent_domain_list += b.tolist()

    except ValueError:
        print("non valid sdr")

    try:
        sdr_metric = museval.evaluate(sum_at_time_domain.cpu().numpy(), sum_quant_decoded.cpu().numpy(), win=22050, hop=22050)
        sdr_metric[0][sdr_metric[0] == np.inf] = np.nan
        c = np.nanmedian(sdr_metric[0], axis=-1)
        median_sdr_ori_vs_sum_time_domain_list += c.tolist()

    except ValueError:
        print("non valid sdr")



    #it_num += 1
    #if save and iter_counter == ELEMENT_CHECK - 1:
    #    torchaudio.save(f'results4/m1_{ELEMENT_CHECK}.wav',
    #                    m1.cpu().squeeze(-1), sample_rate=sample_rate)
    #    torchaudio.save(f'results4/m2_{ELEMENT_CHECK}.wav', m2.cpu().squeeze(-1),
    #                    sample_rate=sample_rate)
    #    torchaudio.save(f'results4/sum_at_timedom_element_{ELEMENT_CHECK}.wav', audio_sumdomain.cpu().squeeze(-1), sample_rate=sample_rate)
    #    torchaudio.save(f'results4/sum_at_latent_element_{ELEMENT_CHECK}.wav', sum_quant_decoded.cpu().squeeze(-1), sample_rate=sample_rate)
        #torchaudio.save(f'results4/diff_{ELEMENT_CHECK}.wav', audio_3.cpu().squeeze(-1), sample_rate=sample_rate)

    #iter_counter += 1

    if i >= NUMBER_OF_PAIRS:
        break



print(f"Mean l2 distance: {sum(l2_distances_list)/len(l2_distances_list)}")
print(f"Mean l2 codes distance: {sum(l2_codes_distances_list)/len(l2_codes_distances_list)}")
print(f"Mean different codes : {sum(number_of_different_codes_list)/len(number_of_different_codes_list)}")
print(f"Mean median sdr : {np.nanmean(median_sdr_list)}")
print(f"Median sdr ori vs sum_latent_domain : {np.nanmean(median_sdr_ori_vs_sum_latent_domain_list)}")
print(f"Median sdr ori vs sum_time_domain : {np.nanmean(median_sdr_ori_vs_sum_time_domain_list)}")

with open('results4/models_info.txt', "a+") as file_object:
    file_object.seek(0)
    # If file is not empty then append '\n'
    data = file_object.read(100)
    if len(data) > 0 :
        file_object.write("\n\n")
    file_object.write(f"restore_vqvae:{restore_vqvae}\n")
    file_object.write(f"NUMBER_OF_PAIRS:{str(NUMBER_OF_PAIRS)}\n")
    file_object.write(f"Mean l2 distance: {sum(l2_distances_list)/len(l2_distances_list)}\n")
    file_object.write(f"Mean l2 codes distance: {sum(l2_codes_distances_list)/len(l2_codes_distances_list)}\n")
    file_object.write(f"Mean different codes : {sum(number_of_different_codes_list)/len(number_of_different_codes_list)}\n")
    file_object.write(f"Mean median sdr : {np.nanmean(median_sdr_list)}\n")
    file_object.write(f"Median sdr ori vs sum_latent_domain : {np.nanmean(median_sdr_ori_vs_sum_latent_domain_list)}\n")
    file_object.write(f"Median sdr ori vs sum_time_domain : {np.nanmean(median_sdr_ori_vs_sum_time_domain_list)}\n")


with open('results4/number_of_different_codes.txt', 'w') as codes:
    with open('results4/l2_distances.txt', 'w') as distances:
        with open('results4/l2_codes_distances.txt', 'w') as codes_distances:
            with open('results4/sdr_median.txt', 'w') as sdr_file:
                for c, d, cd, sdr in zip(number_of_different_codes_list, l2_distances_list, l2_codes_distances_list, median_sdr_list):
                    codes.write("%s\n" % c)
                    distances.write("%s\n" % d)
                    codes_distances.write("%s\n" % d)
                    sdr_file.write("%s\n" % sdr)
#%%
