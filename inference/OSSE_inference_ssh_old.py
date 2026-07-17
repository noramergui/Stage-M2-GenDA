import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '1')
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import cmocean
import datetime
from datetime import date
import sys
sys.path.append('/home/nora/GenDA/modulus')
sys.path.append('/home/nora/GenDA/')
sys.path.append('src')


import torch
import torch.nn.functional as F

from src.dataloaders import *
from modulus.distributed import DistributedManager
from modulus.utils.generative import parse_int_list
from modulus import Module
from src.sda import *
import json
from glob import glob

# ── Fichiers ────────────────────────────────────────────────────────────────
data_dir = '/data2/nora/GenDA_workspace/input_data_gulfstream/'

ds_masks_nad = xr.open_dataset(data_dir + 'mask_gulfstream_128x128_2017_nadir.nc')
ds_oi   = xr.open_dataset(data_dir + 'l4_oi.nc')

ds_m    = xr.open_dataset(data_dir + 'glorys_means_pre_processed_fixed_noislands.nc')

ds_masks_nad = xr.open_dataset(data_dir + 'mask_gulfstream_128x128_2017_nadir.nc')
ds_masks_swot = xr.open_dataset(data_dir + 'mask_gulfstream_128x128_2017_swot.nc')

# ── Gaussian blur (SSH seule, 1 canal) ───────────────────────────────────────
def multichannel_gaussian_blur(img, sigmas_rc):
    """Flou gaussien par canal. img: (B, C, H, W), sigmas_rc: liste de (sigma_row, sigma_col)."""
    device = img.device
    B, C, H, W = img.shape
    out = torch.zeros_like(img, device=device)
    for c in range(C):
        sigma_r, sigma_c = sigmas_rc[c]
        kernel_size_r = int(sigma_r * 3) * 2 + 1
        kernel_size_c = int(sigma_c * 3) * 2 + 1
        kernel_r = torch.exp(-torch.pow(torch.arange(kernel_size_r, device=device) - (kernel_size_r - 1) / 2, 2) / (2 * sigma_r**2))
        kernel_r = kernel_r / kernel_r.sum()
        kernel_c = torch.exp(-torch.pow(torch.arange(kernel_size_c, device=device) - (kernel_size_c - 1) / 2, 2) / (2 * sigma_c**2))
        kernel_c = kernel_c / kernel_c.sum()
        kernel_2d = torch.outer(kernel_r, kernel_c).unsqueeze(0).unsqueeze(0)
        out[:, c] = F.conv2d(img[:, c].unsqueeze(1), kernel_2d, padding='same')[:, 0]
    return out

# ── Facteurs de rescale ──────────────────────────────────────────────────────
with open(data_dir + 'diffusion_training_rescale_factors.json', 'r') as f:
    rescale_factors = json.load(f)
rescale = rescale_factors['zos']



# ── Sigmas de lissage SSH ────────────────────────────────────────────────────
sigma_L_ssh = 25       # km
NN_res = 1/12
deg_lon_in_km = 6378 * 2 * np.pi * np.cos(np.deg2rad(38)) / 360
deg_lat_in_km = 6378 * 2 * np.pi / 360
sigma_lon_ssh = (1 / NN_res) * sigma_L_ssh / deg_lon_in_km
sigma_lat_ssh = (1 / NN_res) * sigma_L_ssh / deg_lat_in_km

# ── Domaine ──────────────────────────────────────────────────────────────────
lon_min, lon_max = -65, -55
lat_min, lat_max = 33, 43
time_min, time_max = '2017-01-01', '2018-01-01'


NN_input_size = 128

buffer_lon = int((NN_input_size - abs(lon_max - lon_min) / NN_res) / 2)
print(f'buffer_lon = {buffer_lon}')
buffer_lat = int((NN_input_size - abs(lat_max - lat_min) / NN_res) / 2)
print(f'buffer_lat = {buffer_lat}')

lon_min_NN, lon_max_NN = lon_min - buffer_lon * NN_res, lon_max + buffer_lon * NN_res
lat_min_NN, lat_max_NN = lat_min - buffer_lat * NN_res, lat_max + buffer_lat * NN_res

#LON_GRID = np.linspace(float(lon_min_NN), float(lon_max_NN), NN_input_size)
#LAT_GRID = np.linspace(float(lat_min_NN), float(lat_max_NN), NN_input_size)

LON_GRID = lon_min_NN + np.arange(128) * NN_res
LAT_GRID = lat_min_NN + np.arange(128) * NN_res

ssh_mean_grid = ds_m['zos'].interp(latitude=LAT_GRID, longitude=LON_GRID).values
ssh_mean_grid = np.nan_to_num(ssh_mean_grid, nan=np.nanmean(ssh_mean_grid))

ds_oi = ds_oi.sel(longitude = slice(lon_min_NN, lon_max_NN - NN_res), latitude = slice(lat_min_NN, lat_max_NN - NN_res))
ds_oi = ds_oi.sel(time = slice(time_min, time_max))

ds_oi['ssh'] = (('time', 'latitude', 'longitude'),
                (ds_oi['ssh'].values - ssh_mean_grid[None, :, :]) / rescale_factors['zos'])

ds_oi = ds_oi.sel(time=slice(time_min, time_max))
ds_oi['ssh'][:, :, :8]  = np.nan
ds_oi['ssh'][:, :, -8:] = np.nan
ds_oi['ssh'][:, :8, :]  = np.nan
ds_oi['ssh'][:, -8:, :] = np.nan

print('loading dataset')


dataset = GenDA_OSSE_Inference_Dataset(
    '/data2/nora/GenDA_workspace/input_data_gulfstream/',
    lon_min=lon_min_NN, lon_max=lon_max_NN - NN_res,
    lat_min=lat_min_NN, lat_max=lat_max_NN - NN_res,
    input_dim=(128, 128),
    date_range=[date(2017, 1, 1), date(2017, 12, 31)],
    variables=['zos'],
    var_stds=rescale_factors,
    multiprocessing=False,
)

print(dataset.ds_model.sizes)              # time doit être > 0
print(dataset.ds_model['time'].values[:5]) # quelles dates existent réellement ?
print(dataset.ds_model['time'].values[-5:])

print('dataset loaded')

# ── Device / réseau ──────────────────────────────────────────────────────────
DistributedManager.initialize()
dist = DistributedManager()
device = dist.device

files = glob('/data2/nora/GenDA_workspace/trainings/gulfstream/ema*')
res_ckpt_filename = sorted(files)[-1]
force_fp16 = True

print(f'Loading residual network from "{res_ckpt_filename}"...')
net_res = Module.from_checkpoint(res_ckpt_filename)
net_res = net_res.eval().to(device).to(memory_format=torch.channels_last)
if force_fp16:
    net_res.use_fp16 = True

# ── Paramètres bruit synthétique (SSH) ───────────────────────────────────────
obs_noise_std = 0.03    # >>> écart-type fixe du bruit d'observation

mask_nadir_da = ds_masks_nad['mask']
mask_swot_da  = ds_masks_swot['mask']

n_members = 24
start_date = datetime.date(2017, 1, 1)
pred_dir = '/data2/nora/GenDA_workspace/osse_preds/'
os.makedirs(pred_dir, exist_ok=True)

eps = eps_edm(net_res, shape=())

ssh_mean = torch.from_numpy(ssh_mean_grid).float()  

# ── Estimation de oi_err_std ───────────
#print('estimating oi_err_std...')
#_err_accum = []
#for _t in range(365):
#    _, _xs = dataset.__getitem__(_t)
#    _ssh_truth = _xs[0].numpy() * rescale_factors['zos'] + dataset.ds_m['zos'].values  # mètres
#   _ssh_oi = ds_oi['ssh'].isel(time=_t).values                                        # mètres
#    _err_accum.append(_ssh_oi - _ssh_truth)
#_err_accum = np.stack(_err_accum)

#oi_err_std_phys = np.nanstd(_err_accum)                    # en mètres
#oi_err_std = oi_err_std_phys / rescale_factors['zos']      # en unités normalisées
#print(f'oi_err_std = {oi_err_std:.4f} (normalisé)  |  {oi_err_std_phys:.4f} m (physique)')

# ── Boucle temporelle ────────────────────────────────────────────────────────

TEST_DATES = ['2017-06-13']

for day_str in TEST_DATES:
    day = np.datetime64(day_str)
    print(f'\n=== {day_str} ===')

    # date -> indice pour la vérité (dataset + masque, qui commencent au 1er janvier)
    t = int(np.where(dataset.ds_model['time'].values == day)[0][0])

    x_star = dataset.__getitem__(t)          # indice (pas date !)
    x_star = x_star.unsqueeze(0)

    # masque
    m_nadir = mask_nadir_da.isel(time=t).values.astype(bool)
    m_swot  = mask_swot_da.isel(time=t).values.astype(bool)
    m_all   = m_nadir | m_swot

    total_mask = m_all[None]                     # (1,128,128)

    # OI : par DATE (commence au 6 janvier -> indice différent)
    oi_field = ds_oi['ssh'].sel(time=day).values
    oi_mask = (~np.isnan(oi_field))[None].astype('bool')
    oi_ground_truth = torch.from_numpy(oi_field[None, None])

    noise = torch.randn_like(x_star[0]) * obs_noise_std
    noise_levels = np.full(x_star[0].shape, obs_noise_std, dtype='float32')
    

    # ── Opérateur d'observation A ────────────────────────────────────────────
    def A(x):
        # 1. obs instantanées éparses (trace altimètre)
        inst_obs = x[:, total_mask]
        # 2. échelle physique
        ssh = x[:, 0:1].clone() * rescale_factors['zos'] + ssh_mean.to(x.device)
        # 3. lissage gaussien spatial
        smoothed_obs = multichannel_gaussian_blur(ssh, sigmas_rc=[(sigma_lat_ssh, sigma_lon_ssh)])
        # 4. re-normaliser puis masquer (là où l'OI existe)
        smoothed_obs[:, 0] = (smoothed_obs[:, 0] - ssh_mean.to(x.device)) / rescale_factors['zos']
        smoothed_obs = smoothed_obs[:, oi_mask]
        # 5. concaténer
        return torch.concat((inst_obs, smoothed_obs), axis=1)

    # observations instantanées + niveaux de bruit
    inst_obs = x_star[0, total_mask] + noise[total_mask]
    inst_obs = inst_obs.reshape(1, -1).repeat(n_members, 1)
    inst_obs_noise_level = torch.from_numpy(noise_levels[total_mask])
    inst_obs_noise_level = inst_obs_noise_level.reshape(1, -1).repeat(n_members, 1)

    # OI L4
    oi_gt = oi_ground_truth.repeat(n_members, 1, 1, 1)[:, oi_mask]
    oi_gt = torch.nan_to_num(oi_gt, 0)
    #oi_err = torch.full_like(oi_gt, oi_err_std)

    # ── y et std assemblés via A pour garantir le bon ordre/longueur ──────────
    y = A(torch.zeros((n_members, x_star.shape[1], 128, 128)))
    n_oi = oi_gt.shape[1]
    y[:, :-n_oi] = inst_obs
    y[:, -n_oi:] = oi_gt

    std = A(torch.zeros((n_members, x_star.shape[1], 128, 128)))
    std[:, :-n_oi] = inst_obs_noise_level
    #std[:, -n_oi:] = oi_err

    # ── Échantillonnage SDE ───────────────────────────────────────────────────
    sde = VPSDE(
        GaussianScore(
            y,
            A=A,
            std=std,
            sde=VPSDE(eps, shape=()),
            gamma=1e-1,
        ),
        shape=x_star.shape[1:],
    ).cuda()
    x = sde.sample((n_members,), steps=256, corrections=0, tau=0.3).cpu().numpy()

    np.save(pred_dir + f'pred{day_str}_07_07_swot.npy', x)



    # ── Plot (SSH uniquement) ─────────────────────────────────────────────────

    
    lon = dataset.ds_model['longitude']
    lat = dataset.ds_model['latitude']
    x_star_np = x_star[0, 0].detach().cpu().numpy()
    x_mean = np.mean(x[:, 0], axis=0)

    row_ssh = [
        ('SSH Observed',        (x_star_np + noise[0].cpu().numpy()) * total_mask[0]),
        ('SSH OI',              oi_field),
        ('SSH Ground Truth',    x_star_np),
        ('SSH Prediction Mean', x_mean),
        ('SSH 1st Member',      x[0, 0]),
    ]
    row_err = [
        ('SSH Prediction Std',  np.std(x[:, 0], axis=0)),
        ('SSH RMSE',            np.sqrt(np.mean((x[:, 0] - x_star_np)**2, axis=0))),
        ('SSH EnsMean RMSE',    np.sqrt((x_mean - x_star_np)**2)),
    ]

    fig, axs = plt.subplots(2, 5, figsize=(20, 10), constrained_layout=True)
    for ax, (title, field) in zip(axs[0], row_ssh):
        im_ssh = ax.pcolormesh(lon, lat, field, cmap='RdBu_r', vmin = -3, vmax = 3)
        ax.set_title(title)
    for ax, (title, field) in zip(axs[1], row_err):
        im_err = ax.pcolormesh(lon, lat, field, cmap=cmocean.cm.amp, vmin=0, vmax=1.5)
        ax.set_title(title)
    for ax in axs[1][len(row_err):]:
        fig.delaxes(ax)

    cb1 = fig.colorbar(im_ssh, ax=axs[0], ticks=[-3,-2,-1,0,1,2,3], location='right', shrink=0.8)
    cb1.set_label('SSH (standard deviations)', fontsize=13)
    cb2 = fig.colorbar(im_err, ax=axs[1], ticks=[0,0.5,1,1.5], location='right', shrink=0.8)
    cb2.set_label('RMSE / std (standard deviations)', fontsize=13)

    fig.suptitle(f'{day_str}', fontsize=15)
    print(f'  sauvegarde de {pred_dir}plot{day_str}_07_07_swot.png ...')
    plt.savefig(pred_dir + f'plot{day_str}_07_07_swot.png', dpi=110, bbox_inches='tight')
    print(f'  ✓ image sauvée')
    plt.close('all')