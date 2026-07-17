import os
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import cmocean
import datetime
import sys
sys.path.append('/home/nora/GenDA/modulus')
sys.path.append('/home/nora/GenDA')
sys.path.append('src')

from datetime import date
import torch
import torch.nn.functional as F

from src.dataloaders import *
from modulus import Module
from src.sda import *
import json
from glob import glob

os.environ['CUDA_VISIBLE_DEVICES'] = '1'


# ── Gaussian blur SSH seule ─────────────────────────────────────────────────
def gaussian_blur_ssh(field, sigma_r, sigma_c):
    """Lissage gaussien 2D d'un champ (B,1,128,128)."""
    device = field.device
    ks_r = int(sigma_r * 3) * 2 + 1
    ks_c = int(sigma_c * 3) * 2 + 1
    kr = torch.exp(-torch.pow(torch.arange(ks_r) - (ks_r-1)/2, 2)/(2*sigma_r**2)).to(device)
    kr = kr / kr.sum()
    kc = torch.exp(-torch.pow(torch.arange(ks_c) - (ks_c-1)/2, 2)/(2*sigma_c**2)).to(device)
    kc = kc / kc.sum()
    k2d = torch.outer(kr, kc).unsqueeze(0).unsqueeze(0)
    return F.conv2d(field, k2d, padding='same')


data_dir = '/data2/nora/GenDA_workspace/input_data_azores/'

with open(data_dir + 'diffusion_training_rescale_factors.json', 'r') as f:
    rescale_factors = json.load(f)
rescale = rescale_factors['zos']

# ── Domaine 128x128 (study box + buffer) ────────────────────────────────────
lon_min, lon_max = -33, -23
lat_min, lat_max = 28, 38
NN_res = 1/12
NN_input_size = 128
buffer_lon = int((NN_input_size - abs(lon_max - lon_min) / NN_res) / 2)
buffer_lat = int((NN_input_size - abs(lat_max - lat_min) / NN_res) / 2)
lon_min_NN, lon_max_NN = lon_min - buffer_lon * NN_res, lon_max + buffer_lon * NN_res
lat_min_NN, lat_max_NN = lat_min - buffer_lat * NN_res, lat_max + buffer_lat * NN_res
LON_GRID = lon_min_NN + np.arange(128) * NN_res
LAT_GRID = lat_min_NN + np.arange(128) * NN_res

# ── sigmas du lissage gaussien L4 (SSH) — CHANGÉ : étaient manquants ────────
sigma_L_ssh = 25  # km
deg_lat_in_km = 6378 * 2 * np.pi / 360
deg_lon_in_km = 6378 * 2 * np.pi * np.cos(np.deg2rad(33)) / 360
sigma_lat_ssh = (1 / NN_res) * sigma_L_ssh / deg_lat_in_km
sigma_lon_ssh = (1 / NN_res) * sigma_L_ssh / deg_lon_in_km

# ── Moyenne GLORYS sur la grille 128x128 ────────────────────────────────────
ds_m = xr.open_dataset(data_dir + 'glorys_means_pre_processed_fixed_noislands.nc')
ssh_mean_grid = ds_m['zos'].interp(latitude=LAT_GRID, longitude=LON_GRID).values 
ssh_mean_grid = np.nan_to_num(ssh_mean_grid, nan=np.nanmean(ssh_mean_grid))


# ── Fichiers L3 (obs), masque, L4 ───────────────────────────────────────────
# CHANGÉ : noms de variables réels (ssh / mask / ssh), L4 coarse interpolé
l3   = xr.open_dataset('/data2/nora/GenDA_workspace/obs_acores_128x128_2017.nc')   # ssh (t,128,128) mètres
mask = xr.open_dataset('/data2/nora/GenDA_workspace/mask_acores_128x128_2017.nc')  # mask (t,128,128) 0/1
l4   = xr.open_dataset('/data1/data/models/GLORYS/coarsened/GLORYS_2017_coarse-5_rolling-10.nc')  # ssh coarse mètres

# L3 normalisé (mètres -> espace modèle)
l3_norm = (l3['ssh'].values - ssh_mean_grid) / rescale

# L4 : découper la boîte, interpoler 128x128, normaliser
l4_box = l4['ssh'].sel(lon=slice(lon_min_NN-0.5, lon_max_NN+0.5),
                       lat=slice(lat_min_NN-0.5, lat_max_NN+0.5))
l4_norm = (l4_box.interp(lat=LAT_GRID, lon=LON_GRID) - ssh_mean_grid) / rescale

# ── Vérité : GLORYS_2017 
ref = xr.open_dataset('/data1/data/models/GLORYS/ref/GLORYS_2017.nc')
def get_truth(t):
    truth = ref['zos'].isel(time=t).interp(latitude=LAT_GRID, longitude=LON_GRID)
    return (truth.values - ssh_mean_grid) / rescale

# ── Erreurs d'obs (espace modèle) ───────────────────────────────────────────
l3_err = 0.1 / rescale      # sigma_noise VarDyn : 0.1 m
l4_err = 0.05 / rescale     # proxy L4

# ── Modèle ──────────────────────────────────────────────────────────────────
device = torch.device('cuda')
files = glob('/home/nora/GenDA/outputs/diffusion_job/output_diffusion/ema*')
res_ckpt_filename = sorted(files)[-1]
print(f'Checkpoint : {res_ckpt_filename}')
net_res = Module.from_checkpoint(res_ckpt_filename).eval().to(device)
eps = eps_edm(net_res, shape=())

# ── Sortie ──────────────────────────────────────────────────────────────────
pred_dir = '/data2/nora/GenDA_workspace/osse_preds_ssh/'
os.makedirs(pred_dir, exist_ok=True)

n_members = 16
gamma = 1e-1
ssh_mean_t = torch.from_numpy(ssh_mean_grid).float().to(device)

# jours de test (mettre range(365) pour toute l'année)
TEST_DAYS = [165, 68, 264]

for t in TEST_DAYS:
    day = date(2017, 1, 1).fromordinal(date(2017, 1, 1).toordinal() + t)
    print(f'\n=== Jour {t} ({day}) ===')

    x_true = get_truth(t)                                  # (128,128)

    # ── masque L3 (depuis le fichier mask) + valeurs obs ────────────────────
    mask_l3 = mask['mask'].isel(time=t).values.astype(bool)        # (128,128)
    l3_day  = l3_norm[t]    # sécurité : ne garder que les pixels du masque ET non-NaN
    mask_l3 = mask_l3 & np.isfinite(l3_day)

    # ── L4 (aligner le temps : commence le 6 janvier) ───────────────────────
    t4 = min(t, l4_norm.time.size - 1)
    l4_day  = l4_norm.isel(time=t4).values
    mask_l4 = np.isfinite(l4_day)

    mask_l3_t = torch.from_numpy(mask_l3).to(device)
    mask_l4_t = torch.from_numpy(mask_l4).to(device)

    # bords masqués (comme Scott Martin : bande de 8 px)
    for m in (mask_l3, mask_l4):
        m[:8, :] = False; m[-8:, :] = False
        m[:, :8] = False; m[:, -8:] = False

    # ── Opérateur A (SSH seule) ─────────────────────────────────────────────
    def A(x):
        # x : (n_members, 1, 128, 128)
        inst_obs = x[:, 0][:, mask_l3_t]                           # obs L3 éparses
        ssh_phys = x[:, 0:1] * rescale + ssh_mean_t                # -> mètres
        smoothed = gaussian_blur_ssh(ssh_phys, sigma_lat_ssh, sigma_lon_ssh)
        smoothed = (smoothed - ssh_mean_t) / rescale               # re-normaliser
        smooth_obs = smoothed[:, 0][:, mask_l4_t]                  # obs L4 lissées
        return torch.cat((inst_obs, smooth_obs), dim=1)

    # ── y et std (observations réelles) ─────────────────────────────────────
    y_l3 = torch.nan_to_num(torch.from_numpy(l3_day[mask_l3]).float(), 0).to(device)
    y_l4 = torch.nan_to_num(torch.from_numpy(l4_day[mask_l4]).float(), 0).to(device)
    y = torch.cat((y_l3, y_l4)).unsqueeze(0).repeat(n_members, 1)
    std = torch.cat((torch.full_like(y_l3, l3_err),
                     torch.full_like(y_l4, l4_err))).unsqueeze(0).repeat(n_members, 1)

    print(f'  L3 obs : {int(mask_l3.sum())} | L4 obs : {int(mask_l4.sum())}')

    # ── Assimilation ────────────────────────────────────────────────────────
    sde = VPSDE(
        GaussianScore(y, A=A, std=std, sde=VPSDE(eps, shape=()), gamma=gamma),
        shape=(1, 128, 128),
    ).cuda()
    x = sde.sample((n_members,), steps=256, corrections=0, tau=0.3).cpu().numpy()
    np.save(pred_dir + f'pred{t}.npy', x)

    x_mean = x[:, 0].mean(0)
    valid = np.isfinite(x_true)
    rmse = float(np.sqrt(np.nanmean((x_mean[valid] - x_true[valid])**2)))
    corr = float(np.corrcoef(x_mean[valid], x_true[valid])[0, 1])
    print(f'  RMSE {rmse:.3f} | corr {corr:.3f}')

    # ── Figure SSH seule ────────────────────────────────────────────────────
    ext = [LON_GRID[0], LON_GRID[-1], LAT_GRID[0], LAT_GRID[-1]]
    fig, axs = plt.subplots(2, 3, figsize=(15, 9))
    panels = [
        (x_true,                              'Vérité (GLORYS_2017)', 'RdBu_r', -3, 3),
        (np.where(mask_l3, l3_day, np.nan),   'Obs L3 (traces)',      'RdBu_r', -3, 3),
        (l4_day,                              'L4 (lissé)',           'RdBu_r', -3, 3),
        (x_mean,                              f'Reconstruction',      'RdBu_r', -3, 3),
        (x[:, 0].std(0),                      'Incertitude (std)',    'viridis', 0, 1),
        (x_mean - x_true,                     f'Erreur (RMSE {rmse:.2f})', 'RdBu_r', -1.5, 1.5),
    ]
    for ax, (fld, title, cm, vn, vx) in zip(axs.ravel(), panels):
        im = ax.imshow(fld, origin='lower', extent=ext, cmap=cm, vmin=vn, vmax=vx)
        ax.set_title(title); plt.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f'Assimilation SSH — {day} — corr {corr:.3f}')
    plt.tight_layout()
    plt.savefig(pred_dir + f'plot{t}.png', dpi=110, bbox_inches='tight')
    plt.close('all')
    print(f'  → {pred_dir}plot{t}.png')

print('\nTerminé.')