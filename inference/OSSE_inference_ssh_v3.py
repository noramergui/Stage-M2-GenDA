"""
OSSE_inference_ssh_v3.py — Assimilation GenDA avec 3 contraintes grande échelle :

  'l4'     : l4_oi.nc          (grande echelle initiale, ~verite coarse-grainee)
  'oi'     : OI.nc             (veritable OI nadir+SWOT, Lx=1 deg)
  'veille' : pred{J-1}_oi.npy  (moyenne d'ensemble de l'assimilation de la veille)

Pour chaque contrainte, le script :
  1. CALIBRE sigma : teste une liste de sigmas et garde celui qui minimise
     std(contrainte - blur_sigma(verite)) -> resolution effective mesuree
  2. estime err_std = std du residu avec ce sigma  (l'erreur prescrite a
     GaussianScore est ainsi coherente avec l'operateur A par construction)
  3. lance l'echantillonnage et sauve pred{day}_{nom}.npy + une figure

Les 3 runs partagent : le meme masque d'obs (nadir | swot), le meme bruit
d'obs (graine numpy fixee par jour), la meme graine torch pour les membres.
Ils ne different QUE par la contrainte grande echelle.

NB 'veille' : necessite pred{J-1}_oi.npy -> lancer le jour J-1 d'abord.
La contrainte veille n'est PAS lissee avant d'etre passee (A applique deja
le flou au candidat ; la lisser aussi ferait un double lissage).
"""

import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '1')
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import sys
sys.path.append('/home/nora/GenDA/modulus')
sys.path.append('/home/nora/GenDA/')
sys.path.append('src')

import json
import datetime
from datetime import date
from glob import glob

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cmocean
from scipy.ndimage import gaussian_filter

import torch
import torch.nn.functional as F

from src.dataloaders import *
from modulus.distributed import DistributedManager
from modulus.utils.generative import parse_int_list
from modulus import Module
from src.sda import *

import json, datetime

def make_run_tag(exp_name, sigma_km, err_std, inflate, oi_sub=1, gamma=1e-1, extra=''):
    tag = f'{exp_name}_s{sigma_km}km_e{err_std:.3f}_x{inflate}_sub{oi_sub}_g{gamma:g}'
    if extra:
        tag += f'_{extra}'
    return tag

def save_run(x, day_str, tag, params):
    """Sauve le .npy + un .json des parametres dans {day}/preds/. Refuse d'ecraser."""
    preds_dir, _, _ = run_dirs(day_str)
    base = os.path.join(preds_dir, f'pred{day_str}_{tag}')
    if os.path.exists(base + '.npy'):
        stamp = datetime.datetime.now().strftime('%m%d-%H%M')
        base += f'_{stamp}'
        print(f'  !! fichier existant -> suffixe horaire : {base}')
    np.save(base + '.npy', x)
    with open(base + '.json', 'w') as f:
        json.dump(params, f, indent=2, default=str)
    return base + '.npy'

# ════════════════════════════════════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════════════════════════════════════
data_dir = '/data2/nora/GenDA_workspace/input_data_gulfstream/'
pred_dir = '/data2/nora/GenDA_workspace/osse_preds/'

# pour la veille 
prev_path = '/data2/nora/GenDA_workspace/osse_preds/2017-06-12/preds/pred2017-06-12_l4_s25km_e0.234_x10.0_sub1_g0.1.npy'

os.makedirs(pred_dir, exist_ok=True)

def run_dirs(day_str):
    """Cree/renvoie les sous-dossiers du jour : preds/, figures/, diagnostics/."""
    preds = os.path.join(pred_dir, day_str, 'preds')
    figs = os.path.join(pred_dir, day_str, 'figures')
    diag = os.path.join(pred_dir, day_str, 'diagnostics')
    for p in (preds, figs, diag):
        os.makedirs(p, exist_ok=True)
    return preds, figs, diag

TEST_DATES = ['2017-06-12','2017-06-13']

# experiences a lancer
RUN_EXPERIMENTS = [
                    #'l4',
                    #'oi',
                    'veille'
                ]

# sigmas testes pour la calibration (km)
SIGMA_CANDIDATES = [5, 8, 10, 12, 15, 20, 25, 35, 50, 62, 75, 90, 110, 130]

n_members = 24
sde_steps = 256

nad_noise_phys = 0.05    # bruit obs nadir (m)
swot_noise_phys = 0.03   # bruit obs swot  (m)

OI_SUB = 1      # sous-echantillonnage de la contrainte
GAMMA = 0.1    # regularisation GaussianScore

ERR_INFLATE = 10.0

# ════════════════════════════════════════════════════════════════════════════
# GRILLE / CONSTANTES
# ════════════════════════════════════════════════════════════════════════════
lon_min, lon_max = -65, -55
lat_min, lat_max = 33, 43
NN_res = 1 / 12
NN_input_size = 128

buffer_lon = int((NN_input_size - abs(lon_max - lon_min) / NN_res) / 2)
buffer_lat = int((NN_input_size - abs(lat_max - lat_min) / NN_res) / 2)
lon_min_NN = lon_min - buffer_lon * NN_res
lat_min_NN = lat_min - buffer_lat * NN_res
lon_max_NN = lon_max + buffer_lon * NN_res
lat_max_NN = lat_max + buffer_lat * NN_res
LON_GRID = lon_min_NN + np.arange(128) * NN_res
LAT_GRID = lat_min_NN + np.arange(128) * NN_res

deg_lon_in_km = 6378 * 2 * np.pi * np.cos(np.deg2rad(38)) / 360
deg_lat_in_km = 6378 * 2 * np.pi / 360

def sigma_km_to_px(s_km):
    """sigma en km -> (sigma_lat, sigma_lon) en pixels de grille 1/12 deg."""
    return ((1 / NN_res) * s_km / deg_lat_in_km,
            (1 / NN_res) * s_km / deg_lon_in_km)

CROP = 8   # bords exclus des stats (coherent avec les NaN 8 px)

# ════════════════════════════════════════════════════════════════════════════
# CHARGEMENTS (une seule fois)
# ════════════════════════════════════════════════════════════════════════════
with open(data_dir + 'diffusion_training_rescale_factors.json') as f:
    rescale_factors = json.load(f)
rescale = rescale_factors['zos']

ds_m = xr.open_dataset(data_dir + 'glorys_means_pre_processed_fixed_noislands.nc')
ssh_mean_grid = ds_m['zos'].interp(latitude=LAT_GRID, longitude=LON_GRID).values
ssh_mean_grid = np.nan_to_num(ssh_mean_grid, nan=np.nanmean(ssh_mean_grid))
ssh_mean = torch.from_numpy(ssh_mean_grid).float()

mask_nadir_da = xr.open_dataset(data_dir + 'mask_gulfstream_128x128_2017_nadir.nc')['mask']
mask_swot_da = xr.open_dataset(data_dir + 'mask_gulfstream_128x128_2017_swot.nc')['mask']

print('loading dataset...')
dataset = GenDA_OSSE_Inference_Dataset(
    data_dir,
    lon_min=lon_min_NN, lon_max=lon_max_NN - NN_res,
    lat_min=lat_min_NN, lat_max=lat_max_NN - NN_res,
    input_dim=(128, 128),
    date_range=[date(2017, 1, 1), date(2017, 12, 31)],
    variables=['zos'],
    var_stds=rescale_factors,
    multiprocessing=False,
)
print('dataset loaded :', dataset.ds_model.sizes)

def _t_of(day):
    """datetime64 -> indice dans le dataset verite."""
    return int(np.where(dataset.ds_model['time'].values == day)[0][0])

def _truth(t):
    """Verite normalisee (128,128) en numpy.
    SEUL endroit qui suppose l'interface de __getitem__ : si chez toi il
    renvoie un tuple, corrige ICI uniquement."""
    return dataset.__getitem__(t)[0].numpy()

def _apply_nan_borders(field):
    f = field.copy()
    f[:CROP, :] = np.nan
    f[-CROP:, :] = np.nan
    f[:, :CROP] = np.nan
    f[:, -CROP:] = np.nan
    return f

def _load_constraint_nc(path, var='ssh', rename=None):
    """Charge un produit L4 netCDF, aligne sur la grille GenDA, normalise,
    NaN sur les bords. Renvoie un Dataset avec 'ssh' normalise."""
    ds = xr.open_dataset(path)
    if rename:
        first_var = list(rename)[0]
        ds = (ds[[first_var]].rename(rename)
                .drop_vars(['gtime', 'ng', 'nobs'], errors='ignore'))
    ds = ds.sel(longitude=LON_GRID, latitude=LAT_GRID,
                method='nearest', tolerance=NN_res / 2)
    ssh = (ds[var].values - ssh_mean_grid[None, :, :]) / rescale
    ssh[:, :CROP, :] = np.nan
    ssh[:, -CROP:, :] = np.nan
    ssh[:, :, :CROP] = np.nan
    ssh[:, :, -CROP:] = np.nan
    return xr.Dataset({'ssh': (('time', 'latitude', 'longitude'), ssh)},
                      coords={'time': ds['time'].values,
                              'latitude': LAT_GRID, 'longitude': LON_GRID})

# ── Reseau ──────────────────────────────────────────────────────────────────
DistributedManager.initialize()
dist = DistributedManager()
device = dist.device

files = glob('/data2/nora/GenDA_workspace/trainings/gulfstream/ema*')
res_ckpt_filename = sorted(files)[-1]
print(f'loading residual network from "{res_ckpt_filename}"...')
net_res = Module.from_checkpoint(res_ckpt_filename)
net_res = net_res.eval().to(device).to(memory_format=torch.channels_last)
net_res.use_fp16 = True
eps = eps_edm(net_res, shape=())

# ── Bruits d'obs normalises ─────────────────────────────────────────────────
nad_noise_std = nad_noise_phys / rescale
swot_noise_std = swot_noise_phys / rescale

# ── Contraintes L4 (chargees une fois) ──────────────────────────────────────
ds_l4 = _load_constraint_nc(data_dir + 'l4_oi.nc')
ds_oi = _load_constraint_nc(
    data_dir + 'OI.nc',
    rename={'gssh': 'ssh', 'lon': 'longitude', 'lat': 'latitude'})

# ════════════════════════════════════════════════════════════════════════════
# CALIBRATION : sigma + err_std pour une serie (temps, lat, lon) de contraintes
# ════════════════════════════════════════════════════════════════════════════
def calibrate(fields_by_day, label):
    """fields_by_day : dict {datetime64 -> champ 2D normalise (avec NaN bords)}.
    Renvoie (sigma_km, err_std_normalise)."""
    print(f'\n[{label}] calibration sigma :')
    days = list(fields_by_day)
    truths = {d: _truth(_t_of(d)) for d in days}

    best = (None, np.inf)
    for s in SIGMA_CANDIDATES:
        s_lat, s_lon = sigma_km_to_px(s)
        err = []
        for d in days:
            smooth_phys = gaussian_filter(truths[d] * rescale + ssh_mean_grid,
                              sigma=[s_lat, s_lon])
            smooth = (smooth_phys - ssh_mean_grid) / rescale
            err.append(fields_by_day[d] - smooth)
        e = float(np.nanstd(np.stack(err)[:, CROP:-CROP, CROP:-CROP]))
        flag = ''
        if e < best[1]:
            best = (s, e)
            flag = '  <-- min'
        print(f'    {s:>4} km -> {e:.4f}{flag}')

    sigma_km, err_std = best
    if sigma_km == SIGMA_CANDIDATES[-1]:
        print(f'    !! minimum au bord de la liste : etendre SIGMA_CANDIDATES')
    print(f'[{label}] sigma = {sigma_km} km | err_std = {err_std:.4f} (norm) '
          f'= {err_std * rescale:.4f} m')
    return sigma_km, err_std

# ════════════════════════════════════════════════════════════════════════════
# UNE ASSIMILATION
# ════════════════════════════════════════════════════════════════════════════

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

def run_assimilation(day_str, exp_name, constraint_field, sigma_km, err_std,
                     x_star, total_mask, noise, noise_levels):
    """constraint_field : champ 2D normalise (NaN bords). Sauve le .npy et la figure."""
    print(f'  -> run "{exp_name}"  (sigma={sigma_km} km, err_std={err_std:.4f})')

    tag = make_run_tag(exp_name, sigma_km, err_std, inflate=ERR_INFLATE, oi_sub=OI_SUB, gamma=GAMMA)

    s_lat, s_lon = sigma_km_to_px(sigma_km)
    oi_mask = (~np.isnan(constraint_field))[None].astype('bool')

    if OI_SUB > 1:
        sub = np.zeros_like(oi_mask)
        sub[:, ::OI_SUB, ::OI_SUB] = True
        oi_mask = oi_mask & sub

    oi_ground_truth = torch.from_numpy(
        constraint_field[None, None].astype('float32'))

    # ── Operateur d'observation A ───────────────────────────────────────────
    def A(x):
        inst_obs = x[:, total_mask]
        ssh = x[:, 0:1].clone() * rescale + ssh_mean.to(x.device)
        smoothed = multichannel_gaussian_blur(ssh, sigmas_rc=[(s_lat, s_lon)])
        smoothed[:, 0] = (smoothed[:, 0] - ssh_mean.to(x.device)) / rescale
        smoothed = smoothed[:, oi_mask]
        return torch.concat((inst_obs, smoothed), axis=1)

    # obs instantanees
    inst_obs = x_star[0, total_mask] + noise[total_mask]
    inst_obs = inst_obs.reshape(1, -1).repeat(n_members, 1)
    inst_lvl = torch.from_numpy(noise_levels[total_mask])
    inst_lvl = inst_lvl.reshape(1, -1).repeat(n_members, 1)

    # contrainte grande echelle
    oi_gt = oi_ground_truth.repeat(n_members, 1, 1, 1)[:, oi_mask]
    oi_gt = torch.nan_to_num(oi_gt, 0)
    oi_err = torch.full_like(oi_gt, err_std * ERR_INFLATE)

    # assemblage via A (garantit ordre et longueurs)
    y = A(torch.zeros((n_members, x_star.shape[1], 128, 128)))
    n_oi = oi_gt.shape[1]
    y[:, :-n_oi] = inst_obs
    y[:, -n_oi:] = oi_gt

    std = A(torch.zeros((n_members, x_star.shape[1], 128, 128)))
    std[:, :-n_oi] = inst_lvl
    std[:, -n_oi:] = oi_err

    # ── Echantillonnage ─────────────────────────────────────────────────────
    torch.manual_seed(0)   # membres comparables entre experiences
    sde = VPSDE(
        GaussianScore(y, A=A, std=std, sde=VPSDE(eps, shape=()), gamma=GAMMA),
        shape=x_star.shape[1:],
    ).cuda()
    x = sde.sample((n_members,), steps=sde_steps, corrections=0, tau=0.3)
    x = x.cpu().numpy()

    out_npy = save_run(x, day_str, tag, params={
        'exp': exp_name, 'day': day_str,
        'sigma_km': sigma_km, 'err_std': err_std,
        'oi_sub': OI_SUB, 'gamma': GAMMA, 'tau': 0.3,
        'corrections': 0, 'steps': sde_steps, 'n_members': n_members,
        'checkpoint': res_ckpt_filename,
        'constraint_file': 'l4_oi.nc' if exp_name == 'l4' else '...',
        'seed_np': 0, 'seed_torch': 0,
    })
    print(f'     sauve : {out_npy}')

    # ── Figure ──────────────────────────────────────────────────────────────
    lon = dataset.ds_model['longitude']
    lat = dataset.ds_model['latitude']
    x_star_np = x_star[0, 0].detach().cpu().numpy()
    x_mean = np.mean(x[:, 0], axis=0)

    row_ssh = [
        ('SSH Observed', (x_star_np + noise[0].cpu().numpy()) * total_mask[0]),
        (f'Constraint ({exp_name})', constraint_field),
        ('SSH Ground Truth', x_star_np),
        ('SSH Prediction Mean', x_mean),
        ('SSH 1st Member', x[0, 0]),
    ]
    row_err = [
        ('SSH Prediction Std', np.std(x[:, 0], axis=0)),
        ('SSH RMSE', np.sqrt(np.mean((x[:, 0] - x_star_np) ** 2, axis=0))),
        ('SSH EnsMean RMSE', np.sqrt((x_mean - x_star_np) ** 2)),
    ]

    fig, axs = plt.subplots(2, 5, figsize=(20, 10), constrained_layout=True)
    for ax, (title, field) in zip(axs[0], row_ssh):
        im_ssh = ax.pcolormesh(lon, lat, field, cmap='RdBu_r', vmin=-3, vmax=3)
        ax.set_title(title)
    for ax, (title, field) in zip(axs[1], row_err):
        im_err = ax.pcolormesh(lon, lat, field, cmap=cmocean.cm.amp,
                               vmin=0, vmax=1.5)
        ax.set_title(title)
    for ax in axs[1][len(row_err):]:
        fig.delaxes(ax)

    fig.colorbar(im_ssh, ax=axs[0], ticks=[-3, -2, -1, 0, 1, 2, 3],
                 location='right', shrink=0.8).set_label(
        'SSH (standard deviations)', fontsize=13)
    fig.colorbar(im_err, ax=axs[1], ticks=[0, 0.5, 1, 1.5],
                 location='right', shrink=0.8).set_label(
        'RMSE / std (standard deviations)', fontsize=13)
    fig.suptitle(f'{day_str} — {exp_name}', fontsize=15)

    _, figs_dir, _ = run_dirs(day_str)
    plt.savefig(os.path.join(figs_dir, f'pred{day_str}_{tag}.png'),
                dpi=110, bbox_inches='tight')
    plt.close('all')

# ════════════════════════════════════════════════════════════════════════════
# CALIBRATION DES CONTRAINTES L4 (sur tous les jours disponibles, une fois)
# ════════════════════════════════════════════════════════════════════════════
avail_truth = set(dataset.ds_model['time'].values)

l4_days = [d for d in ds_l4.time.values if d in avail_truth]
SIGMA_L4, ERR_L4 = calibrate(
    {d: ds_l4['ssh'].sel(time=d).values for d in l4_days}, 'l4')

oi_days = [d for d in ds_oi.time.values if d in avail_truth]
SIGMA_OI, ERR_OI = calibrate(
    {d: ds_oi['ssh'].sel(time=d).values for d in oi_days}, 'oi')

# ════════════════════════════════════════════════════════════════════════════
# BOUCLE PRINCIPALE
# ════════════════════════════════════════════════════════════════════════════
for day_str in TEST_DATES:
    day = np.datetime64(day_str)
    print(f'\n=== {day_str} ===')
    t = _t_of(day)

    # verite + masques + bruit d'obs : COMMUNS aux 3 experiences
    x_star = torch.from_numpy(_truth(t)[None]).unsqueeze(0)   # (1,1,128,128)

    m_nadir = mask_nadir_da.isel(time=t).values.astype(bool)
    m_swot = mask_swot_da.isel(time=t).values.astype(bool)
    m_all = m_nadir | m_swot

    noise_level_map = np.zeros((128, 128), dtype='float32')
    noise_level_map[m_nadir] = nad_noise_std
    noise_level_map[m_swot] = swot_noise_std
    total_mask = m_all[None]
    noise_levels = noise_level_map[None]

    np.random.seed(0)   # meme bruit d'obs pour les 3 experiences
    noise = torch.from_numpy(
        np.random.randn(128, 128).astype('float32') * noise_level_map)[None]

    # ── experience 'l4' ─────────────────────────────────────────────────────
    if 'l4' in RUN_EXPERIMENTS:
        field = ds_l4['ssh'].sel(time=day).values
        run_assimilation(day_str, 'l4', field, SIGMA_L4, ERR_L4,
                         x_star, total_mask, noise, noise_levels)

    # ── experience 'oi' ─────────────────────────────────────────────────────
    if 'oi' in RUN_EXPERIMENTS:
        field = ds_oi['ssh'].sel(time=day).values
        run_assimilation(day_str, 'oi', field, SIGMA_OI, ERR_OI,
                         x_star, total_mask, noise, noise_levels)

    # ── experience 'veille' ─────────────────────────────────────────────────
    if 'veille' in RUN_EXPERIMENTS:
        print(f'  veille <- {prev_path}')
        prev_mean = np.load(prev_path)[:, 0].mean(0)
        prev_mean = _apply_nan_borders(prev_mean)
        # calibration sur ce jour : persistance J-1 vs verite J
        s_v, e_v = calibrate({day: prev_mean}, f'veille({day_str})')
        run_assimilation(day_str, 'veille', prev_mean, s_v, e_v,
                            x_star, total_mask, noise, noise_levels)

print('\nTermine.')