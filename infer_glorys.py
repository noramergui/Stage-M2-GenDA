import os
import sys
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import date

sys.path.insert(0, '/home/nora/GenDA/modulus')
sys.path.insert(0, '/home/nora/GenDA')

os.environ['CUDA_VISIBLE_DEVICES'] = '1'    

from src.dataloaders import Diffusion_Training_Dataset
from src.sda import VPSDE, eps_edm
from modulus import Module

# ── Configuration ──────────────────────────────────────────────────────────
region     = 'azores'   # 'azores' ou 'gulfstream'
data_dir   = f'/data2/nora/GenDA_workspace/input_data_{region}/'
output_dir = f'/home/nora/GenDA/outputs/diffusion_test/output_diffusion_test/'
plot_dir   = output_dir + 'inference_plots/'
os.makedirs(plot_dir, exist_ok=True)

# Dernier checkpoint EMA disponible
import glob
ema_files = sorted(glob.glob(output_dir + 'ema-state-*.mdlus'))
ckpt_path = ema_files[-1]
print(f'Checkpoint : {ckpt_path}')

n_members = 4
n_steps   = 256
device    = torch.device('cuda')

# ── Données ────────────────────────────────────────────────────────────────
with open(data_dir + 'diffusion_training_rescale_factors.json') as f:
    rescale_factors = json.load(f)

# Dataset de test : décembre 2010
dataset = Diffusion_Training_Dataset(
    data_dir        = data_dir,
    n_lon           = 128,
    n_lat           = 128,
    variables       = ['zos'],
    date_range      = [date(2010,12,1), date(2010,12,31)],
    var_stds        = rescale_factors,
    lon_buffers     = [12, 12],
    lat_buffers     = [12, 18],
    multiprocessing = False,
    augment         = False,
)
print(f'Dataset test : {dataset.N_time} jours')

# ── Modèle ─────────────────────────────────────────────────────────────────
print('Chargement du checkpoint...')
net = Module.from_checkpoint(ckpt_path)
net = net.eval().to(device)
print('Modèle chargé.')

eps = eps_edm(net, shape=())
sde = VPSDE(eps, shape=(1, 128, 128)).to(device)

# ── Inférence sur 3 jours ─────────────────────────────────────────────────
for t in range(min(3, dataset.N_time)):
    print(f'\nJour {t+1}/3...')

    # Vérité terrain
    x_true    = dataset[t]           # (1, 64, 64)
    x_true_np = x_true[0].numpy()   # (64, 64)

    # Génération inconditionnelle
    with torch.no_grad():
        x_gen = sde.sample(
            shape = (n_members,),
            steps = n_steps,
            corrections = 0, 
            tau = 0.3
        ).cpu().numpy()  # (n_members, 1, 64, 64)

    # ── Visualisation ──────────────────────────────────────────────────────
    fig, axs = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle(f'SSH GLORYS {region} — Jour test {t+1} (écarts-types)', fontsize=13)
    vmin, vmax = -3, 3
    cmap = 'RdBu_r'

    # Vérité terrain
    im = axs[0,0].pcolormesh(x_true_np, cmap=cmap, vmin=vmin, vmax=vmax)
    axs[0,0].set_title('Vérité terrain (GLORYS)')
    plt.colorbar(im, ax=axs[0,0])

    # Moyenne ensemble
    axs[0,1].pcolormesh(x_gen[:,0].mean(axis=0), cmap=cmap, vmin=vmin, vmax=vmax)
    axs[0,1].set_title(f'Moyenne ensemble ({n_members} membres)')
    plt.colorbar(im, ax=axs[0,1])

    # Écart-type ensemble
    axs[0,2].pcolormesh(x_gen[:,0].std(axis=0), cmap='Reds', vmin=0, vmax=2)
    axs[0,2].set_title('Écart-type ensemble')
    plt.colorbar(im, ax=axs[0,2])

    # 3 membres individuels
    for m in range(3):
        axs[1,m].pcolormesh(x_gen[m,0], cmap=cmap, vmin=vmin, vmax=vmax)
        axs[1,m].set_title(f'Membre {m+1}')
        plt.colorbar(im, ax=axs[1,m])

    plt.tight_layout()
    path = plot_dir + f'jour_{t+1:02d}.png'
    plt.savefig(path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f'  → {path}')
    print(f'  Vérité — mean: {x_true_np.mean():.3f}, std: {x_true_np.std():.3f}')
    print(f'  Généré — mean: {x_gen[:,0].mean():.3f}, std: {x_gen[:,0].std():.3f}')

print('\nInférence terminée.')
