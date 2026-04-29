import os
import sys
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from datetime import date

sys.path.insert(0, '/home/nora/GenDA/modulus')
sys.path.insert(0, '/home/nora/GenDA')

from src.dataloaders_enatl60 import eNATL60_Inference_Dataset
from src.sda import VPSDE, eps_edm
from modulus.models.diffusion import EDMPrecond

# ── Configuration ──────────────────────────────────────────────────────────
data_dir   = '/data2/nora/GenDA_workspace/input_data_enatl60/'
output_dir = '/data2/nora/GenDA_workspace/experiments/exp01_minitest_10kimg_lr=10000/'
ckpt_path  = output_dir + 'ema-state-diffusion-000010.mdlus'
n_members  = 4     # nombre de reconstructions indépendantes
n_steps    = 64    # pas du reverse process (256 en vrai, 64 pour le test)
device     = torch.device('cuda')

# ── Données ────────────────────────────────────────────────────────────────
with open(data_dir + 'diffusion_training_rescale_factors.json') as f:
    rescale_factors = json.load(f)

# Dataset de test : juin 2010
dataset = eNATL60_Inference_Dataset(
    data_dir   = data_dir,
    date_range = [date(2010,6,1), date(2010,6,30)],
    var_stds   = rescale_factors,
)

# ── Modèle ─────────────────────────────────────────────────────────────────
print('Chargement du checkpoint...')
from modulus import Module
net = Module.from_checkpoint(ckpt_path)
net = net.eval().to(device)
print('Checkpoint chargé.')

# Convertir en epsilon compatible VPSDE
eps = eps_edm(net, shape=(1, 128, 128))
sde = VPSDE(eps, shape=(1, 128, 128)).to(device)

# ── Inférence sur 3 jours de test ─────────────────────────────────────────
os.makedirs(output_dir + 'inference_plots/', exist_ok=True)

for t in range(3):
    print(f'\nJour {t+1}/3...')

    # Vérité terrain
    x_true = dataset[t].unsqueeze(0)  # (1, 1, 128, 128)

    # Génération inconditionnelle — n_members tirages indépendants
    with torch.no_grad():
        x_gen = sde.sample(
            shape   = (n_members,),
            steps   = n_steps,
        ).cpu().numpy()   # (n_members, 1, 128, 128)

    x_true_np = x_true[0, 0].numpy()  # (128, 128)

    # ── Visualisation ──────────────────────────────────────────────────
    fig, axs = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle(f'SSH — Jour de test {t+1} (en écarts-types)', fontsize=13)

    vmin, vmax = -3, 3

    # Vérité terrain
    im = axs[0, 0].pcolormesh(x_true_np, cmap='RdBu_r', vmin=vmin, vmax=vmax)
    axs[0, 0].set_title('Vérité terrain (eNATL60)')
    plt.colorbar(im, ax=axs[0, 0])

    # Moyenne ensemble
    axs[0, 1].pcolormesh(x_gen[:, 0].mean(axis=0), cmap='RdBu_r', vmin=vmin, vmax=vmax)
    axs[0, 1].set_title(f'Moyenne ensemble ({n_members} membres)')
    plt.colorbar(im, ax=axs[0, 1])

    # Écart-type ensemble
    axs[0, 2].pcolormesh(x_gen[:, 0].std(axis=0), cmap='Reds', vmin=0, vmax=2)
    axs[0, 2].set_title('Écart-type ensemble')
    plt.colorbar(im, ax=axs[0, 2])

    # 3 membres individuels
    for m in range(3):
        axs[1, m].pcolormesh(x_gen[m, 0], cmap='RdBu_r', vmin=vmin, vmax=vmax)
        axs[1, m].set_title(f'Membre {m+1}')
        plt.colorbar(im, ax=axs[1, m])

    plt.tight_layout()
    plot_path = output_dir + f'inference_plots/jour_{t+1:02d}.png'
    plt.savefig(plot_path, dpi=100)
    plt.close()
    print(f'  → Figure sauvegardée : {plot_path}')

    # Stats rapides
    print(f'  Vérité  — mean: {x_true_np.mean():.3f}, std: {x_true_np.std():.3f}')
    print(f'  Généré  — mean: {x_gen[:,0].mean():.3f}, std: {x_gen[:,0].std():.3f}')

print('\nInférence terminée.')
