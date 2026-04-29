import os
import sys
import json
import torch
import numpy as np
from datetime import date

sys.path.insert(0, '/home/nora/GenDA/modulus')
sys.path.insert(0, '/home/nora/GenDA')
sys.path.insert(0, '/home/nora/GenDA/training')

from torch.utils.data import DataLoader
from src.dataloaders_enatl60 import eNATL60_Diffusion_Training_Dataset
from modulus.distributed import DistributedManager
from modulus.launch.logging import PythonLogger, RankZeroLoggingWrapper
from modulus.utils.generative import EasyDict, InfiniteSampler
from training_diff import training_loop

# ── Configuration ──────────────────────────────────────────────────────────
data_dir   = '/data2/nora/GenDA_workspace/input_data_enatl60/'
output_dir = '/data2/nora/GenDA_workspace/experiments/exp01_minitest_10kimg_lr=10000/'
os.makedirs(output_dir, exist_ok=True)
os.makedirs('logs', exist_ok=True)

with open(data_dir + 'diffusion_training_rescale_factors.json') as f:
    rescale_factors = json.load(f)

# ── Distributed ────────────────────────────────────────────────────────────
DistributedManager.initialize()
dist = DistributedManager()

# logger = système qui écrit des messages pendant l'entraînement. 
# il horodate chaque message [11:41:52 - training_loop - INFO]
# écrit à la fois dans le terminal et dans un fichier log
logger  = PythonLogger(name='train_enatl60')
logger0 = RankZeroLoggingWrapper(logger, dist) # seul le GPU principal écrit les messages
logger.file_logging(file_name=f'logs/train_{dist.rank}.log')

# ── Datasets ───────────────────────────────────────────────────────────────

# training : juillet 2009 --> mars 2010
dataset_train = eNATL60_Diffusion_Training_Dataset(
    data_dir   = data_dir,
    n_lon      = 128,
    n_lat      = 128,
    date_range = [date(2009,7,1), date(2010,3,31)],
    var_stds   = rescale_factors,
    lon_buffers= [36, 36],
    lat_buffers= [36, 36],
)

# validation : avril 2010 --> mai 2010
dataset_val = eNATL60_Diffusion_Training_Dataset(
    data_dir   = data_dir,
    n_lon      = 128,
    n_lat      = 128,
    date_range = [date(2010,4,1), date(2010,5,31)],
    var_stds   = rescale_factors,
    lon_buffers= [36, 36],
    lat_buffers= [36, 36],
)

# InfiniteSampler : comme GenDA, échantillonnage infini sans epoch
# tire des index aléatoires sans jamais s'arrêter.
# adapté à l'entrainement diffusion qui ne raisonne pas en epochs mais en nombre d'images vues
sampler_train = InfiniteSampler(
    dataset=dataset_train,
    rank=dist.rank,                 # quel GPU on est 
    num_replicas=dist.world_size,   # nombre total de GPUs
    seed=0
)
sampler_val = InfiniteSampler(
    dataset=dataset_val,
    rank=dist.rank,
    num_replicas=dist.world_size,
    seed=0
)

# DataLoader = distributeur. prépare les batches dans l'ordre défini par le sampler. 
# iter() transforme le DataLoader en un objet sur lequel on peut appeler next().
# dans la boucle d'entrainement : 
# batch = next(dataset_iter) # donne le batch suivant
# batch = next(dataset_iter) # donne le batch suivant
# batch = next(dataset_iter) # ... 

dataset_iter = iter(DataLoader(dataset_train, sampler=sampler_train, batch_size=4))
valid_iter   = iter(DataLoader(dataset_val,   sampler=sampler_val,   batch_size=4))

# ── Config réseau — identique à GenDA ─────────────────────────────────────
# création d'un dictionnaire c qui est ensuite passé à training_loop

c = EasyDict()                      
c.task            = 'diffusion'
c.fp_optimizations = 'amp-fp16'
c.grad_clip_threshold = None
c.lr_decay        = 1.0

c.network_kwargs  = EasyDict()      # architecture du réseau (SongUNet, nombre de canaux...)

c.loss_kwargs     = EasyDict()      # type de loss (EDMLoss)

c.optimizer_kwargs = EasyDict(      # "utilise Adam avec ces réglages". Recette pour mettre à jour les poids du réseau
    
    class_name='torch.optim.Adam',  # nom de l'algorithme d'optimisation. Met à jour les poids du réseau à chaque pas d'entrainement. C'est le plus utilisé en deep learning. 
    
    lr=2e-4,                        # learning rate : taille du pas quand on met à jour les poids. 2e-4 = valeur standard pour la diffusion. trop grand : le modèle diverge, la loss explose. trop petit : le modèle apprend très lentement. 
    
    betas=[0.9, 0.999],             # paramètres internes d'Adam qui contrôlent la mémoire de l'optimiseur beta1=0.9 : mémoire sur le gradient récent / beta2=0.999 : mémoire sur la variance du gradient
    
    eps=1e-8                        # tout petit nombre ajouté pour éviter une division par zéro dans les calculs internes d'Adam
)

c.network_kwargs.update(            # architecture du réseau (SongUNet, nombre de canaux...)
    model_type     = 'SongUNet',

    model_channels = 64,            # nombre de canaux de base dans le UNet. Plus c'est grand, plus le réseau est puissant mais plus il est lent et gourmand en mémoire

    num_blocks     = 2,             # nombre de blocs de convolution à chaque niveau de résolution
    
    embedding_type = 'positional',  # comment le niveau de bruit t est encodé et injecté dans le réseau
    
    encoder_type   = 'standard',
    
    decoder_type   = 'standard',
    
    checkpoint_level = 0,
    
    dropout        = 0.13,          # éteint aléatoirement 13% des neurones pendant l'entrainement pour limiter le surapprentissage
    
    use_fp16       = False,
)
c.network_kwargs.class_name = 'modulus.models.diffusion.EDMPrecond' # précondtionnement EDM de Karras et al. 2022 qui enveloppe le SongUNet
c.loss_kwargs.class_name    = 'modulus.metrics.diffusion.EDMLoss'

# ── Hyperparamètres ────────────────────────────────────────────────────────
c.total_kimg        = 10        # mini-test : 10 kimg (~2500 pas avec batch=4)  # durée totale de l'entraînement en kimg
                                # vrai entraînement : 200                       
c.ema_halflife_kimg = 1         # mini-test : 1 kimg                            # demi vie de l'EMA (Exponential Moving Average). Copie lissée des poids du réseau utilisée à l'inférence. Concrètement : après ema_halflife_kimg kimg d'entraînement, l'influence d'un ancien poids sur l'EMA a été divisée par 2. Plus la valeur est grande, plus l'EMA est lisse mais réagit lentement aux nouvelles mises à jour. Pour GenDA : 500 kimg.
                                # vrai entraînement : 500
c.batch_size_global = 4         # mini-test                                     # taille du batch
                                # vrai entraînement : 64
c.batch_size_gpu    = 4         # Nombre de patches tratés par GPU à chaque pas. Si on a 2 GPUs et batch size global = 64, alors batch size gpu = 32

c.loss_scaling      = 1         # facteur multiplicatif appliqué à la loss avant le backward(). Utile en fp16 où les gradients peuvent être très petits et arrondir à zéro. Avec la valeur 1, ça ne change rien — c'est désactivé.

c.cudnn_benchmark   = False     #cuDNN est la librairie NVIDIA qui optimise les opérations GPU. Quand benchmark=True, cuDNN teste plusieurs algorithmes au démarrage et choisit le plus rapide pour ta configuration. On le met à False car nos patches ont toujours la même taille (128×128) — le benchmark ne servirait à rien et ralentirait le démarrage.

c.kimg_per_tick     = 1

c.state_dump_ticks  = 5         # sauvegarde tous les 5 ticks

c.valid_dump_ticks  = 2         # fréquence de calcul de la loss de validation  

c.num_validation_evals = 5      # nombre de batches pour estimer la loss de validation

c.lr_rampup_kimg = 10000            # durée du warmup du learning rate en kimg. Exemple lr_rampup_kimg=10000 : le warump dure 10kimg sur 200, soit 5% de l'entrainement, c'est raisonnable.
                                # vrai entraînement : 10000
c.run_dir           = output_dir

logger0.info('Démarrage entraînement GenDA/eNATL60')
logger0.info(f'GPU : {dist.device}')
logger0.info(f'total_kimg : {c.total_kimg}')
logger0.info(f'batch_size : {c.batch_size_global}')

# ── Lancement — même appel que train.py de GenDA ──────────────────────────
training_loop.training_loop(
    dataset_train, dataset_iter,
    dataset_val,   valid_iter,
    **c                         # **c = dépaquetage de dictionnaire. Passe chaque clé comme un argument nommé à la fonction. 
)
