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
data_dir   = '/home/nora/GenDA/input_data_enatl60/'
output_dir = '/home/nora/GenDA/outputs_enatl60/'
os.makedirs(output_dir, exist_ok=True)
os.makedirs('logs', exist_ok=True)

with open(data_dir + 'diffusion_training_rescale_factors.json') as f:
    rescale_factors = json.load(f)

# ── Distributed ────────────────────────────────────────────────────────────
DistributedManager.initialize()
dist = DistributedManager()

logger  = PythonLogger(name='train_enatl60')
logger0 = RankZeroLoggingWrapper(logger, dist)
logger.file_logging(file_name=f'logs/train_{dist.rank}.log')

# ── Datasets ───────────────────────────────────────────────────────────────
dataset_train = eNATL60_Diffusion_Training_Dataset(
    data_dir   = data_dir,
    n_lon      = 128,
    n_lat      = 128,
    date_range = [date(2009,7,1), date(2010,3,31)],
    var_stds   = rescale_factors,
    lon_buffers= [36, 36],
    lat_buffers= [36, 36],
)

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
sampler_train = InfiniteSampler(
    dataset=dataset_train,
    rank=dist.rank,
    num_replicas=dist.world_size,
    seed=0
)
sampler_val = InfiniteSampler(
    dataset=dataset_val,
    rank=dist.rank,
    num_replicas=dist.world_size,
    seed=0
)

dataset_iter = iter(DataLoader(dataset_train, sampler=sampler_train, batch_size=4))
valid_iter   = iter(DataLoader(dataset_val,   sampler=sampler_val,   batch_size=4))

# ── Config réseau — identique à GenDA ─────────────────────────────────────
c = EasyDict()
c.task            = 'diffusion'
c.fp_optimizations = 'amp-fp16'
c.grad_clip_threshold = None
c.lr_decay        = 1.0

c.network_kwargs  = EasyDict()
c.loss_kwargs     = EasyDict()
c.optimizer_kwargs = EasyDict(
    class_name='torch.optim.Adam',
    lr=2e-4,
    betas=[0.9, 0.999],
    eps=1e-8
)

c.network_kwargs.update(
    model_type     = 'SongUNet',
    model_channels = 64,
    num_blocks     = 2,
    embedding_type = 'positional',
    encoder_type   = 'standard',
    decoder_type   = 'standard',
    checkpoint_level = 0,
    dropout        = 0.13,
    use_fp16       = False,
)
c.network_kwargs.class_name = 'modulus.models.diffusion.EDMPrecond'
c.loss_kwargs.class_name    = 'modulus.metrics.diffusion.EDMLoss'

# ── Hyperparamètres ────────────────────────────────────────────────────────
c.total_kimg        = 10        # mini-test : 10 kimg (~2500 pas avec batch=4)
                                # vrai entraînement : 200
c.ema_halflife_kimg = 1         # mini-test : 1 kimg
                                # vrai entraînement : 500
c.batch_size_global = 4         # mini-test
                                # vrai entraînement : 64
c.batch_size_gpu    = 4
c.loss_scaling      = 1
c.cudnn_benchmark   = False
c.kimg_per_tick     = 1
c.state_dump_ticks  = 5         # sauvegarde tous les 5 ticks
c.valid_dump_ticks  = 2
c.num_validation_evals = 5
c.run_dir           = output_dir

logger0.info('Démarrage entraînement GenDA/eNATL60')
logger0.info(f'GPU : {dist.device}')
logger0.info(f'total_kimg : {c.total_kimg}')

# ── Lancement — même appel que train.py de GenDA ──────────────────────────
training_loop.training_loop(
    dataset_train, dataset_iter,
    dataset_val,   valid_iter,
    **c
)
