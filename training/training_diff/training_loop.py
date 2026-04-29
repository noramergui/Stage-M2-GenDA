# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


#################
# NVIDIA Modulus CorrDiff code with minor adaptations from Scott Martin to apply to surface ocean state estimation
#################

"""Boucle principale d'entraînement GenDA — adaptée de NVIDIA Modulus CorrDiff"""

import copy
import json
import os
import sys
import time
import wandb as wb
import numpy as np
import psutil
import torch

from hydra.utils import to_absolute_path
from torch.nn.parallel import DistributedDataParallel
from . import training_stats

sys.path.append("../")
from modulus import Module
from modulus.distributed import DistributedManager
from modulus.launch.logging import PythonLogger, RankZeroLoggingWrapper, initialize_wandb
from modulus.utils.generative import construct_class_by_name, ddp_sync, format_time


def training_loop(
    dataset,                        # dataset d'entraînement
    dataset_iterator,               # itérateur sur le dataset d'entraînement (next() à chaque pas)
    validation_dataset,             # dataset de validation
    validation_dataset_iterator,    # itérateur sur le dataset de validation
    *,
    task,                           # 'diffusion' ou 'regression'
    run_dir=".",                    # dossier de sauvegarde des checkpoints
    network_kwargs={},              # architecture du réseau (SongUNet, EDMPrecond...)
    loss_kwargs={},                 # type de loss (EDMLoss...)
    optimizer_kwargs={},            # optimiseur (Adam, lr...)
    augment_kwargs=None,            # augmentation de données (None = désactivé)
    seed=0,                         # graine aléatoire pour la reproductibilité
    batch_size_global=512,          # taille du batch total (tous GPUs confondus)
    batch_size_gpu=None,            # taille du batch par GPU (None = pas de limite)
    total_kimg=200000,              # durée totale en kimg (1 kimg = 1000 images vues)
    ema_halflife_kimg=500,          # demi-vie de l'EMA en kimg
    ema_rampup_ratio=0.05,          # ramp-up de l'EMA au début de l'entraînement
    lr_rampup_kimg=1,           # durée du warmup du learning rate en kimg
    loss_scaling=1,                 # facteur multiplicatif sur la loss (utile en fp16)
    kimg_per_tick=50,               # fréquence des prints de progression
    state_dump_ticks=500,           # fréquence de sauvegarde des checkpoints
    cudnn_benchmark=True,           # optimisation cuDNN (False si taille fixe)
    wandb_mode="disabled",          # mode WandB pour le logging en ligne
    wandb_project="Modulus-Generative",
    wandb_entity="CorrDiff-DDP-Group",
    wandb_name="CorrDiff",
    wandb_group="CorrDiff-Group",
    fp_optimizations="fp32",        # précision : 'fp32', 'fp16', 'amp-fp16', 'amp-bf16'
    regression_checkpoint_path=None,
    grad_clip_threshold=None,       # seuil de clipping des gradients (None = désactivé)
    lr_decay=0.8,                   # décroissance du lr après le warmup
    valid_dump_ticks=5000,          # fréquence de calcul de la loss de validation
    num_validation_evals=10         # nombre de batches pour estimer la loss de validation
):

    # =========================================================================
    # SECTION 1 — INITIALISATION
    # =========================================================================

    # Gestionnaire multi-GPU
    dist = DistributedManager()
    device = dist.device

    # Logger : écrit dans le terminal ET dans un fichier log, horodaté
    # RankZeroLoggingWrapper : en multi-GPU, seul le GPU principal écrit
    logger = PythonLogger(name="training_loop")
    logger0 = RankZeroLoggingWrapper(logger, dist)
    logger.file_logging(file_name=f"logs/training_loop_{dist.rank}.log")

    # WandB : outil de suivi en ligne des métriques d'entraînement (désactivé par défaut)
    if dist.rank == 0:
        initialize_wandb(
            project=wandb_project, entity=wandb_entity, name=wandb_name,
            mode=wandb_mode, group=wandb_group, save_code=True,
        )
        wb.run.log_code(
            to_absolute_path("."),
            exclude_fn=lambda path: ("outputs" in path) and (os.getcwd() not in path),
        )

    # AMP (Automatic Mixed Precision) : utilise fp16 pour accélérer l'entraînement
    # amp-fp16 est plus rapide mais moins stable que fp32
    enable_amp = fp_optimizations.startswith("amp")
    amp_dtype = torch.float16 if (fp_optimizations == "amp-fp16") else torch.bfloat16

    # Graine aléatoire pour la reproductibilité
    # Chaque GPU a une graine différente pour éviter les corrélations
    start_time = time.time()
    np.random.seed((seed * dist.world_size + dist.rank) % (1 << 31))
    torch.manual_seed(np.random.randint(1 << 31))

    # Paramètres cuDNN
    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

    # =========================================================================
    # SECTION 2 — CALCUL DU BATCH PAR GPU ET GRADIENT ACCUMULATION
    # =========================================================================

    # batch_gpu_total : nombre d'images à traiter par GPU
    # Exemple : batch_size_global=64, 2 GPUs → batch_gpu_total=32
    batch_gpu_total = batch_size_global // dist.world_size
    logger0.info(f"batch_size_gpu: {batch_size_gpu}")

    if batch_size_gpu is None or batch_size_gpu > batch_gpu_total:
        batch_size_gpu = batch_gpu_total

    # num_accumulation_rounds : nombre de mini-batches avant une mise à jour des poids
    # Utile si le batch complet ne tient pas en mémoire GPU
    # Exemple : batch_gpu_total=64, batch_size_gpu=16 → 4 rounds d'accumulation
    num_accumulation_rounds = batch_gpu_total // batch_size_gpu
    if batch_size_global != batch_size_gpu * num_accumulation_rounds * dist.world_size:
        raise ValueError(
            "batch_size_global must be equal to batch_size_gpu * num_accumulation_rounds * dist.world_size"
        )

    # =========================================================================
    # SECTION 3 — CONFIGURATION DU RÉSEAU SELON LA TÂCHE
    # =========================================================================

    # Pour la diffusion : entrée = sortie = champ complet (SSH, SST, SSS...)
    # Pour la régression : entrée = observations masquées + OI, sortie = champ complet
    if task == 'diffusion':
        img_in_channels = len(dataset.variables)                        # nombre de variables (ex: 7)
        (img_shape_y, img_shape_x) = (dataset.n_lat, dataset.n_lon)     # taille des patches
        img_out_channels = len(dataset.variables)
        network_kwargs.img_resolution = img_shape_y                     # résolution spatiale (128)
        network_kwargs.img_channels = img_in_channels                   # nombre de canaux en entrée
    else:
        img_in_channels = len(dataset.variables_in) + len(dataset.variables_oi)
        (img_shape_y, img_shape_x) = (dataset.n_lat, dataset.n_lon)
        img_out_channels = len(dataset.variables_out)
        network_kwargs.img_resolution = img_shape_y
        network_kwargs.in_channels = img_in_channels
        network_kwargs.out_channels = img_out_channels

    # =========================================================================
    # SECTION 4 — CONSTRUCTION DU RÉSEAU, DE LA LOSS ET DE L'OPTIMISEUR
    # =========================================================================

    logger0.info("Constructing network...")
    net = construct_class_by_name(**network_kwargs)
    net.train()                 # mode entraînement : active le dropout
    net.requires_grad_(True)    # autorise le calcul des gradients
    net.to(device)              # déplace sur GPU

    logger0.info("Setting up optimizer...")

    # interface_kwargs : arguments supplémentaires pour la loss ResLoss (non utilisé en diffusion simple)
    if task == "diffusion":
        if loss_kwargs["class_name"] == "modulus.metrics.diffusion.ResLoss":
            interface_kwargs = dict(
                regression_net=net_reg, img_shape_x=img_shape_x, img_shape_y=img_shape_y,
                patch_shape_x=patch_shape_x, patch_shape_y=patch_shape_y,
                patch_num=patch_num, hr_mean_conditioning=hr_mean_conditioning,
            )
        else:
            interface_kwargs = {}
    else:
        interface_kwargs = {}

    # Crée une instance de EDMLoss — gère l'ajout du bruit et le calcul de la loss
    loss_fn = construct_class_by_name(**loss_kwargs, **interface_kwargs)

    # Crée l'optimiseur Adam avec les poids du réseau comme paramètres à optimiser
    # net.parameters() = liste de tous les poids que l'optimiseur va mettre à jour
    optimizer = construct_class_by_name(params=net.parameters(), **optimizer_kwargs)

    # Augmentation de données (non utilisée dans GenDA)
    augment_pipe = construct_class_by_name(**augment_kwargs) if augment_kwargs is not None else None

    # DDP (DistributedDataParallel) : répartit le calcul sur plusieurs GPUs
    # Avec 1 seul GPU, ddp = net directement
    if dist.world_size > 1:
        ddp = DistributedDataParallel(
            net, device_ids=[dist.local_rank], broadcast_buffers=True,
            output_device=dist.device, find_unused_parameters=dist.find_unused_parameters,
        )
    else:
        ddp = net

    # EMA : copie lissée des poids, utilisée à l'inférence
    # .eval() : désactive le dropout
    # .requires_grad_(False) : on ne calcule pas les gradients pour l'EMA
    ema = copy.deepcopy(net).eval().requires_grad_(False)

    # =========================================================================
    # SECTION 5 — REPRISE DEPUIS UN CHECKPOINT
    # =========================================================================

    # Cherche le checkpoint le plus récent dans run_dir
    # Les fichiers sont nommés : training-state-diffusion-000050.mdlus
    max_index = -1
    max_index_file = " "
    for filename in os.listdir(run_dir):
        if filename.startswith(f"training-state-{task}-") and filename.endswith(".mdlus"):
            index_str = filename.split("-")[-1].split(".")[0]
            try:
                index = int(index_str)
                if index > max_index:
                    max_index = index
                    max_index_file = filename
                    max_index_file_optimizer = f"optimizer-state-{task}-{index_str}.pt"
            except ValueError:
                continue

    try:
        # Charge les poids du réseau
        net.load(os.path.join(run_dir, max_index_file))
        # Charge l'état de l'optimiseur (mémoire interne d'Adam)
        map_location = {"cuda:%d" % 0: "cuda:%d" % int(dist.local_rank)}
        optimizer_state_dict = torch.load(
            os.path.join(run_dir, max_index_file_optimizer), map_location=map_location
        )
        optimizer.load_state_dict(optimizer_state_dict["optimizer_state_dict"])
        cur_nimg = max_index * 1000  # reprend le compteur d'images vues
        logger0.success(f"Loaded network and optimizer states with index {max_index}")
    except FileNotFoundError:
        cur_nimg = 0  # premier démarrage, pas de checkpoint
        logger0.warning("Could not load network and optimizer states")

    # =========================================================================
    # SECTION 6 — BOUCLE PRINCIPALE D'ENTRAÎNEMENT
    # =========================================================================

    logger0.info(f"Training for {total_kimg} kimg...")
    cur_tick = 0
    tick_start_nimg = cur_nimg
    tick_start_time = time.time()
    maintenance_time = tick_start_time - start_time
    stats_jsonl = None

    while True:

        # ----- 6a. Accumulation des gradients -----
        # zero_grad() : remet les gradients à zéro avant chaque pas
        # Sans ça, les gradients s'accumulent et les mises à jour sont incorrectes
        optimizer.zero_grad(set_to_none=True)
        loss_accum = 0

        for round_idx in range(num_accumulation_rounds):
            with ddp_sync(ddp, (round_idx == num_accumulation_rounds - 1)):

                if task == 'diffusion':
                    # Récupérer un batch de patches SSH propres
                    img_clean = next(dataset_iterator)
                    labels = None
                    img_clean = img_clean.to(device).to(torch.float32).contiguous()

                    # EDMLoss : ajoute du bruit aléatoire et calcule ||D(x+n) - x||²
                    with torch.autocast("cuda", dtype=amp_dtype, enabled=enable_amp):
                        loss = loss_fn(net=ddp, images=img_clean)
                else:
                    invar, outvar = next(dataset_iterator)
                    with torch.autocast("cuda", dtype=amp_dtype, enabled=enable_amp):
                        loss = loss_fn(
                            outvar.to(device).to(torch.float32).contiguous(),
                            net(invar.to(device).to(torch.float32).contiguous())
                        )

                training_stats.report("Loss/loss", loss)
                loss = loss.sum().mul(loss_scaling / batch_gpu_total)
                loss_accum += loss / num_accumulation_rounds

                # Calcul des gradients : remonte l'erreur à travers le réseau
                loss.backward()

        # Moyenne de la loss sur tous les GPUs (pour le logging)
        loss_sum = torch.tensor([loss_accum], device=device)
        if dist.world_size > 1:
            torch.distributed.all_reduce(loss_sum, op=torch.distributed.ReduceOp.SUM)
        average_loss = loss_sum / dist.world_size
        if dist.rank == 0:
            wb.log({"training loss": average_loss}, step=cur_nimg)

        # ----- 6b. Mise à jour du learning rate -----
        for g in optimizer.param_groups:
            # Warmup : lr monte progressivement de 0 à lr_max sur lr_rampup_kimg kimg
            # Évite les mises à jour trop brutales quand les poids sont encore aléatoires
            g["lr"] = optimizer_kwargs["lr"] * min(
                cur_nimg / max(lr_rampup_kimg * 1000, 1e-8), 1
            )
            # Décroissance : lr diminue lentement après le warmup
            # Permet des pas plus précis quand on approche du minimum
            g["lr"] *= lr_decay ** ((cur_nimg - lr_rampup_kimg * 1000) // 5e6)
            if dist.rank == 0:
                wb.log({"lr": g["lr"]}, step=cur_nimg)

        # Remplace les gradients NaN/inf par des valeurs sûres
        for param in net.parameters():
            if param.grad is not None:
                torch.nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad)

        # Clipping des gradients : évite les explosions de gradients
        if grad_clip_threshold:
            grad_norm = torch.nn.utils.clip_grad_norm_(net.parameters(), grad_clip_threshold)

        # ----- 6c. Mise à jour des poids -----
        # Adam utilise les gradients calculés par backward() pour mettre à jour les poids
        optimizer.step()

        # ----- 6d. Calcul de la loss de validation -----
        if validation_dataset_iterator is not None:
            valid_loss_accum = 0
            if cur_tick % valid_dump_ticks == 0:
                with torch.no_grad():  # pas de calcul de gradients en validation
                    for _ in range(num_validation_evals):
                        if task == 'diffusion':
                            img_clean_valid = next(validation_dataset_iterator)
                            img_clean_valid = img_clean_valid.to(device).to(torch.float32).contiguous()
                            loss_valid = loss_fn(net=ddp, images=img_clean_valid)
                        else:
                            invar, outvar = next(validation_dataset_iterator)
                            with torch.autocast("cuda", dtype=amp_dtype, enabled=enable_amp):
                                loss_valid = loss_fn(
                                    outvar.to(device).to(torch.float32).contiguous(),
                                    net(invar.to(device).to(torch.float32).contiguous())
                                )
                        training_stats.report("Loss/validation loss", loss_valid)
                        loss_valid = loss_valid.sum().mul(loss_scaling / batch_gpu_total)
                        valid_loss_accum += loss_valid / num_validation_evals

                    valid_loss_sum = torch.tensor([valid_loss_accum], device=device)
                    if dist.world_size > 1:
                        torch.distributed.all_reduce(valid_loss_sum, op=torch.distributed.ReduceOp.SUM)
                    average_valid_loss = valid_loss_sum / dist.world_size
                    if dist.rank == 0:
                        wb.log({"validation loss": average_valid_loss}, step=cur_nimg)

        # ----- 6e. Mise à jour de l'EMA -----
        # L'EMA est une moyenne glissante des poids — plus stable que les poids bruts
        # Utilisée à l'inférence pour de meilleures générations
        ema_halflife_nimg = ema_halflife_kimg * 1000
        if ema_rampup_ratio is not None:
            # Ramp-up : au début l'EMA suit les poids rapidement, puis ralentit
            ema_halflife_nimg = min(ema_halflife_nimg, cur_nimg * ema_rampup_ratio)
        # ema_beta proche de 1 → l'EMA change très lentement (effet de lissage)
        ema_beta = 0.5 ** (batch_size_global / max(ema_halflife_nimg, 1e-8))
        for p_ema, p_net in zip(ema.parameters(), net.parameters()):
            # lerp = interpolation linéaire :
            # p_ema = (1 - ema_beta) * p_net + ema_beta * p_ema
            # 99.9% ancienne valeur EMA + 0.1% nouveau poids
            p_ema.copy_(p_net.detach().lerp(p_ema, ema_beta))

        # ----- 6f. Vérification de fin et gestion des ticks -----
        cur_nimg += batch_size_global
        done = cur_nimg >= total_kimg * 1000 # condition d'arrêt
        if (not done) and (cur_nimg < tick_start_nimg + kimg_per_tick * 1000):
            continue  # pas encore la fin d'un tick → on continue sans print ni save

        # =========================================================================
        # SECTION 7 — LOGGING ET SAUVEGARDE (une fois par tick)
        # =========================================================================

        tick_end_time = time.time()

        # ----- 7a. Print de progression -----
        fields = []
        fields += [f"tick {training_stats.report0('Progress/tick', cur_tick):<5d}"]
        fields += [f"kimg {training_stats.report0('Progress/kimg', cur_nimg / 1e3):<9.1f}"]
        fields += [f"time {format_time(training_stats.report0('Timing/total_sec', tick_end_time - start_time)):<12s}"]
        fields += [f"sec/tick {training_stats.report0('Timing/sec_per_tick', tick_end_time - tick_start_time):<7.1f}"]
        fields += [f"sec/kimg {training_stats.report0('Timing/sec_per_kimg', (tick_end_time - tick_start_time) / (cur_nimg - tick_start_nimg) * 1e3):<7.2f}"]
        fields += [f"maintenance {training_stats.report0('Timing/maintenance_sec', maintenance_time):<6.1f}"]
        fields += [f"cpumem {training_stats.report0('Resources/cpu_mem_gb', psutil.Process(os.getpid()).memory_info().rss / 2**30):<6.2f}"]
        fields += [f"gpumem {training_stats.report0('Resources/peak_gpu_mem_gb', torch.cuda.max_memory_allocated(device) / 2**30):<6.2f}"]
        fields += [f"reserved {training_stats.report0('Resources/peak_gpu_mem_reserved_gb', torch.cuda.max_memory_reserved(device) / 2**30):<6.2f}"]
        torch.cuda.reset_peak_memory_stats()
        logger0.info(" ".join(fields))

        # ----- 7b. Sauvegarde des checkpoints -----
        if (state_dump_ticks is not None) and (done or cur_tick % state_dump_ticks == 0) and dist.rank == 0:

            # Poids du réseau (pour reprendre l'entraînement)
            # Nom formaté sur 6 chiffres : 000050 pour 50 kimg
            filename = f"training-state-{task}-{cur_nimg//1000:06d}.mdlus"
            net.save(os.path.join(run_dir, filename), verbose=True)
            logger0.info(f"Saved model in the {run_dir} directory")

            # Sauvegarde de l'EMA
            filename_ema = f"ema-state-diffusion-{cur_nimg//1000:06d}.mdlus"
            ema.save(os.path.join(run_dir, filename_ema))
            logger0.info(f"Saved EMA model in the {run_dir} directory")

            # État de l'optimiseur (mémoire interne d'Adam — nécessaire pour reprendre)
            filename = f"optimizer-state-{task}-{cur_nimg//1000:06d}.pt"
            torch.save({"optimizer_state_dict": optimizer.state_dict()}, os.path.join(run_dir, filename))
            logger0.info(f"Saved optimizer state in the {run_dir} directory")

        # ----- 7c. Sauvegarde des stats dans un fichier jsonl -----
        training_stats.default_collector.update()
        if dist.rank == 0:
            if stats_jsonl is None:
                stats_jsonl = open(os.path.join(run_dir, "stats.jsonl"), "at")
            stats_jsonl.write(json.dumps(dict(training_stats.default_collector.as_dict(), timestamp=time.time())) + "\n")
            stats_jsonl.flush()

        # ----- 7d. Mise à jour des compteurs de tick -----
        cur_tick += 1
        tick_start_nimg = cur_nimg
        tick_start_time = time.time()
        maintenance_time = tick_start_time - tick_end_time
        if done:
            break

    logger0.info("Exiting...")