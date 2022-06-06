import os

import argparse
import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time

from src.ml_seq_preproc import generate_seq_ml1m
from src.vae import vae_utils
from src.vae.seqvae_runner import load_data, SeqTrainer
from src.vae.vae_models import RecSysVAE
from src.vae.vae_models.seqvae import SeqVAE
from src.vae.vae_utils import str2bool, CurvatureOptimizer
from src.batchrunner import report_metrics
from src.random import random_seeds, fix_torch_seed

# +
# in our experiments, we have used wandb framework to run experiments
# entity = ...
# project = ...
# import wandb
# wandb.init(entity=entity, project=project)
# -

parser = argparse.ArgumentParser()
parser.add_argument("--datapack", type=str, required=True, choices=["recvae", "urm"])
parser.add_argument("--dataname", type=str, required=True)  # depends on choice of data pack
parser.add_argument("--data_dir", type=str, default="./data/")
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--batch_size_eval", type=int, default=1)
parser.add_argument('--learning_rate', type=float, default=1e-3)
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument("--gamma", type=float, default=0.7)
parser.add_argument("--step_size", type=int, default=7)
parser.add_argument("--scheduler", default=True, action='store_true')
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--item_embed_size", type=int, default=128)
parser.add_argument("--rnn_size", type=int, default=100)
parser.add_argument("--embedding_dim", type=int, default=75)
parser.add_argument("--dropout", type=float, default=0.5)
parser.add_argument("--c", type=float, default=0.005)
parser.add_argument('--model', type=str, default="e200", help="Model latent space description.")
parser.add_argument("--fixed_curvature", type=str2bool, default=True,
                    help="Whether to fix curvatures to (-1, 0, 1).")
parser.add_argument('--total_anneal_steps', type=int, default=200000,
                    help='the total number of gradient updates for annealing')
parser.add_argument('--anneal_cap', type=float, default=0.2,
                    help='largest annealing parameter')
parser.add_argument("--show_progress", default=False, action='store_true')
parser.add_argument("--multilayer", default=False, action='store_true')
parser.add_argument("--unproc_data_dir", type=str, default=None)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

radius = 1.0 / np.sqrt(args.c)
if args.model[0] != "e":
    components = vae_utils.parse_components(args.model, args.fixed_curvature, radius=radius)
else:
    components = vae_utils.parse_components(args.model, args.fixed_curvature)
print(components)

# ##############INITIALIZATION###############

# data description
userid = "userid"
itemid = "itemid"
feedback = None

# randomization control
seeds = random_seeds(6, args.seed)
rand_seed_val, rand_seed_test = seeds[:2]
runner_seed_val, runner_seed_test = seeds[2:4]
sampler_seed_val, sampler_seed_test = seeds[4:]
fix_torch_seed(args.seed)

data_dir, data_pack, data_name = args.data_dir, args.datapack, args.dataname
data_base = data_dir + data_pack + '/%s/' % data_name

if not os.path.isdir(data_base):
    print('No processed dataset is found on %s, launching preprocessing...' % data_base)

    if args.unproc_data_dir is None:
        raise ValueError('Please provide path to unprocessed dataset in --unproc_data_dir')

    generate_seq_ml1m(args.unproc_data_dir, data_base, n_heldout_users=750, max_seq_len=1000, threshold=3.5)

train_loader, val_loader, test_loader, total_items = load_data(data_base=data_base,
                                                               batch_size=args.batch_size,
                                                               number_users_to_keep=1_000_000_000)
print("LOADED DATA")
###model

ddims = [args.embedding_dim, total_items]

if args.multilayer:
    ddims = [args.embedding_dim // 2, args.embedding_dim, total_items]

model = SeqVAE(
    decoder_dims=ddims,
    components=components,
    item_embed_size=args.item_embed_size,
    rnn_size=args.rnn_size,
    total_items=total_items,
    scalar_parametrization=False,
    dropout=args.dropout,
    batch_size=1
).to(device)

print("CREATED MODEL")
print(model)

update_count = 0
print("STARTING TRAINING")

trainer = SeqTrainer(model, total_anneal_steps=args.total_anneal_steps,
                     anneal_cap=args.anneal_cap, fixed_curvature=args.fixed_curvature)

optimizer = trainer.build_optimizer(learning_rate=args.learning_rate)

scheduler = None
if args.scheduler:
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.step_size, gamma=args.gamma
    )

try:
    for epoch in range(1, args.epochs + 1):
        epoch_mean_loss, update_count = trainer.train(optimizer=optimizer,
                                                      train_loader=train_loader,
                                                      update_count=update_count
                                                      )
        if scheduler:
            scheduler.step()

        scores, _ = trainer.evaluate(val_loader, train_cp_users=200, topk=[10, 20, 100], is_train_set=False)
        scores['loss'] = epoch_mean_loss
        report_metrics(scores, epoch)
#         wandb.log(scores)

except KeyboardInterrupt:
    print('-' * 102)
