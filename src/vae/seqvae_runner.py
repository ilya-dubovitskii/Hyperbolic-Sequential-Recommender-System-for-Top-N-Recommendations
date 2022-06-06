from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from scipy.sparse import vstack
from torch.autograd import Variable
from tqdm import tqdm
import numpy as np
import os
import sys
import importlib.util

from .vae_utils import CurvatureOptimizer
from ..recvae import validate, get_data, data_folder_path
from ..datasets import observations_loader, UserBatchDataset
from ..datareader import read_data

if torch.cuda.is_available():
    print("Using CUDA...\n")
    LongTensor = torch.cuda.LongTensor
    FloatTensor = torch.cuda.FloatTensor
    device = torch.device('cuda')
else:
    LongTensor = torch.LongTensor
    FloatTensor = torch.FloatTensor
    device = torch.device('cpu')

def load_data(data_base, batch_size, number_users_to_keep):
    f = open(data_base + 'train.csv')
    lines_train = f.readlines()[1:]

    f = open(data_base + 'validation_tr.csv')
    lines_val_tr = f.readlines()[1:]

    f = open(data_base + 'validation_te.csv')
    lines_val_te = f.readlines()[1:]

    f = open(data_base + 'test_tr.csv')
    lines_test_tr = f.readlines()[1:]

    f = open(data_base + 'test_te.csv')
    lines_test_te = f.readlines()[1:]

    unique_sid = list()
    with open(data_base + 'unique_sid.txt', 'r') as f:
        for line in f:
            unique_sid.append(line.strip())
    num_items = len(unique_sid)

    train_reader = DataReader(lines_train, None, num_items, batch_size, number_users_to_keep, True)
    val_reader = DataReader(lines_val_tr, lines_val_te, num_items, batch_size, number_users_to_keep, False)
    test_reader = DataReader(lines_test_tr, lines_test_te, num_items, batch_size, number_users_to_keep, False)

    return train_reader, val_reader, test_reader, num_items


class DataReader:
    def __init__(self, a, b, num_items, batch_size, number_users_to_keep, is_training):
        self.number_users_to_keep = number_users_to_keep
        self.batch_size = batch_size

        num_users = 0
        min_user = 1000000000000000000000000  # Infinity
        unique_users = set()
        for line in a:
            line = line.strip().split(",")
            unique_users.add(int(line[0]))
        #             num_users = max(num_users, int(line[0]))
        #             min_user = min(min_user, int(line[0]))

        #         num_users = num_users - min_user + 1

        self.num_users = len(unique_users)
        self.id2idx = dict(zip(unique_users, range(self.num_users)))
        self.min_user = min_user
        self.num_items = num_items

        self.data_train = a
        self.data_test = b
        self.is_training = is_training
        self.all_users = []

        self.prep()
        self.number()

    def prep(self):
        print(f'num_users:{self.num_users}, len data_train:{len(self.data_train)}')
        self.data = []
        for i in range(self.num_users): self.data.append([])

        for i in tqdm(range(len(self.data_train))):
            line = self.data_train[i]
            line = line.strip().split(",")
            self.data[self.id2idx[int(line[0])]].append([int(line[1]), 1])

        if self.is_training == False:
            self.data_te = []
            for i in range(self.num_users): self.data_te.append([])

            for i in tqdm(range(len(self.data_test))):
                line = self.data_test[i]
                line = line.strip().split(",")
                self.data_te[self.id2idx[int(line[0])]].append([int(line[1]), 1])

    def number(self):
        self.num_b = int(min(len(self.data), self.number_users_to_keep) / self.batch_size)

    def iter(self):
        users_done = 0

        x_batch = []

        user_iterate_order = list(range(len(self.data)))

        # Randomly shuffle the training order
        np.random.shuffle(user_iterate_order)

        for user in user_iterate_order:

            if users_done > self.number_users_to_keep: break
            users_done += 1

            # TODO leave len(self.data[user]) - 1
            y_batch_s = torch.zeros(self.batch_size, len(self.data[user]) - 1, self.num_items)
            y_batch_s = y_batch_s.to(device)

            for timestep in range(len(self.data[user]) - 1):
                y_batch_s[len(x_batch), timestep, :].scatter_(
                    0, LongTensor([i[0] for i in [self.data[user][timestep + 1]]]), 1.0
                )

            x_batch.append([i[0] for i in self.data[user][:-1]])

            if len(x_batch) == self.batch_size:  # batch_size always = 1
                # print('User: ', user, 'x_batch.shape: ', list(x_batch.shape), '; y_batch_s.shape: ', list(y_batch_s.shape))
                yield Variable(LongTensor(x_batch)), Variable(y_batch_s, requires_grad=False)
                x_batch = []

    def iter_eval(self):

        x_batch = []
        test_movies, test_movies_r = [], []

        users_done = 0

        for user in range(len(self.data)):

            users_done += 1
            if users_done > self.number_users_to_keep: break

            if self.is_training == True:
                split = 0.8
                base_predictions_on = self.data[user][:int(split * len(self.data[user]))]
                heldout_movies = self.data[user][int(split * len(self.data[user])):]
            else:
                base_predictions_on = self.data[user]
                heldout_movies = self.data_te[user]

            y_batch_s = torch.zeros(self.batch_size, len(base_predictions_on) - 1, self.num_items).to(device)

            for timestep in range(len(base_predictions_on) - 1):
                y_batch_s[len(x_batch), timestep, :].scatter_(
                    0, LongTensor([i[0] for i in [base_predictions_on[timestep + 1]]]), 1.0
                )

            test_movies.append([i[0] for i in heldout_movies])
            test_movies_r.append([i[1] for i in heldout_movies])
            x_batch.append([i[0] for i in base_predictions_on[:-1]])

            if len(x_batch) == self.batch_size:  # batch_size always = 1

                yield Variable(LongTensor(x_batch)), Variable(y_batch_s,
                                                                    requires_grad=False), test_movies, test_movies_r, user
                x_batch = []
                test_movies, test_movies_r = [], []


class SeqTrainer:
    def __init__(self, model, total_anneal_steps, anneal_cap, fixed_curvature):
        self.model = model
        self.epoch = 0

        self.total_anneal_steps = total_anneal_steps
        self.anneal_cap = anneal_cap
        self.fixed_curvature = fixed_curvature

    def epoch(self):
        return self.epoch

    def ncurvature_param_cond(self, key: str) -> bool:
        return "nradius" in key or "curvature" in key

    def pcurvature_param_cond(self, key: str) -> bool:
        return "pradius" in key

    def build_optimizer(self, learning_rate: float) -> torch.optim.Optimizer:
        net_params = [
            v for key, v in self.model.named_parameters()
            if not self.ncurvature_param_cond(key) and not self.pcurvature_param_cond(key)
        ]
        neg_curv_params = [v for key, v in self.model.named_parameters() if self.ncurvature_param_cond(key)]
        pos_curv_params = [v for key, v in self.model.named_parameters() if self.pcurvature_param_cond(key)]
        curv_params = neg_curv_params + pos_curv_params

        net_optimizer = torch.optim.Adam(net_params, lr=learning_rate)
        if not self.fixed_curvature and not curv_params:
            warnings.warn("Fixed curvature disabled, but found no curvature parameters. Did you mean to set "
                          "fixed=True, or not?")
        if not pos_curv_params:
            c_opt_pos = None
        else:
            c_opt_pos = torch.optim.SGD(pos_curv_params, lr=5e-4)

        if not neg_curv_params:
            c_opt_neg = None
        else:
            c_opt_neg = torch.optim.SGD(neg_curv_params, lr=1e-3)

        def condition() -> bool:
            return (not self.fixed_curvature) and (self.epoch >= 10)

        return CurvatureOptimizer(net_optimizer, neg=c_opt_neg,
                                  pos=c_opt_pos, should_do_curvature_step=condition)

    def train(self, optimizer, train_loader, update_count):
        # Turn on training mode
        self.model.train()
        train_loss = 0.0
        batch_idx = 0

        for batch_idx, (x, y_s) in enumerate(train_loader.iter()):
            # x = x.to_dense()
            x = x.to(device)
            if self.total_anneal_steps > 0:
                anneal = min(self.anneal_cap, 1. * update_count / self.total_anneal_steps)
            else:
                anneal = self.anneal_cap

            train_loss += self.model.train_step(optimizer, x, y_s, anneal)

            update_count += 1

        ###print radius value
        if not self.fixed_curvature:
            neg_curv_params = [v for key, v in self.model.named_parameters() if self.ncurvature_param_cond(key)]
            print("radius = {}, c = {}".format(neg_curv_params[0].item(),
                                               1 / (neg_curv_params[0].item() ** 2)))

        self.epoch += 1

        return  train_loss / (batch_idx + 1) , update_count

    def evaluate(self, reader, train_cp_users, topk, is_train_set):
        self.model.eval()

        metrics = {}
        metrics['loss'] = 0.0
        for k in topk:
            metrics['NDCG@' + str(k)] = 0.0
            metrics['Rec@' + str(k)] = 0.0
            metrics['Prec@' + str(k)] = 0.0

        batch = 0
        total_users = 0.0

        # For plotting the results (seq length vs. NDCG@100)
        len_to_ndcg_at_100_map = {}

        for x, y_s, test_movies, test_movies_r, uid in reader.iter_eval():
            batch += 1
            if is_train_set == True and batch > train_cp_users: break

            decoder_output, z_mean, z_log_sigma = self.model(x)

            # Making the logits of previous items in the sequence to be "- infinity"
            decoder_output = decoder_output.data
            x_scattered = torch.zeros(decoder_output.shape[0], decoder_output.shape[2])
            x_scattered = x_scattered.to(device)
            x_scattered[0, :].scatter_(0, x[0].data, 1.0)
            last_predictions = decoder_output[:, -1, :] - (
                    torch.abs(decoder_output[:, -1, :] * x_scattered) * 100000000)

            for batch_num in range(last_predictions.shape[
                                       0]):  # batch_num is ideally only 0, since batch_size is enforced to be always 1
                predicted_scores = last_predictions[batch_num]
                actual_movies_watched = test_movies[batch_num]
                actual_movies_ratings = test_movies_r[batch_num]

                # Calculate NDCG
                _, argsorted = torch.sort(-1.0 * predicted_scores)
                for k in topk:
                    best, now_at, dcg, hits = 0.0, 0.0, 0.0, 0.0

                    rec_list = list(argsorted[:k].cpu().numpy())
                    for m in range(len(actual_movies_watched)):
                        movie = actual_movies_watched[m]
                        now_at += 1.0
                        if now_at <= k: best += 1.0 / float(np.log2(now_at + 1))

                        if movie not in rec_list: continue
                        hits += 1.0
                        dcg += 1.0 / float(np.log2(float(rec_list.index(movie) + 2)))

                    try:
                        metrics['NDCG@' + str(k)] += float(dcg) / float(best)
                        metrics['Rec@' + str(k)] += float(hits) / float(len(actual_movies_watched))
                        metrics['Prec@' + str(k)] += float(hits) / float(k)
                    except:
                        print('Failed uid: ', uid)
                        print(actual_movies_watched)

                    # Only for plotting the graph (seq length vs. NDCG@100)
                    if k == 100:
                        seq_len = int(len(actual_movies_watched)) + int(x[batch_num].shape[0]) + 1
                        if seq_len not in len_to_ndcg_at_100_map: len_to_ndcg_at_100_map[seq_len] = []
                        len_to_ndcg_at_100_map[seq_len].append(float(dcg) / float(best))

                total_users += 1.0

        for k in topk:
            metrics['NDCG@' + str(k)] = round((100.0 * metrics['NDCG@' + str(k)]) / float(total_users), 4)
            metrics['Rec@' + str(k)] = round((100.0 * metrics['Rec@' + str(k)]) / float(total_users), 4)
            metrics['Prec@' + str(k)] = round((100.0 * metrics['Prec@' + str(k)]) / float(total_users), 4)

        return metrics, len_to_ndcg_at_100_map
