import os
import sys

import numpy as np
import pandas as pd


def get_count(tp, id):
    playcount_groupbyid = tp[[id]].groupby(id, as_index=False)
    count = playcount_groupbyid.size()
    return count


def filter_triplets(tp, min_uc=5, min_sc=0):
    # Only keep the triplets for items which were clicked on by at least min_sc users.
    if min_sc > 0:
        itemcount = get_count(tp, 'movieId')
        tp = tp[tp['movieId'].isin(itemcount[itemcount['size'] >= min_sc]['movieId'])]

    # Only keep the triplets for users who clicked on at least min_uc items
    # After doing this, some of the items will have less than min_sc users, but should only be a small proportion
    if min_uc > 0:
        usercount = get_count(tp, 'userId')
        tp = tp[tp['userId'].isin(usercount[usercount['size'] >= min_uc]['userId'])]

    # Update both usercount and itemcount after filtering
    usercount, itemcount = get_count(tp, 'userId'), get_count(tp, 'movieId')
    return tp, usercount, itemcount


def split_train_test_proportion(data, test_prop=0.2):
    data_grouped_by_user = data.groupby('userId')
    tr_list, te_list = list(), list()

    np.random.seed(98765)

    for i, (uid, group) in enumerate(data_grouped_by_user):
        n_items_u = len(group)
        #         print(f'uid:{uid}, n_items_u:{n_items_u}')
        if n_items_u >= 5:
            idx = np.zeros(n_items_u, dtype='bool')
            # idx[np.random.choice(n_items_u, size=int(test_prop * n_items_u), replace=False).astype('int64')] = True
            idx[int((1.0 - test_prop) * n_items_u):] = True
            # print(idx)

            tr_list.append(group[np.logical_not(idx)])
            te_list.append(group[idx])

        if i % 1000 == 0:
            print("%d users sampled" % i)
            sys.stdout.flush()

    data_tr = pd.concat(tr_list)
    data_te = pd.concat(te_list)

    return data_tr, data_te


def numerize(tp, profile2id, show2id):
    uid = list(map(lambda x: profile2id[x], tp['userId']))
    sid = list(map(lambda x: show2id[x], tp['movieId']))
    ra = list(map(lambda x: x, tp['rating']))
    ret = pd.DataFrame(data={'uid': uid, 'sid': sid, 'rating': ra}, columns=['uid', 'sid', 'rating'])
    ret['rating'] = ret['rating'].apply(pd.to_numeric)
    return ret

def generate_seq_ml1m(data_dir, out_dir, n_heldout_users=750, max_seq_len=1000, threshold=3.5):
    cols = ['userId', 'movieId', 'rating', 'timestamp']
    dtypes = {'userId': 'int', 'movieId': 'int', 'timestamp': 'int', 'rating': 'int'}
    raw_data = pd.read_csv(os.path.join(data_dir, 'ratings.dat'), sep='::', names=cols, parse_dates=['timestamp'], dtype=dtypes)

    max_seq_len = 1000

    # binarize the data (only keep ratings >= 4)
    raw_data = raw_data[raw_data['rating'] > threshold]

    # Remove users with greater than $max_seq_len number of watched movies
    raw_data = raw_data.groupby(["userId"]).filter(lambda x: len(x) <= max_seq_len)

    # Sort data values with the timestamp
    raw_data = raw_data.groupby(["userId"]).apply(lambda x: x.sort_values(["timestamp"], ascending=True)).reset_index(
        drop=True)

    raw_data, user_activity, item_popularity = filter_triplets(raw_data)

    sparsity = 1. * raw_data.shape[0] / (user_activity.shape[0] * item_popularity.shape[0])

    print("After filtering, there are %d watching events from %d users and %d movies (sparsity: %.3f%%)" %
          (raw_data.shape[0], user_activity.shape[0], item_popularity.shape[0], sparsity * 100))

    unique_uid = user_activity.index

    np.random.seed(98765)
    idx_perm = np.random.permutation(unique_uid.size)
    unique_uid = user_activity['userId'][idx_perm]

    # create train/validation/test users
    n_users = unique_uid.size

    tr_users = unique_uid[:(n_users - n_heldout_users * 2)]
    vd_users = unique_uid[(n_users - n_heldout_users * 2): (n_users - n_heldout_users)]
    te_users = unique_uid[(n_users - n_heldout_users):]

    train_plays = raw_data.loc[raw_data['userId'].isin(tr_users)]

    unique_sid = pd.unique(train_plays['movieId'])

    show2id = dict((sid, i) for (i, sid) in enumerate(unique_sid))
    profile2id = dict((pid, i) for (i, pid) in enumerate(unique_uid))

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    with open(os.path.join(out_dir, 'unique_sid.txt'), 'w') as f:
        for sid in unique_sid:
            f.write('%s\n' % sid)

    vad_plays = raw_data.loc[raw_data['userId'].isin(vd_users)]
    vad_plays = vad_plays.loc[vad_plays['movieId'].isin(unique_sid)]

    vad_plays_tr, vad_plays_te = split_train_test_proportion(vad_plays)

    test_plays = raw_data.loc[raw_data['userId'].isin(te_users)]
    test_plays = test_plays.loc[test_plays['movieId'].isin(unique_sid)]

    test_plays_tr, test_plays_te = split_train_test_proportion(test_plays)

    train_data = numerize(train_plays, profile2id, show2id)
    train_data.to_csv(os.path.join(out_dir, 'train.csv'), index=False)

    vad_data_tr = numerize(vad_plays_tr, profile2id, show2id)
    vad_data_tr.to_csv(os.path.join(out_dir, 'validation_tr.csv'), index=False)

    vad_data_te = numerize(vad_plays_te, profile2id, show2id)
    vad_data_te.to_csv(os.path.join(out_dir, 'validation_te.csv'), index=False)

    test_data_tr = numerize(test_plays_tr, profile2id, show2id)
    test_data_tr.to_csv(os.path.join(out_dir, 'test_tr.csv'), index=False)

    test_data_te = numerize(test_plays_te, profile2id, show2id)
    test_data_te.to_csv(os.path.join(out_dir, 'test_te.csv'), index=False)