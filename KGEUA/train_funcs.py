import math
import multiprocessing

import gc
import sys

import numpy as np
import time
import random

from param import P
from model import KGEUA_Model

import utils as ut

if P.seed != 0:
    random.seed(P.seed)

g = 1000000000


def get_model(folder):
    print("data folder:", folder)
    ori_triples1, ori_triples2, seed_sup_ent1, seed_sup_ent2,\
    ref_ent1, ref_ent2, _, ent_n, rel_n, real_ent1, real_ent2 = ut.read_input(folder)
    triples1, triples2 = ut.add_sup_triples(ori_triples1, ori_triples2, seed_sup_ent1, seed_sup_ent2)

    model = KGEUA_Model(ent_n, rel_n, seed_sup_ent1, seed_sup_ent2, ref_ent1, ref_ent2, ori_triples1.ent_list,
                      ori_triples2.ent_list, len(seed_sup_ent1) + len(ref_ent1), P.embed_size, P.learning_rate)
    return ori_triples1, ori_triples2, triples1, triples2, model, real_ent1, real_ent2


def train_tris_k_epo(model, tris1, tris2, k, trunc_ent_num, ents1, ents2, is_test=True):
    if trunc_ent_num > 0:
        t1 = time.time()
        nbours1 = generate_neighbours_multi_embed(model.eval_kb1_embed(), model.kb1_ents, trunc_ent_num)
        nbours2 = generate_neighbours_multi_embed(model.eval_kb2_embed(), model.kb2_ents, trunc_ent_num)
        print("generate neighbours: {:.3f} s, size: {:.6f} G".format(time.time() - t1, sys.getsizeof(nbours1)/g))
    else:
        nbours1, nbours2 = None, None
    for i in range(k):
        loss, t2 = train_tris_1epo(model, tris1, tris2, nbours1, nbours2)
        if ents1 is not None and len(ents1) > 0:
            train_alignment_1epo(model, tris1, tris2, ents1, ents2, 1)
        print("triple_loss = {:.3f}, time = {:.3f} s".format(loss, t2))
    if nbours1 is not None:
        del nbours1, nbours2
        gc.collect()
    if is_test:
        return model.test()
    else:
        return None


def train_tris_1epo(model, triples1, triples2, nbours1, nbours2):
    triple_loss = 0
    start = time.time()
    triples_num = triples1.triples_num + triples2.triples_num
    triple_steps = math.ceil(triples_num / P.batch_size)
    triple_fetches = {"triple_loss": model.triple_loss, "train_op": model.triple_optimizer}
    for step in range(triple_steps):
        if nbours2 is None:
            batch_pos, batch_neg = generate_pos_neg_batch(triples1, triples2, step, P.batch_size, multi=P.nums_neg)
        else:
            batch_pos, batch_neg = generate_batch_via_neighbour(triples1, triples2, step, P.batch_size,
                                                                nbours1, nbours2, multi=P.nums_neg)
        triple_feed_dict = {model.pos_hs: [x[0] for x in batch_pos],
                            model.pos_rs: [x[1] for x in batch_pos],
                            model.pos_ts: [x[2] for x in batch_pos],
                            model.neg_hs: [x[0] for x in batch_neg],
                            model.neg_rs: [x[1] for x in batch_neg],
                            model.neg_ts: [x[2] for x in batch_neg]}
        vals = model.session.run(fetches=triple_fetches, feed_dict=triple_feed_dict)
        triple_loss += vals["triple_loss"]
        triple_loss /= triple_steps
    random.shuffle(triples1.triple_list)
    random.shuffle(triples2.triple_list)
    end = time.time()
    return triple_loss, round(end - start, 2)


def train_alignment_1epo(model, tris1, tris2, ents1, ents2, ep):
    if ents1 is None or len(ents1) == 0:
        return
    newly_tris1, newly_tris2 = generate_triples_of_latent_ents(tris1, tris2, ents1, ents2)
    steps = math.ceil(((len(newly_tris1) + len(newly_tris2)) / P.batch_size))
    if steps == 0:
        steps = 1
    for i in range(ep):
        t1 = time.time()
        alignment_loss = 0
        for step in range(steps):
            newly_batch1, newly_batch2 = generate_pos_batch(newly_tris1, newly_tris2, step, P.batch_size)
            newly_batch1.extend(newly_batch2)
            alignment_fetches = {"loss": model.alignment_loss, "train_op": model.alignment_optimizer}
            alignment_feed_dict = {model.new_h: [tr[0] for tr in newly_batch1],
                                   model.new_r: [tr[1] for tr in newly_batch1],
                                   model.new_t: [tr[2] for tr in newly_batch1]}
            alignment_vals = model.session.run(fetches=alignment_fetches, feed_dict=alignment_feed_dict)
            alignment_loss += alignment_vals["loss"]
        alignment_loss /= (len(newly_tris1) + len(newly_tris2))
        print("alignment_loss = {:.3f}, time = {:.3f} s".format(alignment_loss, time.time() - t1))


def cal_neighbours_embed(frags, ent_list, sub_embed, embed, k):
    dic = dict()
    sim_mat = np.matmul(sub_embed, embed.T)

    for i in range(sim_mat.shape[0]):
        sort_index = np.argpartition(-sim_mat[i, :], k + 1)
        dic[frags[i]] = ent_list[sort_index[0:k + 1]]
    del sim_mat
    gc.collect()
    return dic


def generate_neighbours_multi_embed(embed, ent_list, k):
    ent_frags = ut.div_list(np.array(ent_list), P.nums_threads)
    ent_frag_indexes = ut.div_list(np.array(range(len(ent_list))), P.nums_threads)
    pool = multiprocessing.Pool(processes=len(ent_frags))
    results = list()
    for i in range(len(ent_frags)):
        results.append(pool.apply_async(cal_neighbours_embed,
                                        (ent_frags[i], np.array(ent_list), embed[ent_frag_indexes[i], :], embed, k)))
    pool.close()
    pool.join()
    dic = dict()
    for res in results:
        dic = ut.merge_dic(dic, res.get())
    del embed
    gc.collect()
    return dic


def trunc_sampling(pos_triples, all_triples, dic, ent_list):
    neg_triples = list()
    for (h, r, t) in pos_triples:
        h2, r2, t2 = h, r, t
        while True:
            choice = random.randint(0, 999)
            if choice < 500:
                candidates = dic.get(h, ent_list)
                index = random.sample(range(0, len(candidates)), 1)[0]
                h2 = candidates[index]
            elif choice >= 500:
                candidates = dic.get(t, ent_list)
                index = random.sample(range(0, len(candidates)), 1)[0]
                t2 = candidates[index]
            if (h2, r2, t2) not in all_triples:
                break
        neg_triples.append((h2, r2, t2))
    return neg_triples


def trunc_sampling_multi(pos_triples, all_triples, dic, ent_list, multi):
    neg_triples = list()
    ent_list = np.array(ent_list)
    for (h, r, t) in pos_triples:
        choice = random.randint(0, 999)
        if choice < 500:
            candidates = dic.get(h, ent_list)
            index = random.sample(range(0, len(candidates)), multi)
            h2s = candidates[index]
            for h2 in h2s:
                if (h2, r, t) not in all_triples:
                    neg_triples.append((h2, r, t))
        elif choice >= 500:
            candidates = dic.get(t, ent_list)
            index = random.sample(range(0, len(candidates)), multi)
            t2s = candidates[index]
            for t2 in t2s:
                if (h, r, t2) not in all_triples:
                    neg_triples.append((h, r, t2))
    return neg_triples


def generate_batch_via_neighbour(triples1, triples2, step, batch_size, neighbours_dic1, neighbours_dic2, multi=1):
    assert multi >= 1
    pos_triples1, pos_triples2 = generate_pos_batch(triples1.triple_list, triples2.triple_list, step, batch_size)
    neg_triples = list()
    if len(triples1.ent_list) < 10000:
        for i in range(multi):
            neg_triples.extend(trunc_sampling(pos_triples1, triples1.triples, neighbours_dic1, triples1.ent_list))
            neg_triples.extend(trunc_sampling(pos_triples2, triples2.triples, neighbours_dic2, triples2.ent_list))
    else:
        neg_triples.extend(
            trunc_sampling_multi(pos_triples1, triples1.triples, neighbours_dic1, triples1.ent_list, multi))
        neg_triples.extend(
            trunc_sampling_multi(pos_triples2, triples2.triples, neighbours_dic2, triples2.ent_list, multi))
    pos_triples1.extend(pos_triples2)
    return pos_triples1, neg_triples


def generate_pos_batch(triples1, triples2, step, batch_size):
    num1 = int(len(triples1) / (len(triples1) + len(triples2)) * batch_size)
    num2 = batch_size - num1
    start1 = step * num1
    start2 = step * num2
    end1 = start1 + num1
    end2 = start2 + num2
    if end1 > len(triples1):
        end1 = len(triples1)
    if end2 > len(triples2):
        end2 = len(triples2)
    pos_triples1 = triples1[start1: end1]
    pos_triples2 = triples2[start2: end2]
    return pos_triples1, pos_triples2


def generate_neg_triples(pos_triples, triples_data):
    all_triples = triples_data.triples
    ents = triples_data.ent_list
    neg_triples = list()
    for (h, r, t) in pos_triples:
        h2, r2, t2 = h, r, t
        while True:
            choice = random.randint(0, 999)
            if choice < 500:
                h2 = random.sample(ents, 1)[0]
            elif choice >= 500:
                t2 = random.sample(ents, 1)[0]
            if (h2, r2, t2) not in all_triples:
                break
        neg_triples.append((h2, r2, t2))
    return neg_triples


def generate_neg_triple_ht(triple, all_triples, ents, ht):
    h2, r2, t2 = triple[0], triple[1], triple[2]
    while True:
        choice = random.randint(0, 999)
        if choice < 500:
            h2 = random.sample(ents, 1)[0]
        elif choice >= 500:
            t2 = random.sample(ents, 1)[0]
        if (h2, r2, t2) not in all_triples and (h2, t2) not in ht:
            return h2, r2, t2


def generate_neg_triples_batch(pos_triples, triples_data, is_head):
    all_triples = triples_data.triples
    ents = triples_data.ent_list
    n = len(pos_triples)
    pos_triples_mat = np.matrix(pos_triples)
    neg_ent_mat = np.matrix(np.random.choice(np.array(ents), n)).T
    if is_head:
        neg_triples_mat = np.column_stack((neg_ent_mat, pos_triples_mat[:, [1, 2]]))
    else:
        neg_triples_mat = np.column_stack((pos_triples_mat[:, [0, 1]], neg_ent_mat))
    ii, jj = neg_triples_mat.shape
    neg_triples = list()
    for i in range(ii):
        tr = (neg_triples_mat[i, 0], neg_triples_mat[i, 1], neg_triples_mat[i, 2])
        if tr not in all_triples:
            neg_triples.append(tr)
    return neg_triples


def generate_pos_neg_batch(triples1, triples2, step, batch_size, multi=1):
    assert multi >= 1
    pos_triples1, pos_triples2 = generate_pos_batch(triples1.triple_list, triples2.triple_list, step, batch_size)
    neg_triples = list()
    for i in range(multi):
        choice = random.randint(0, 999)
        if choice < 500:
            h = True
        else:
            h = False
        neg_triples.extend(generate_neg_triples_batch(pos_triples1, triples1, h))
        neg_triples.extend(generate_neg_triples_batch(pos_triples2, triples2, h))
    pos_triples1.extend(pos_triples2)
    return pos_triples1, neg_triples

def generate_triples_of_latent_ents(triples1, triples2, ents1, ents2):
    assert len(ents1) == len(ents2)
    newly_triples1, newly_triples2 = list(), list()
    for i in range(len(ents1)):
        newly_triples1.extend(generate_newly_triples(ents1[i], ents2[i], triples1.rt_dict, triples1.hr_dict))
        newly_triples2.extend(generate_newly_triples(ents2[i], ents1[i], triples2.rt_dict, triples2.hr_dict))
    print("newly triples: {}, {}".format(len(newly_triples1), len(newly_triples2)))
    return newly_triples1, newly_triples2


def generate_newly_triples(ent1, ent2, rt_dict1, hr_dict1):
    newly_triples = list()
    for r, t in rt_dict1.get(ent1, set()):
        newly_triples.append((ent2, r, t))
    for h, r in hr_dict1.get(ent1, set()):
        newly_triples.append((h, r, ent2))
    return newly_triples
