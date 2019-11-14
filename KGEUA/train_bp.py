import itertools
import gc
import numpy as np
from model import P


def filter_alignment(labeled_alignment, model):
    new_labeled_alignment = set()
    ref_sim_mat = model.eval_ref_sim_mat(sigmoid=True)
    for x, y in labeled_alignment:
        if ref_sim_mat[x, y] > P.lambda_4:
            new_labeled_alignment.add((x, y))
    print('after filter alignment: ', len(new_labeled_alignment))
    return new_labeled_alignment

def my_find_potential_alignment(ref_sim_mat, th, top_k):
    curr_labeled_alignment = list()
    for i in range(ref_sim_mat.shape[0]):
        sim_rank = (-ref_sim_mat[i, :]).argsort()[: top_k]
        sim_rank_filter = [index for index in sim_rank if ref_sim_mat[i, index] > th]
        if sim_rank_filter:
            curr_labeled_alignment.extend([pair for pair in itertools.product([i], sim_rank_filter)])
    return curr_labeled_alignment

def get_min_sim(array):
    sim_dict = dict()
    sim_list = list()
    count = 0
    for i in range(array.shape[0]):
        sim_dict[i] = np.where(array[i] == array[i].max())[0][0]
        if i == sim_dict[i]:
            sim_list.append(array[i, i])
        else:
            count += 1
    print('min right sim: {}, wrong sim count: {}'.format(min(sim_list), count))
    return min(sim_list)

def match(sup_sim_mat, ref_sim_mat, th, common_mat=None):
    if P.dynamic:
        min_sim = get_min_sim(sup_sim_mat)
        if min_sim > th:
            th = min_sim
    try:
        ref_sim_mat = (1 - P.lambda_4) * ref_sim_mat + P.lambda_4 * common_mat
    except:
        pass
    row_num, col_num = ref_sim_mat.shape
    min_num = min(row_num, col_num)
    matched_num = 0
    matched = np.zeros_like(ref_sim_mat)
    for index in (-ref_sim_mat).argsort(axis=None):
        row = int(index / col_num)
        col = int(index % col_num)
        if (1 not in matched[row, :]) and (1 not in matched[:, col]):
            matched[row, col] = 1
            matched_num += 1
        if ref_sim_mat[row, col] < th:
            break
        if matched_num > min_num:
            break
    pairs = list(zip(np.where(matched == 1)[0], np.where(matched == 1)[1]))
    return pairs

def bootstrapping(model, common_mat=None):
    is_sigmoid = False
    ref_sim_mat = model.eval_ref_sim_mat(sigmoid=is_sigmoid)
    sup_sim_mat = model.eval_sup_sim_mat(sigmoid=is_sigmoid)
    th = P.lambda_3
    labeled_alignment = match(sup_sim_mat, ref_sim_mat, th, common_mat=common_mat)

    print('len labeled_alignment: {}'.format(len(labeled_alignment)))

    if labeled_alignment is not None:
        ents1 = [model.ref_ent1[pair[0]] for pair in labeled_alignment]
        ents2 = [model.ref_ent2[pair[1]] for pair in labeled_alignment]
    else:
        ents1, ents2 = None, None
    del ref_sim_mat
    gc.collect()
    return labeled_alignment, ents1, ents2
