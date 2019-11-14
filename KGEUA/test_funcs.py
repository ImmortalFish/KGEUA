import numpy as np
from param import P
from sklearn.preprocessing import normalize, StandardScaler

g = 1000000000


def pair2dic(ents_1, ents_2):
    dic = dict()
    for i, j in zip(ents_1, ents_2):
        if i not in dic.keys():
            dic[i] = j
    return dic

def normalization(mat, axis):
    if P.norm == 'None':
        print('no normalization')
        return mat
    elif P.norm == 'max-min-axis':
        print('max-min-axis normalization')
        return np.apply_along_axis(lambda x: (x - x.min()) / (x.max() - x.min()), axis, mat)
    elif P.norm == 'max':
        print('max normalization')
        return normalize(mat, axis=axis, norm=P.norm)
    elif P.norm == 'neg_one':
        print('-1 to 1 normalization')
        return np.apply_along_axis(lambda x: (x - x.min()) / (1.0 * (x.max() - x.min())) * 2 -1, axis, mat)
    elif P.norm == 'zscore':
        print('zscore normalization')
        scaler = StandardScaler()
        return scaler.fit_transform(mat)
    elif P.norm == 'max-min-global':
        print('max-min-global normalization')
        return (mat - mat.min()) / (mat.max() - mat.min())

def evaluate(embed1, embed2, top_k, real_ent1, real_ent2, ref_ent1, ref_ent2, common_mat):
    real_ent_dict = pair2dic(real_ent1, real_ent2)
    sim_mat = np.matmul(embed1, embed2.T)
    if common_mat is not None:
        try:
            sim_mat = P.alpha * sim_mat + (1 - P.alpha) * common_mat
        except:
            sim_mat = P.alpha * sim_mat + (1 - P.alpha) * common_mat.T
    num = dict(zip(top_k, [0] * len(top_k)))
    num_1 = dict(zip(top_k, [0] * len(top_k)))
    for i in range(sim_mat.shape[0]):
        if ref_ent1[i] not in real_ent1:
            continue
        rank = (-sim_mat[i, :]).argsort() # 从大到小排序的索引值
        label_ent = real_ent_dict[ref_ent1[i]]
        label_index = np.where(rank == ref_ent2.index(label_ent))[0][0]
        for k in num.keys():
            if label_index < k:
                num[k] += 1
        for j, index in enumerate(rank):
            if j == 50:
                break
            if ref_ent2[index] == label_ent:
                for k in num_1.keys():
                    if  j < k:
                        num_1[k] += 1
    return num, num_1

def my_evaluate(embed1, embed2, top_k, real_ent1, real_ent2, ref_ent1, ref_ent2, common_mat):
    if common_mat is not None:
        common_mat_col = normalization(common_mat, axis=0)
        common_mat_row = normalization(common_mat, axis=1)
    else:
        common_mat_col = None
        common_mat_row = None

    num_1_to_2_1, num_1_to_2_2 = evaluate(embed1, embed2, top_k, real_ent1, real_ent2, ref_ent1, ref_ent2,
                                          common_mat_row)
    num_2_to_1_1, num_2_to_1_2 = evaluate(embed2, embed1, top_k, real_ent2, real_ent1, ref_ent2, ref_ent1,
                                          common_mat_col)
    result = []
    for k in top_k:
        hits = (num_1_to_2_1[k] + num_2_to_1_1[k]) / (len(real_ent1) * 2)
        result.append(hits)
    print('1 hits@{} = {}'.format(top_k, result))
    print(num_1_to_2_1, num_2_to_1_1)

    result = []
    for k in top_k:
        hits = (num_1_to_2_2[k] + num_2_to_1_2[k]) / (len(real_ent1) * 2)
        result.append(hits)
    print('2 hits@{} = {}'.format(top_k, result))
    print(num_1_to_2_2, num_2_to_1_2)
    return result