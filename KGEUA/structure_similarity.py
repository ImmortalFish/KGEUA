import numpy as np
from collections import Counter
import itertools

def read_truth(folder):
    """
    读取对齐的用户id对
    :param folder: 数据所在文件夹
    :return: 已知的对齐的用户id对[(id_1, id_2)]
    """
    truth = list()
    with open(folder + 'truth_ents_ids', 'r') as f:
        for line in f.readlines():
            ent_1, ent_2 = line.strip('\n').split('\t')
            truth.append((int(ent_1), int(ent_2)))
    return truth

def read_ent(folder, name, split=True):
    """
    读取用户id
    :param folder: 数据所在文件夹
    :param name: 用户id文件名
    :param split: 是否按分隔符切分行
    :return: ents:[id], ent_user_dict:{id:用户名}
    """
    ents = list()
    ent_user_dict = dict()
    with open(folder + name, 'r') as f:
        for line in f.readlines():
            if split:
                ent, user = line.strip('\n').split('\t')
                ent_user_dict[int(ent)] = user
            else:
                ent = line.strip('\n')
            ents.append(int(ent))
    return ents, ent_user_dict

def read_triple(folder, name):
    """
    读取三元组文件
    :param folder: 数据所在文件夹
    :param name: 文件名
    :return: [id_1, relationship, id_2]
    """
    triples = list()
    with open(folder + name, 'r') as f:
        for line in f.readlines():
            ent_1, rel, ent_2 = line.strip('\n').split('\t')
            triples.append((int(ent_1), int(rel), int(ent_2)))
    return triples

def get_other_ent(train_truth_ent1, train_truth_ent2, ents_1, ents_2):
    other_ent1 = [ent for ent in ents_1 if ent not in train_truth_ent1]
    other_ent2 = [ent for ent in ents_2 if ent not in train_truth_ent2]
    return other_ent1, other_ent2

def split_data(truth, ents_1, ents_2, ratio):
    train, test = ratio
    all_truth_1 = [pair[0] for pair in truth] # 网络1中的对齐用户
    all_truth_2 = [pair[1] for pair in truth] # 网络2中的对齐用户
    train_num = int(len(all_truth_1) * train) # 训练集的数量
    print(ratio, train_num)
    '''根据已知的对齐用户对划分数据集'''
    test_truth_ent1, test_truth_ent2 = all_truth_1[train_num:], all_truth_2[train_num:] # 测试集
    train_truth_ent1, train_truth_ent2 = all_truth_1[: train_num], all_truth_2[: train_num] # 训练集
    other_ent1, other_ent2 = get_other_ent(train_truth_ent1, train_truth_ent2, ents_1, ents_2) # 不在训练集中的其他用户
    return test_truth_ent1, test_truth_ent2, train_truth_ent1, train_truth_ent2, other_ent1, other_ent2

def get_related(ents, triples):
    min_index = min(ents)
    related = np.zeros([len(ents), len(ents)])
    for h, r, t in triples:
        row, col = h - min_index, t - min_index
        related[row, col] = related[row, col] + 1
    return related

def get_total_related(related, sup, min_index=None):
    total_count = np.zeros(len(sup))
    if min_index:
        sup = [ent - min_index for ent in sup]
    for i, ent in enumerate(sup):
        total_count[i] = related[ent, :].sum() + related[:, ent].sum()
    return total_count

def cal_importance(ref, related, total_sup, sup, min_index=None):
    im_mat = np.zeros([len(ref), len(sup)])
    if min_index:
        ref = [ent - min_index for ent in ref]
        sup = [ent - min_index for ent in sup]
    for i, ent in enumerate(ref):
        count = list(np.where(related[ent, sup] != 0)[0])
        col_count = list(np.where(related[sup, ent] != 0)[0])
        count.extend(col_count)
        c = Counter(count)
        for k, v in c.items():
            im = v / total_sup[k]
            im_mat[i, k] = im
    return im_mat

def get_common_pair(related_1, related_2, ref_ent1, ref_ent2, sup_ent1, sup_ent2, im_mat_1, im_mat_2, min_index=None):
    sim_mat = np.zeros([len(ref_ent1), len(ref_ent2)])
    if min_index:
        ref_ent2 = [ent - min_index for ent in ref_ent2]
        sup_ent2 = [ent - min_index for ent in sup_ent2]
    for sup_1, sup_2 in zip(sup_ent1, sup_ent2):
        related_sup_1 = set(np.where(related_1[sup_1, :] != 0)[0]) | set(np.where(related_1[:, sup_1] != 0)[0])
        related_sup_2 = set(np.where(related_2[sup_2, :] != 0)[0]) | set(np.where(related_2[:, sup_2] != 0)[0])
        related_sup_1 = [ref_ent1.index(ent) for ent in related_sup_1 if ent in ref_ent1]
        related_sup_2 = [ref_ent2.index(ent) for ent in related_sup_2 if ent in ref_ent2]
        for ref_1, ref_2 in itertools.product(related_sup_1, related_sup_2):
            im_12 = (im_mat_1[ref_1, sup_ent1.index(sup_1)] + im_mat_2[ref_2, sup_ent2.index(sup_2)]) / 2
            sim_mat[ref_1, ref_2] += im_12
    return sim_mat

def save_data(folder, train_truth_ent1, train_truth_ent2, test_truth_ent1, test_truth_ent2, suffix):
    """
    保存划分后的训练集和测试集的groundtruth
    """
    sup_writter = open(folder + 'twitter_foursquare_groundtruth_{}_train'.format(suffix), 'w', encoding='utf-8')
    for sup_1, sup_2 in zip(train_truth_ent1, train_truth_ent2):
        sup_writter.write(str(sup_1) + '\t' + str(sup_2) + '\n')

    real_writter = open(folder + 'twitter_foursquare_groundtruth_{}_test'.format(suffix), 'w', encoding='utf-8')
    for real_1, real_2 in zip(test_truth_ent1, test_truth_ent2):
        real_writter.write(str(real_1) + '\t' + str(real_2) + '\n')

def save_data_with_user(folder, train_truth_ent1, train_truth_ent2, test_truth_ent1, test_truth_ent2,
                        ent_user_dict_1, ent_user_dict_2, suffix):
    """
    保存划分后的训练集和测试集的groundtruth的对应的用户名
    """
    sup_writter = open(folder + 'twitter_foursquare_groundtruth_{}_train_username'.format(suffix.split('_')[0]),
                       'w', encoding='utf-8')
    for sup_1, sup_2 in zip(train_truth_ent1, train_truth_ent2):
        if ent_user_dict_1[sup_1] != ent_user_dict_2[sup_2]:
            print(ent_user_dict_1[sup_1], ent_user_dict_2[sup_2])
        line = ent_user_dict_1[sup_1] + '\n'
        sup_writter.write(line)

    real_writter = open(folder + 'twitter_foursquare_groundtruth_{}_test_username'.format(suffix.split('_')[0]), 'w',
                       encoding='utf-8')
    for real_1, real_2 in zip(test_truth_ent1, test_truth_ent2):
        if ent_user_dict_1[real_1] != ent_user_dict_2[real_2]:
            print(ent_user_dict_1[real_1], ent_user_dict_2[real_2])
        line = ent_user_dict_1[real_1]+ '\n'
        real_writter.write(line)


def run(folder, rate, min_index):
    truth = read_truth(folder)
    ents_1, ent_user_dict_1 = read_ent(folder, 'twitter_user_ids')
    ents_2, ent_user_dict_2 = read_ent(folder, 'foursquare_user_ids')
    triples_1, triples_2 = read_triple(folder, 'twitter_triples'), read_triple(folder, 'foursquare_triples')
    related_1, related_2 = get_related(ents_1, triples_1), get_related(ents_2, triples_2)
    '''
    test_truth_ent*:groundtruth中的测试集
    train_truth_ent*:groundtruth中的训练集
    other_ent*:去除训练集的其他实体，当然也包含了测试集，最终计算hits时只选取测试集的实体
    '''
    test_truth_ent1, test_truth_ent2, train_truth_ent1, train_truth_ent2, other_ent1, other_ent2 = split_data(truth, ents_1, ents_2, rate)
    total_train_1, total_train_2 = get_total_related(related_1, train_truth_ent1), get_total_related(related_2, train_truth_ent2,
                                                                                         min_index=min_index)
    im_mat_1 = cal_importance(other_ent1, related_1, total_train_1, train_truth_ent1)
    im_mat_2 = cal_importance(other_ent2, related_2, total_train_2, train_truth_ent2, min_index=min_index)
    sim_mat = get_common_pair(related_1, related_2, other_ent1, other_ent2, train_truth_ent1, train_truth_ent2, im_mat_1, im_mat_2,
                              min_index=min_index)

    count = 0
    for test_1, test_2 in zip(test_truth_ent1, test_truth_ent2):
        index = other_ent1.index(test_1)
        first = (-sim_mat[index]).argsort()[0]
        if other_ent2[first] == test_2:
            count += 1
    print(len(test_truth_ent1), count)

    suffix = str(round(rate[0] * 10)) + '_' + str(round(rate[1] * 10))
    np.save(folder + 'structure_sim_mat_{}_no_norm.npy'.format(suffix), sim_mat)

    '''
    如果要保存划分后的groundtruth则使用下面两个方法，记得修改方法中的名称
    '''
    # save_data_with_user(folder, sup_ent1, sup_ent2, real_ent1, real_ent2, ent_user_dict_1, ent_user_dict_2, suffix)
    # save_data(folder, sup_ent1, sup_ent2, real_ent1, real_ent2, suffix)


if __name__ == '__main__':
    folder = '../dataset/twitter_foursquare_without_user/' # 数据集存放文件夹
    rates = [(round(i * 0.1, 1), round(1 - i * 0.1, 1)) for i in range(1, 10)] # 训练集-测试集划分比例
    min_index = 5120 # twitter中的用户数量
    for rate in rates:
        run(folder, rate, min_index)