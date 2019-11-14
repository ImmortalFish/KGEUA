import time

import numpy as np
from train_funcs import get_model, train_tris_k_epo, train_alignment_1epo
from train_bp import bootstrapping
from model import P

def train(folder):
    ori_triples1, ori_triples2, triples1, triples2, model, real_ent1, real_ent2 = get_model(folder)
    real_ent_pairs = [(ent1, ent2) for ent1, ent2 in zip(real_ent1, real_ent2)]
    ents1, ents2 = None, None

    hits = None

    print('reading structure sim mat')
    suffix = str(round(P.train_ratio * 10)) + '_' + str(round((1 - P.train_ratio) * 10))
    mat_path = '../dataset/twitter_foursquare_without_user/structure_sim_mat_{}_no_norm.npy'
    mat = np.load(mat_path.format(suffix))

    trunc_ent_num = P.near_k

    for t in range(1, P.iter + 1):
        print("iteration ", t)
        train_tris_k_epo(model, triples1, triples2, P.k_epoch, trunc_ent_num, None, None, is_test=False)
        if P.alignment:
            train_alignment_1epo(model, triples1, triples2, ents1, ents2, P.alignment_epo)
        train_tris_k_epo(model, triples1, triples2, P.k_epoch, trunc_ent_num, None, None, is_test=False)
        if P.alignment:
            labeled_align, ents1, ents2 = bootstrapping(model, common_mat=mat)
            train_alignment_1epo(model, triples1, triples2, ents1, ents2, P.alignment_epo)

        true_num = 0
        for index, pair in enumerate(zip(ents1, ents2)):
            if (pair[0], pair[1]) in real_ent_pairs:
                true_num += 1
        print('true pair num is: ', true_num)
        print('\n')

        if t == P.iter:
            cos_sim_mat = model.my_eval_sim_mat()
            np.save(folder + 'cos_sim_mat.npy', cos_sim_mat)

        hits = model.test(real_ent1, real_ent2, common_mat=mat)
    return hits

def run():
    t = time.time()
    folder = '../dataset/twitter_foursquare_without_user'
    hits = train(folder)
    hits = list(map(lambda x: round(x, 6), hits))
    if P.file != '':
        with open(P.file, 'a+', encoding='utf-8') as f:
            params = P.__dict__
            params.pop('params')
            for k, v in params.items():
                f.write(str(k) + ': ' + str(v) + '\n')
            f.write('\n')
            f.write('hits@{} = {}'.format(params['ent_top_k'], hits))
            f.write('\n')
            f.write('-------------------------------------------------------------------------------'
                    '--------------------------------------------------------\n')
    print("total time = {:.3f} s".format(time.time() - t))


if __name__ == '__main__':
    run()
