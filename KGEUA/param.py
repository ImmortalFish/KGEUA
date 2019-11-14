import argparse
import ast

class Params:
    def __init__(self):
        self.params = self.sys_param()
        self.embed_size = self.params.embed_size
        self.batch_size = self.params.batch_size
        self.iter = self.params.iter
        self.k_epoch = self.params.k_epoch
        self.learning_rate = self.params.learning_rate

        self.ent_top_k = list(range(1, 51))
        self.nums_threads = 10

        self.lambda_1 = self.params.lambda_1
        self.lambda_2 = self.params.lambda_2
        self.lambda_3 = self.params.lambda_3
        self.lambda_4 = self.params.lambda_4
        self.mu_1 = self.params.mu_1

        self.near_k = self.params.near_k
        self.nums_neg = self.params.nums_neg

        self.heuristic = False

        self.alpha = self.params.alpha
        self.use_mat = self.params.use_mat
        self.dynamic = self.params.dynamic
        self.range_0 = self.params.range_0
        self.match_degree = self.params.match_degree
        self.train_ratio = self.params.train_ratio
        self.alignment_epo = self.params.alignment_epo
        self.norm = self.params.norm
        self.alignment = self.params.alignment
        self.seed = self.params.seed
        self.file = self.params.file

    def sys_param(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--embed_size', default=120, type=int)
        parser.add_argument('--batch_size', default=20000, type=int)
        parser.add_argument('--iter', default=50, type=int)
        parser.add_argument('--k_epoch', default=5, type=int)
        parser.add_argument('--learning_rate', default=0.01, type=float)
        parser.add_argument('--lambda_1', default=0.01, type=float)
        parser.add_argument('--lambda_2', default=2.0, type=float)
        parser.add_argument('--lambda_3', default=0.75, type=float)
        parser.add_argument('--lambda_4', default=0.0, type=float)
        parser.add_argument('--mu_1', default=0.2, type=float)
        parser.add_argument('--near_k', default=100, type=int)
        parser.add_argument('--nums_neg', default=10, type=int)
        parser.add_argument('--alpha', default=0.6, type=float)
        parser.add_argument('--dynamic', default=True, type=ast.literal_eval)
        parser.add_argument('--range_0', default=True, type=ast.literal_eval)
        parser.add_argument('--match_degree', default=True, type=ast.literal_eval)
        parser.add_argument('--use_mat', default=True, type=ast.literal_eval)
        parser.add_argument('--train_ratio', default=0.8, type=float)
        parser.add_argument('--alignment_epo', default=1, type=int)
        parser.add_argument('--norm', default='max-min-global', type=str)
        parser.add_argument('--alignment', default=True, type=ast.literal_eval)
        parser.add_argument('--seed', default=0, type=int)
        parser.add_argument('--file', default='', type=str)
        params = parser.parse_args()
        return params

    def print(self):
        print("Parameters used in this running are as follows:")
        for item in self.__dict__.items():
            print("%s: %s" % item)
        print()


P = Params()
P.sys_param()
P.print()
