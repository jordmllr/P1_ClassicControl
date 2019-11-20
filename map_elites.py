import numpy as np
import random
import torch
from agent import agent

class Feature_Space:
    '''
        Descretized phenotypic feature space
    '''
    def __init__(self):
        self.bins = {}

    def add_feature(self, name, low, high, d):
        '''
        :param name: name of feature dimension
        :param low: lower bound for bins
        :param high: upper bound for bins
        :param d: number of bins
        '''
        bins = np.linspace(low, high, d)
        self.bins[name] = bins

    def get_cell_index(self, b):
        '''
        :param b: dictionary - keys(feature names),values(raw feature value)
        :return: tuple - cell index for given value, b
        '''
        index = [np.digitize(b[ft], self.bins[ft]) for ft in b.keys()]
        return tuple(index)

class map_elites:
    '''
        feature_space: Feature_Space object that allows archive to interface with feature map
        archive: dictionary of valid solutions - keys(tuple: index in feature space), values(tuple: performance, genotype)
    '''
    def __init__(self):
        self.feature_space = Feature_Space()
        self.archive = {}

    def run(self, G, I, env):
        self.env = env
        #   initialize population with G random genotypes
        print('Generating Population...')
        for g in range(G):
            print('Generation: ', g)
            x = self.random_x()
            self.update_archive(x)
        print('Evolution Beginning...')
        for i in range(I):
            print('Epoch: ', i)
            x = self.selection()
            x_new = self.variation(x)
            self.update_archive(x_new)

    def update_archive(self, x):
        p,b = x.evaluate(self.env)
        index = self.feature_space.get_cell_index(b)
        if index not in self.archive or p > self.archive[index][0]:
            self.archive[index] = (p, x)

    def random_x(self):
        #   generate a random agent
        x = agent()
        return x

    def selection(self):
        key, entry = random.choice(list(self.archive.items()))
        return entry[1]

    def variation(self, x):
        '''
        random mutation of genome with p=0.9 probability of remaining the same
        '''
        p=0.9
        #   create a mask of muatation targets
        mutation_mask = torch.BoolTensor(np.random.choice(2,(x.genotype.sz_total,x.genotype.sz_total),[p,1-p]))
        #   flip selected mutation targets
        mutated_vals = x.genotype.adj_mat[mutation_mask]
        mutated_vals = torch.FloatTensor([bool(val)^bool(1) for val in mutated_vals])
        x.genotype.adj_mat[mutation_mask] = mutated_vals
        return x