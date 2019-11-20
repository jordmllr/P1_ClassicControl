import torch
import numpy as np

def to_sparse(x):
    '''
        converts dense tensor x to sparse format
        perhaps used in future iteration
    '''
    x_typename = torch.typename(x).split('.')[-1]
    sparse_tensortype = getattr(torch.sparse, x_typename)

    indices = torch.nonzero(x)
    if len(indices.shape) == 0:  # if all elements are zeros
        return sparse_tensortype(*x.shape)
    indices = indices.t()
    values = x[tuple(indices[i] for i in range(indices.shape[0]))]
    return sparse_tensortype(indices, values, x.size())

class genotype:
    '''
        stores the genetic information for SNN agents and the methods
        to turn the genotype into a phenotype.
        The genotype contains all information for constructing the
        network. Every network has 3 populations of neurons:
        1. Sensory - Input neurons for observing the environment
        2. Interneurons - interior processing neurons
        3. Motor - spiking output is mapped to action space
        each of these populations can be connected to each other and themselves.
        We are interested in how structure emerges through evolution, so we make
        minimal assumptions to begin with. Thus, in the most general case, we have
        6 adjacency matrices to represent: S-S, S-I, S-M, I-I, I-M, M-M.
        We make an assumption that these matrices will be sparse. This may turn out
        to not be the case, but it has considerable theoretical backing and if it
        turns out to be true, will save the memory cost of each genotype.
    '''

    def __init__(self, sz=None, adj_mat=None, p_max=None):
        if sz is None:
            self.generate_random(p_max)
        else:
            self.generate(sz, adj_mat)

    def generate(self, sz, adj_mat):
        self.sz = sz
        self.adj_mat = adj_mat

    def generate_random(self, pop_max):
        '''
            creates a random genotype.
            arg populations: dictionary of size
        '''
        p = np.random.rand()
        self.sz = np.random.randint(1, pop_max, 3)
        self.sz[2]=2
        self.sz_total = np.sum(self.sz)
        self.adj_mat = torch.FloatTensor(np.random.choice(2,(self.sz_total,self.sz_total),[p,1-p]))
        self.w = self.adj_mat

    def express(self):
        '''
            method for converting the encoded genotype
            into an agent for evalutation
        '''
        s_idx = self.sz[0]
        i_idx = self.sz[0]+self.sz[1]
        m_idx = self.sz[0]+self.sz[1]+self.sz[2]
        c_dict = {'SS': self.w[0:s_idx,0:s_idx], 'SI': self.w[0:s_idx,s_idx:i_idx], 'SM': self.w[0:s_idx,i_idx:m_idx],
                  'IS': self.w[s_idx:i_idx,0:s_idx], 'II': self.w[s_idx:i_idx,s_idx:i_idx], 'IM': self.w[s_idx:i_idx,i_idx:m_idx],
                  'MS': self.w[i_idx:m_idx,0:s_idx], 'MI': self.w[i_idx:m_idx,s_idx:i_idx], 'MM': self.w[i_idx:m_idx,i_idx:m_idx]}
        return c_dict