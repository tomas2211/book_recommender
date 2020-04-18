import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from tqdm import tqdm
import time
from data_utils import load_ratings
import os
from gmodel import filter_graph
import node2vec.src.node2vec as n2v
from gensim.models import Word2Vec
import pickle
import gensim

class N2Vmodel:
    def __init__(self, embed_fn='node2vec_model'):
        self.model = gensim.models.KeyedVectors.load_word2vec_format(embed_fn)

    def __call__(self, query_isbn, ret_k=5, ret_scores=False, fill_to_k=False):
        outs = self.model.most_similar(query_isbn, topn=ret_k)
        ret = [isbn for isbn, _ in outs]
        if ret_scores:
            ret_sc = [score for _, score in outs]
            return ret, ret_sc
        return ret


if __name__=='__main__':
    graph_path='data/corat_graph_18-04-2020_1655'
    sigma_mul=1.0
    robdist_clip=40
    minimal_corats=2
    G = nx.read_gpickle(graph_path)

    Gf = filter_graph(G, minimal_corats=2, sigma_mul=0.1, robdist_clip=80)

    for u, v, d in Gf.edges(data=True):
        Gf.edges[u,v]['weight'] = d['corat']


    G = n2v.Graph(Gf, is_directed=False, p=1, q=10)
    G.preprocess_transition_probs()

    with open('n2v_graph_preproc', 'wb') as f:
        pickle.dump(G, f)


    walks = G.simulate_walks(num_walks=100, walk_length=80)
    walks = [list(map(str, walk)) for walk in walks]

    with open('n2v_walks_preproc', 'wb') as f:
        pickle.dump(walks, f)

    import logging
    logging.basicConfig(level=logging.INFO)

    model = Word2Vec(walks, size=128, window=10, min_count=0, sg=1, workers=4, iter=30)
    model.wv.save_word2vec_format('node2vec_model', fvocab='n2v_vocab')

