import pickle
import os

import gensim
import networkx as nx
from gensim.models import Word2Vec

from models.gmodel import filter_graph


class N2Vmodel:
    def __init__(self, embed_fn='data/node2vec_dict'):
        self.model = gensim.models.KeyedVectors.load_word2vec_format(embed_fn)

    def __call__(self, query_isbn, ret_k=5, ret_scores=False, fill_to_k=False):
        outs = self.model.most_similar(query_isbn, topn=ret_k)
        ret = [isbn for isbn, _ in outs]
        if ret_scores:
            ret_sc = [score for _, score in outs]
            return ret, ret_sc
        return ret


if __name__ == '__main__':
    import logging
    import node2vec.src.node2vec as n2v

    # Generate random walks and train w2v model
    graph_path = 'data/corat_graph_18-04-2020_1655'
    G = nx.read_gpickle(graph_path)
    Gf = filter_graph(G, minimal_corats=3, sigma_mul=0.1, robdist_clip=80)

    for u, v, d in Gf.edges(data=True):
        Gf.edges[u, v]['weight'] = d['closeness']

    G = n2v.Graph(Gf, is_directed=False, p=0.8, q=1.5)
    G.preprocess_transition_probs()

    if not os.path.exists('n2v_save'):
        os.makedirs('n2v_save')
    if not os.path.exists('data'):
        os.makedirs('data')

    with open('n2v_save/n2v_graph_preproc', 'wb') as f:
        pickle.dump(G, f)

    walks = G.simulate_walks(num_walks=50, walk_length=80)
    walks = [list(map(str, walk)) for walk in walks]

    with open('n2v_save/n2v_walks_preproc', 'wb') as f:
        pickle.dump(walks, f)

    logging.basicConfig(level=logging.INFO)

    model = Word2Vec(walks, size=128, window=10, min_count=0, sg=1, workers=4, iter=15)
    model.wv.save_word2vec_format('data/node2vec_dict')
    model.wv.save('n2v_save/node2vec_model')
