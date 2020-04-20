import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from tqdm import tqdm
import time
from data_utils import load_ratings
import os


class Gmodel():
    def __init__(self, graph_path='data/corat_graph_18-04-2020_1655', sigma_mul=1.0, robdist_clip=40, minimal_corats=3):
        G = nx.read_gpickle(graph_path)
        self.Gf = filter_graph(G, minimal_corats=minimal_corats, sigma_mul=sigma_mul, robdist_clip=robdist_clip)
        self.nodes_by_deg = \
            [n for n, _ in sorted(self.Gf.degree, key=lambda x: x[1], reverse=True)]  # sort nodes by degree

    def __call__(self, query_isbn, ret_k=5, verbose=False, ret_scores=False, fill_to_k=False):
        cutoff = 20
        niter = 1
        # Exponentially increasing cutoff to get at least K samples
        dists, _ = nx.single_source_dijkstra(self.Gf, query_isbn, cutoff=cutoff, weight='rob_dist')
        while (len(dists) - 1) < ret_k and niter < 10:
            cutoff *= 1.3
            dists, _ = nx.single_source_dijkstra(self.Gf, query_isbn, cutoff=cutoff, weight='rob_dist')
            niter += 1

        del dists[query_isbn]  # remove the query isbn

        if fill_to_k and len(dists) < ret_k:  # if still missing, fill with sorted nodes
            missing = ret_k - len(dists)
            for n in self.nodes_by_deg[:missing]:
                dists[n] = 100

        ret = [k for k in sorted(dists, key=dists.get)][:ret_k]
        if ret_scores:
            ret_dists = [dists[k] for k in ret][:ret_k]
            return ret, ret_dists
        return ret


def graph_addedge_cumulative(u, v, w, G):
    if G.has_edge(u, v):
        G[u][v]['weight'] += w
        G[u][v]['weight_sq'] += w ** 2  # to compute std
        G[u][v]['n'] += 1
    else:
        G.add_edge(u, v, weight_sq=w ** 2, weight=w, n=1)


def filter_edge(d, sigma_mul=1.0):
    corat = d['weight'] / d['n']  # mean corating
    ncorat = corat / 100  # normalize - range 0.01 to 1

    corat_std = np.sqrt(d['weight_sq'] / d['n'] - corat ** 2)  # calc corating std
    ncorat_std = corat_std / 100

    # calculate robust distance
    closeness = ncorat - sigma_mul * ncorat_std  # divisor
    if closeness < 0.01:  # too close to zero or negative
        closeness = 0.01

    rob_dist = 1 / closeness
    return {
        'n': d['n'] / 2,
        'corat': corat, 'ncorat': ncorat,
        'corat_std': corat_std, 'ncorat_std': ncorat_std,
        'rob_dist': rob_dist, 'closeness': closeness
    }


def filter_graph(G, minimal_corats=3, sigma_mul=0.1, robdist_clip=40, small_component=4, small_comp_add=3):
    Gf = nx.Graph()
    Gf.add_nodes_from(G.nodes)

    for u, v, d in tqdm(G.edges(data=True), desc='Filtering edges'):  # process information from edges
        n = d['n'] / 2  # number of coratings
        if n >= minimal_corats:
            edge_dat = filter_edge(d, sigma_mul)
            if edge_dat['rob_dist'] < robdist_clip:
                Gf.add_edge(u, v, **edge_dat)

    for cc in tqdm(nx.connected_components(Gf), desc='Connecting small components'):  # connect all small components
        if len(cc) <= small_component:
            for n in cc:
                neigh = [(nbr, filter_edge(G[n][nbr], sigma_mul)) for nbr in G[n]]
                neigh.sort(key=lambda x: x[1]['rob_dist'])
                for v, edge_dat in neigh[:small_comp_add]:
                    if edge_dat['rob_dist'] < robdist_clip:
                        Gf.add_edge(n, v, **edge_dat)

    return Gf


if __name__ == "__main__":
    GRAPHGEN_MAX_USER_NRAT = 1000

    ratings, isbns_filtered, test_useridx, train_useridx = load_ratings()

    # Filter ratings to explicit and by isbns_filtered
    rat_explicit = ratings[ratings['Book-Rating'] != 0]
    rat_exp = rat_explicit[rat_explicit.ISBN.isin(isbns_filtered)]
    rat = ratings[ratings.ISBN.isin(isbns_filtered)]

    # Generate graph with coratings
    G = nx.Graph()
    G.add_nodes_from(isbns_filtered.array)

    # Filter users by number of ratings
    user_ratcnt = rat_exp['User-ID'].value_counts()
    users_sel = user_ratcnt[(user_ratcnt > 1) & (user_ratcnt < GRAPHGEN_MAX_USER_NRAT)]
    users_sel_train = users_sel[users_sel.index.isin(train_useridx)]

    coratings = np.sum((users_sel_train.values ** 2 - users_sel_train.values))
    print('Number of coratings: %d' % (coratings / 2))

    with tqdm(total=coratings, desc='Processing coratings') as pbar:
        for u in users_sel_train.index:
            rat_u = rat_exp[rat_exp['User-ID'] == u]

            cart_p = pd.merge(rat_u, rat_u, on='User-ID')
            cart_p = cart_p[cart_p['ISBN_x'] != cart_p['ISBN_y']]

            for idx, x in cart_p.iterrows():  # could be done much more effectively perhaps...
                graph_addedge_cumulative(x['ISBN_x'], x['ISBN_y'], x['Book-Rating_x'] * x['Book-Rating_y'], G)
                pbar.update(1)

    nx.write_gpickle(G, 'data/corat_graph_%s' % time.strftime('%d-%m-%Y_%H%M'))

    # Stat the graph
    if not os.path.exists('../vis'):
        os.mkdir('../vis')

    SIGMA_MUL = 0.1
    Gf = filter_graph(G, minimal_corats=3, sigma_mul=SIGMA_MUL, robdist_clip=40)  # filter to compute edge costs

    print('Unfiltered edges: %d' % len(G.edges))
    print('Filtered edges: %d' % len(Gf.edges))

    edges_ns = []  # number of coratings for each edge histogram
    for u, v, d in G.edges(data=True):
        edges_ns.append(d['n'])

    edges_ns = np.array(edges_ns)
    plt.hist(edges_ns, bins=100)
    plt.title('Unfiltered graph - number of edge coratings histogram')
    plt.xlabel('Number of coratings')
    plt.ylabel('# of edges')
    plt.tight_layout()
    plt.savefig('vis/gmodel_unfil_coratn.png')
    plt.close()

    # filtered graph
    cc_sizes = np.array([len(cc) for cc in nx.connected_components(Gf)])  # connected components
    cc_sizes[::-1].sort()
    print('Filtered conn. components: %d' % len(cc_sizes))
    print('Number of single books: %d' % np.sum(cc_sizes == 1))

    edges_ns = []  # corating numbers
    edges_dists = []  # edge costs
    edges_nstds = []  # edge coratings std
    edges_close = []
    for u, v, d in Gf.edges(data=True):
        edges_ns.append(d['n'])
        edges_dists.append(d['rob_dist'])
        edges_nstds.append(d['ncorat_std'])
        edges_close.append(d['closeness'])

    edges_ns = np.array(edges_ns)  # number of coratings
    edges_ns.sort()
    plt.hist(edges_ns, bins=100)
    plt.title('Filtered graph - number of edge coratings histogram')
    plt.xlabel('Number of coratings')
    plt.ylabel('# of edges')
    plt.tight_layout()
    plt.savefig('vis/gmodel_fil_coratn.png')
    plt.close()

    edges_dists = np.array(edges_dists)  # edge costs
    edges_dists = edges_dists[edges_dists < 5]
    plt.hist(edges_dists, bins=100)
    plt.title('Edge distance histogram (sigma mul %.2f)' % SIGMA_MUL)
    plt.xlabel('Edge distance')
    plt.ylabel('# of edges')
    plt.tight_layout()
    plt.savefig('vis/gmodel_fil_edgedist.png')
    plt.close()

    edges_close = np.array(edges_close)  # edge costs
    plt.hist(edges_close, bins=100)
    plt.title('Edge closeness histogram (sigma mul %.2f)' % SIGMA_MUL)
    plt.xlabel('Edge closeness')
    plt.ylabel('# of edges')
    plt.tight_layout()
    plt.savefig('vis/gmodel_fil_edgeclose.png')
    plt.close()

    edges_nstds = np.array(edges_nstds)  # std of coratings
    plt.hist(edges_nstds, bins=100)
    plt.title('Edge coratings std histogram')
    plt.xlabel('Corating std')
    plt.ylabel('# of edges')
    plt.tight_layout()
    plt.savefig('vis/gmodel_fil_coratstd.png')
    plt.close()
