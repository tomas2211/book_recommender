import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from tqdm import tqdm
import time
from data_utils import load_ratings


def graph_addedge_cumulative(u, v, w, G):
    if G.has_edge(u, v):
        G[u][v]['weight'] += w
        G[u][v]['weight_sq'] += w ** 2  # to compute std
        G[u][v]['n'] += 1
    else:
        G.add_edge(u, v, weight_sq=w ** 2, weight=w, n=1)


if __name__ == "__main__":
    GRAPHGEN_MAX_USER_NRAT = 1000

    ratings, isbns_filtered, test_useridx, train_useridx = load_ratings()

    # Filter ratings to explicit and by isbns_filtered
    rat_explicit = ratings[ratings['Book-Rating'] != 0]
    cnts = rat_explicit['ISBN'].value_counts()
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

            for idx, x in cart_p.iterrows():
                graph_addedge_cumulative(x['ISBN_x'], x['ISBN_y'], x['Book-Rating_x'] * x['Book-Rating_y'], G)
                pbar.update(1)

    nx.write_gpickle(G, 'data/corat_graph_%s' % time.strftime('%d-%m-%Y_%H%M'))




print('Unfiltered edges: %d' % len(G.edges))
edges_ns = []

for u, v, d in G.edges(data=True):
    edges_ns.append(d['n'])

edges_ns = np.array(edges_ns)
edges_ns.sort()
plt.hist(edges_ns, bins=100)
plt.ylim([0, 200000])
plt.show()

# Filter graph

Gf = nx.Graph()
Gf.add_nodes_from(G.nodes)

for u, v, d in tqdm(G.edges(data=True)):
    n = d['n'] / 2
    w = d['weight'] / d['n']
    if d['n'] > 4:
        nweight = w / 100  # range 0.01 to 1
        std = np.sqrt(d['weight_sq'] / d['n'] - w ** 2)
        nstd = std / 100
        nweight_inv = 1 / (nweight - 1 * nstd)
        rob_dist = np.exp(nweight_inv).clip(0, 40)
        Gf.add_edge(u, v, weight=w, n=n, nweight=nweight, nweight_inv=nweight_inv, rob_dist=rob_dist,
                    nstd=nstd)

nodes_by_deg = [n for n, _ in sorted(Gf.degree, key=lambda x: x[1], reverse=True)]

edges_ns = []
edges_dists = []
edges_nstds = []
for u, v, d in Gf.edges(data=True):
    edges_ns.append(d['n'])
    edges_dists.append(d['rob_dist'])
    edges_nstds.append(d['nstd'])

edges_ns = np.array(edges_ns)
edges_ns.sort()
plt.hist(edges_ns, bins=100)
plt.ylim([0, 10000])
plt.show()
print('Filtered edges: %d' % len(Gf.edges))

cc_sizes = np.array([len(cc) for cc in nx.connected_components(Gf)])
cc_sizes[::-1].sort()
print('Filtered conn. components: %d' % len(cc_sizes))
print('Number of single books: %d' % np.sum(cc_sizes == 1))

edges_dists = np.array(edges_dists)
plt.hist(edges_dists, bins=100)
plt.title('Edge costs histogram')
plt.xlabel('Edge cost')
plt.ylabel('# of edges')
plt.show()

edges_nstds = np.array(edges_nstds)
plt.hist(edges_nstds, bins=100)
plt.title('Edge std cost histogram')
plt.xlabel('Edge std')
plt.ylabel('# of edges')
plt.show()

q_isbn = '0316666343'  # popular
q_isbn = '044023722X'
q_isbn = '0684853159'
q_isbn = '0385335466'

cutoff = 20
niter = 1
K = 5000

# Exponentially increasing cutoff to get at least K samples
dists, _ = nx.single_source_dijkstra(Gf, q_isbn, cutoff=cutoff, weight='rob_dist')
while (len(dists) - 1) < K and niter < 10:
    cutoff *= 1.3
    dists, _ = nx.single_source_dijkstra(Gf, q_isbn, cutoff=cutoff, weight='rob_dist')
    niter += 1

del dists[q_isbn]

if len(dists) < K:
    missing = K - len(dists)
    for n in nodes_by_deg[:missing]:
        dists[n] = 100

ret = [k for k in sorted(dists, key=dists.get)]
ret_dists = [dists[k] for k in ret]
