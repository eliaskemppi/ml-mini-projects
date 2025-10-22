import pandas as pd
import networkx as nx
from itertools import combinations
import matplotlib.pyplot as plt

df = pd.read_csv('./data/actorfilms.csv')

min_year = 2000
max_year = 2022

df = df[(df["Year"] >= min_year) & (df["Year"] <= max_year)]
#print(df.head())

G = nx.Graph()

# make a graph where actors are nodes and edges represent co-appearances, weighted by number of co-appearances
for movie_id, group in df.groupby("FilmID"):

    actors = group["ActorID"].tolist()
    for a1, a2 in combinations(actors, 2):
        if G.has_edge(a1, a2):
            G[a1][a2]['weight'] += 1
        else:
            G.add_edge(a1, a2, weight=1)

# compute pagerank
pr = nx.pagerank(G, alpha=0.85, weight="weight")


id_to_name = dict(zip(df["ActorID"], df["Actor"]))

# print top 10 actors by pagerank
top_actors = sorted(pr.items(), key=lambda x: x[1], reverse=True)[:10]
for name, score in top_actors:
    actor_name = id_to_name.get(name, name)


#print(top_actors)
top_actors = sorted(pr.items(), key=lambda x: x[1], reverse=True)[:20]
top_actor_ids = [a for a, _ in top_actors]

subG = G.subgraph(top_actor_ids)

import numpy as np

# Get PageRank values for nodes in subgraph
scores = np.array([pr[node] for node in subG.nodes()])

# normalize to 0–1 range
scores_norm = (scores - scores.min()) / (scores.max() - scores.min())

# Scale to a visible size range (e.g. 50–2550)
node_sizes = 50 + scores_norm * 2500


import matplotlib.pyplot as plt

plt.figure(figsize=(10, 8))

# get positions for all nodes
pos = nx.spring_layout(subG, k=0.3, seed=42)

# draw nodes
nx.draw_networkx_nodes(
    subG, pos,
    node_size=node_sizes,
    node_color="skyblue",
    alpha=0.8
)

# draw edges with weights
edge_widths = [subG[u][v]["weight"] for u, v in subG.edges()]
nx.draw_networkx_edges(
    subG, pos,
    width=edge_widths,
    alpha=0.4
)

# draw actor names
nx.draw_networkx_labels(
    subG, pos,
    labels={node: id_to_name.get(node, node) for node in subG.nodes()},
    font_size=8
)

plt.axis("off")
plt.show()
