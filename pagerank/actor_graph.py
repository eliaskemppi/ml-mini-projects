import streamlit as st
import pandas as pd
import networkx as nx
from itertools import combinations
import matplotlib.pyplot as plt
import numpy as np


st.title("Actor Co-appearance Network (PageRank Explorer)")

# load data with caching
@st.cache_data
def load_data():
    return pd.read_csv("./data/actorfilms.csv") #https://www.kaggle.com/datasets/darinhawley/imdb-films-by-actor-for-10k-actors/data

df = load_data()

# sidebar for filters
st.sidebar.header("Filters")
min_year = int(df["Year"].min())
max_year = int(df["Year"].max())

year_range = st.sidebar.slider(
    "Select Year Range",
    min_value=min_year,
    max_value=max_year,
    value=(2000, 2022),
    step=1
)

n = st.sidebar.slider(
    "Number of Top Actors to Display",
    min_value=5,
    max_value=40,
    value=20,
    step=1
)

show_button = st.sidebar.button("Generate Graph")

# filter data
df_filtered = df[(df["Year"] >= year_range[0]) & (df["Year"] <= year_range[1])]
st.write(f"### Graph based on movies from {year_range[0]}-{year_range[1]}")
#st.write(f"Number of rows: {len(df_filtered)}")

if show_button:
    # build graph
    G = nx.Graph()
    for movie_id, group in df_filtered.groupby("FilmID"):
        actors = group["ActorID"].tolist()
        for a1, a2 in combinations(actors, 2):
            if G.has_edge(a1, a2):
                G[a1][a2]["weight"] += 1
            else:
                G.add_edge(a1, a2, weight=1)

    # compute PageRank
    pr = nx.pagerank(G, alpha=0.85, weight="weight")

    # map ActorID to Actor name
    id_to_name = dict(zip(df_filtered["ActorID"], df_filtered["Actor"]))

    # get top N actors by PageRank
    top_actors = sorted(pr.items(), key=lambda x: x[1], reverse=True)[:n]
    top_actor_ids = [a for a, _ in top_actors]

    # get subgraph of top actors (for visualization)
    subG = G.subgraph(top_actor_ids)

    # get node sizes based on PageRank (visualization)
    scores = np.array([pr[node] for node in subG.nodes()])
    scores_norm = (scores - scores.min()) / (scores.max() - scores.min())
    node_sizes = 100 + scores_norm * 3000  # visible range

    # __ draw graph __
    fig, ax = plt.subplots(figsize=(10, 8))
    pos = nx.spring_layout(subG, k=0.3, seed=42)

    nx.draw_networkx_nodes(subG, pos,
        node_size=node_sizes,
        node_color="skyblue",
        alpha=0.8,
        ax=ax
    )

    edge_widths = [subG[u][v]["weight"] for u, v in subG.edges()]
    nx.draw_networkx_edges(subG, pos,
        width=edge_widths,
        alpha=0.4,
        ax=ax
    )

    nx.draw_networkx_labels(
        subG, pos,
        labels={node: id_to_name.get(node, node) for node in subG.nodes()},
        font_size=8,
        ax=ax
    )

    ax.set_title(f"Actor Co-appearance Graph (Top {n} by PageRank)")
    ax.axis("off")

    st.pyplot(fig)

    # show top actors with scores
    st.write(f"### Top {n} Actors by PageRank")
    for aid, score in top_actors[:n]:
        st.write(f"**{id_to_name.get(aid, aid)}**  | PR: {score:.5f}")
