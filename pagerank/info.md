# PageRank Visualization on Actor Co-Appearance Network

## Overview

Mini-project demonstrating PageRank on a network of actors who co-appeared in movies

Goal: identify the most “influential” actors based on collaboration patterns

Built with Python, NetworkX, and Streamlit for visualization

## Method

Nodes = actors

Edges = co-appearance in the same movie

Applied PageRank algorithm to compute actor importance

Higher PageRank → higher score to actors connected to other **highly** ranked actors

## Visualization

Interactive Streamlit app

Node size proportional to PageRank score

Edge width proportional to number of co-appearances

The number of nodes per visualization and the year range tunable.

## Insights

Reveals collaboration clusters and key connector actors, for example Samuel L. Jackson was the top PageRank score of the whole dataset.

Demonstrates how graph algorithms can uncover hidden influence structures


Run with: streamlit run actor_graph.py