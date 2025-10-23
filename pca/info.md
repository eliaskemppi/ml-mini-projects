# Principal Component Analysis (PCA) — From Scratch & Applied to Songs

## Overview

Mini-project demonstrating Principal Component Analysis (PCA) in two parts:

- Manual PCA implementation using NumPy to understand the math step by step.

- Applied PCA on a large Spotify songs dataset to visualize relationships between songs in a reduced-dimensional space.

## Part 1 — PCA from Scratch

#### Method

Implement PCA by hand:

- Define a small 3D dataset

- Standardize features (zero mean, unit variance)

- Compute covariance matrix

- Perform eigenvalue decomposition

- Sort eigenvalues and eigenvectors by descending variance

- Project data onto top 2 principal components

#### Visualization

- 2D scatter plot showing how PCA projects the original 3D data onto its main variance directions.

#### Insights

- PCA finds orthogonal axes of maximum variance

- Demonstrates the geometric meaning of covariance and eigenvectors

## Part 2 — PCA on Songs

#### Dataset

Spotify Audio Features (April 2019)

Each song represented by numeric audio features (e.g., energy, danceability, tempo, etc.)

#### Method

- Remove metadata (artist, track names, etc.)

- Standardize all numeric features

- Apply PCA (k = 3) to reduce high-dimensional feature space

- Manually select 10 diverse songs to visualize variability in PCA space

#### Visualization

- Interactive 3D Plotly scatter plot

- Each point = one song

- Display artist name next to the point

- Axes = top 3 principal components

#### Insights

- Songs that are musically similar (based on audio features) appear closer in PCA space

- PCA helps uncover relationships between different songs

- Demonstrates how dimensionality reduction can aid in exploratory analysis and visualization