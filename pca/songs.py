import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Read the data and drop unnecessary columns
df = pd.read_csv('./data/SpotifyAudioFeaturesApril2019.csv')


df.drop(columns=['artist_name', 'track_id', 'track_name'], inplace=True)

# Scale data
sc = StandardScaler()
scaled = sc.fit_transform(df)
df = pd.DataFrame(scaled)

# Do PCA
pca = PCA(3)
songs_pca = pca.fit_transform(df)


# print(songs_pca.shape)

# Some example songs that I manually picked. The goal was to get a lot of variability
indices = [124886, 30764, 51752, 123905, 123604, 98817, 78579, 123994, 115492, 4469]
subset = songs_pca[indices]

x, y, z = subset[:, 0], subset[:, 1], subset[:, 2]

# Visualizing
import plotly.graph_objects as go

titles = ["Queen", "Sibelius", "Rachmaninoff", "Drake", "Ed Sheeran", "Rammstein", "Lady Gaga", "5SOS", "Zac Efron", "SYML"]

fig = go.Figure(data=[go.Scatter3d(
    x=x, y=y, z=z,
    mode='markers+text',
    text=titles,
    hovertemplate="%{text}",
    marker=dict(size=6, color='skyblue')
)])

fig.update_layout(
    scene=dict(
        xaxis_title='PC1',
        yaxis_title='PC2',
        zaxis_title='PC3'
    ),
    title="PCA Visualization of Songs"
)

fig.show()
