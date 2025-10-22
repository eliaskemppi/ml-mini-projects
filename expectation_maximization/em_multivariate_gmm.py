### Multivariate GMM

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from matplotlib.animation import FuncAnimation

# Step 1: Generate data

mu1_real, mu2_real, mu3_real = [0, 0], [1, 3], [-2, 2]
cov1_real, cov2_real, cov3_real = [[1, 0.3], [0.3, 1]], [[1, -0.7], [-0.7, 1]], [[1, 0], [0, 0.5]] 
np.random.seed(42)
data = np.vstack([
    np.random.multivariate_normal(mu1_real, cov1_real, 300),
    np.random.multivariate_normal(mu2_real, cov2_real, 100),
    np.random.multivariate_normal(mu3_real, cov3_real, 250)
])

#print(data)
data = np.array(data)

# Step 2: Initialize parameters

N, d = data.shape # N = n datapoints, d = data variables
K = 3 # n clusters

#print(data.shape)

pi = np.ones(K) / K # shape: K
mu = data[np.random.choice(N, K, replace=False)] # shape: K, d
Sigma = [np.eye(d) for _ in range(K)]

# Step 3: Run EM algorithm loop

# Store history for animation
history = []

for iter in range(50):
    # E-step:
    pdfs = np.array([ #likelihood per class and datapoint
        pi[k] * multivariate_normal.pdf(data, mean=mu[k], cov=Sigma[k])
        for k in range(K)
    ]).T

    gammas = pdfs / (pdfs.sum(axis=1, keepdims=True)) # normalized class likelihoods per datapoint


    # M-step: update parameters
    Nk = gammas.sum(axis=0) # estimate of number of datapoints per class
    pi = Nk / N
    mu = (gammas.T @ data) / Nk[:, np.newaxis] # shape K, d

    Sigma = []
    for k in range(K):
        diff = data - mu[k]
        cov = (gammas[:, k][:, np.newaxis] * diff).T @ diff / Nk[k]
        Sigma.append(cov)

    history.append((mu.copy(), Sigma.copy()))

# Step 4: Visualize

### Animation
fig, ax = plt.subplots(figsize=(7, 6))
x, y = np.mgrid[-4:8:.05, -4:8:.05]
pos = np.dstack((x, y))
colors = ['r', 'b', 'g']

def update(frame):
    ax.clear()
    ax.scatter(data[:, 0], data[:, 1], s=10, alpha=0.4)
    mu, Sigma = history[frame]
    for k in range(K):
        pdf = multivariate_normal(mu[k], Sigma[k]).pdf(pos)
        ax.contour(x, y, pdf, colors=colors[k])
    ax.set_title(f"Iteration {frame + 1}")
    ax.set_xlim(data[:,0].min()-1, data[:,0].max()+1)
    ax.set_ylim(data[:,1].min()-1, data[:,1].max()+1)
    return ax.collections

ani = FuncAnimation(fig, update, frames=len(history), interval=400, repeat=True)
plt.show()

ani.save("./visuals/em_multivariate_gmm.gif", writer="pillow", fps=5, dpi=150)