import numpy as np
#from scipy.linalg import eigh
from matplotlib import pyplot as plt


points = np.array([
    [0, 0, 0], 
    [1, 0.5, 0.3], 
    [2, 1, 1], 
    [1.5, -1, -1],
    [2.5, -1.5, -2], 
    [3.5, -2, 0]
])

# Scaling features

mu = np.mean(points, axis=0)
sigma = np.std(points, axis=0)
scaled = (points-mu)/sigma

#print(scaled)

# Compute covariance matrix
n = len(points)
cov = 1/(n-1)*(scaled.T @ scaled)
#print(cov)

# Get the k vectors corresponding with the largest eigenvalues
k = 2
eigenvalues, eigenvectors = np.linalg.eigh(cov)
idxs = np.argsort(eigenvalues)[::-1][:k]

k_vectors = eigenvectors[:, idxs]
#print(k_vectors)

# Get projected data
pca_data = scaled @ k_vectors

# Visualize 
x = pca_data[:, 0]
y = pca_data[:, 1]

plt.scatter(x, y)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA Projection')
plt.show()