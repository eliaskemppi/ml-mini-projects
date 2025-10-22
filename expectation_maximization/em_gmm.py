### Example, GMM
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# --- Step 1: Generate data ---
np.random.seed(42)
data = np.concatenate([
    np.random.normal(-1, 2, 300),
    np.random.normal(3, 0.5, 100),
    np.random.normal(5, 1, 250)
])

#print(data)
data = np.array(data)

# --- Step 2: Initialize parameters ---
pi1, pi2, pi3 = 0.5, 0.3, 0.2  # mixing weights
mu1, mu2, mu3 = -1, 4, 3
sigma1, sigma2, sigma3 = 1, 2, 1

# --- Step 3: Run EM algorithm loop ---

for iter in range(35):
    # E-step:
    pdf1 = norm.pdf(data, loc=mu1, scale=sigma1) #likelihood per class and datapoint
    pdf2 = norm.pdf(data, loc=mu2, scale=sigma2)
    pdf3 = norm.pdf(data, loc=mu3, scale=sigma3)

    gamma1 = pi1 * pdf1
    gamma2 = pi2 * pdf2
    gamma3 = pi3 * pdf3

    denom = gamma1 + gamma2 + gamma3 
    #normalized likelihoods
    w1 = gamma1 / denom
    w2 = gamma2 / denom
    w3 = gamma3 / denom


    # M-step:
    N = len(data)
    Nk1 = np.sum(w1)
    Nk2 = np.sum(w2)
    Nk3 = np.sum(w3)

    pi1 = Nk1 / N
    pi2 = Nk2 / N
    pi3 = Nk3 / N

    mu1 = np.sum(w1 * data) / Nk1
    mu2 = np.sum(w2 * data) / Nk2
    mu3 = np.sum(w3 * data) / Nk3

    var1 = np.sum(w1 * (data - mu1)**2) / Nk1
    var2 = np.sum(w2 * (data - mu2)**2) / Nk2
    var3 = np.sum(w3 * (data - mu3)**2) / Nk3

    sigma1 = np.sqrt(var1)
    sigma2 = np.sqrt(var2)
    sigma3 = np.sqrt(var3)

    # print progress
    if iter == 0 or iter == 34:
        print("pi1: ", pi1, "pi2: ", pi2, "pi3: ", pi3)
        print("mu1:", mu1, "mu2:", mu2, "mu3:", mu3)
        print("sig1:", sigma1, "sig2:", sigma2, "sig3:", sigma3)

    if iter == 0:
        history = []
    history.append(((pi1, pi2, pi3), (mu1, mu2, mu3), (sigma1, sigma2, sigma3)))

    
# Visualize

from matplotlib.animation import FuncAnimation

fig, ax = plt.subplots(figsize=(8, 5))
x = np.linspace(-6, 10, 400)
bins = np.linspace(-6, 10, 30)
ax.hist(data, bins=bins, density=True, alpha=0.4, color='gray')

lines = [ax.plot([], [], lw=2)[0] for _ in range(3)]
title = ax.text(0.5, 1.02, "", transform=ax.transAxes, ha="center")

def update(frame):
    (pis, mus, sigmas) = history[frame]
    for k, line in enumerate(lines):
        line.set_data(x, pis[k] * norm.pdf(x, mus[k], sigmas[k]))
    title.set_text(f"Iteration {frame+1}")
    return lines + [title]

ani = FuncAnimation(fig, update, frames=len(history), blit=True, interval=300, repeat=False)
plt.show()

ani.save("./visuals/em_gmm.gif", writer="pillow", fps=3)