## EM Algorithm (Expectation–Maximization)

The EM algorithm is used to estimate parameters of models that depend on latent variables. I use EM for Gaussian Mixture Models (GMMs), the idea is that the data comes from several Gaussian distributions, but we don’t know which point belongs to which Gaussian. We only know the number of clusters and that the data comes Gaussian distributions.

## EM alternates between:

Start with an initial guess for every parameter.

**E-step:** Estimate the probability that each data point belongs to each Gaussian.

**M-step:** Update the parameters (in the case of GMMS: means, variances, and mixing weights) based on those probabilities.

Repeating these steps makes the model gradually fit the data better.

## Common uses for EM:

- Clustering and density estimation (GMMs)

- Image segmentation

- Speech recognition

- Handling missing data or incomplete observations