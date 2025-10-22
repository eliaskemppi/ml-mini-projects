import numpy as np

### Example 1

# --- Step 1: Generate synthetic data ---
np.random.seed(42)
p1 = 0.5
p2 = 0.65
n_trials = 10

coin_choices = np.random.choice(['1', '2'], size=n_trials)

data = []
for coin in coin_choices:
    if coin == '1':
        tosses = np.random.binomial(1, p1, size=15)
    else:
        tosses = np.random.binomial(1, p2, size=15)
    data.append(tosses)
data = np.array(data)

#print(data)

#--- Step 2: Initialize parameters ---
p_A, p_B = 0.5, 0.6  # initial theta

# --- Step 3: Run EM algorithm loop ---
for iteration in range(10):

    # E-step:
    # 1) Compute likelihoods
    # P(data|theta) = ∏ p^x * (1-p)^(1-x)
    likelihood_A = [np.prod(p_A**x * (1-p_A)**(1-x)) for x in data] 
    likelihood_B = [np.prod(p_B**x * (1-p_B)**(1-x)) for x in data]

    # 2) Compute weights
    w_A = np.array(likelihood_A) / (np.array(likelihood_A) + np.array(likelihood_B)) # probability of coin A given data
    w_B = np.array(likelihood_B) / (np.array(likelihood_A) + np.array(likelihood_B))

    # M-step:
    # Update theta
    p_A = np.sum(w_A * np.sum(data, axis=1)) / (np.sum(w_A) * data.shape[1]) 
    p_B = np.sum(w_B * np.sum(data, axis=1)) / (np.sum(w_B) * data.shape[1])

    print(f"Iteration {iteration+1}: p_A={p_A:.3f}, p_B={p_B:.3f}")

print("\nTrue values: p_1=0.5, p_2=0.65")

