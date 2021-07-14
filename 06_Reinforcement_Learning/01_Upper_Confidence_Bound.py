# UCB - Upper Confidence Bound

# Import the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# Import the dataset
dir_path = os.path.dirname(os.path.realpath(__file__))
data_path = dir_path + ('data/' if dir_path.endswith('/') else '/data/')
dataset = pd.read_csv(data_path + 'Ads_CTR_Optimisation.csv')

# Implementing UCB
import math
N = 10000 # try with less values to check if the model can find the best ad quickly
d = 10
ads_selected = []
selections = [0] * d
rewards = [0] * d
total_reward = 0
for n in range(0, N):
    ad = 0
    max_upper_bound = 0
    for i in range(0, d):
        if (selections[i] > 0):
            average_reward = rewards[i] / selections[i]
            delta_i = math.sqrt(3/2 * math.log(n + 1) / selections[i])
            upper_bound = average_reward + delta_i
        else:
            upper_bound = 1e400
        if upper_bound > max_upper_bound:
            max_upper_bound = upper_bound
            ad = i
    ads_selected.append(ad)
    selections[ad] += 1
    reward = dataset.values[n, ad]
    rewards[ad] += reward
    total_reward += reward

# Visualising the results
plt.hist(ads_selected)
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()

