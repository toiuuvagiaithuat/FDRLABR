import matplotlib.pyplot as plt
import numpy as np
from numpy import arange, array, random
import pickle

import numpy as np

synthetic100u = []
sigma = 0.1 * 10 ** 6  # bps
group_mean = np.array([1, 2, 3, 4, 5]) * 10 ** 6  # bps
bw_len = 1000  # seconds
user_per_group = 20

for mean in group_mean:
    for user in range(0, user_per_group):
        bw = np.random.normal(mean, sigma, bw_len)
        synthetic100u.append(bw)

synthetic100u = np.asarray(synthetic100u)
print(synthetic100u.shape)

with open('synthetic100u.pickle', 'wb') as f:
    pickle.dump(synthetic100u, f)


