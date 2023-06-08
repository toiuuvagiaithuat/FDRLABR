import pickle
import matplotlib.pyplot as plt
import numpy as np
from numpy import arange, array, random
import pickle
from typing import Dict, Any, List, Type


# def plot(pickle_file_name):
#     bitrate_list = pickle.load(open(pickle_file_name, 'rb'))
#     plt.figure(figsize=(15, 3))
#     plt.plot(bitrate_list)
#     plt.xlabel('time')
#     plt.ylabel('bitrate')
#     plt.title(pickle_file_name)
#     plt.show()


def markov_bw(p, nrow, ht, filename):
    # ht = [0.5, 1, 2, 3, 4, 5, 6, 8]
    sigma = np.random.uniform(0.05, 0.5)

    mbw=np.empty((0, 400), float)
    for j in range(nrow):
        bw=np.array([])
        ind = random.choice(len(ht) - 1)
        for j in range(400):
            rnum = np.random.uniform(0,1)
            if (rnum <= p*1/3):
                ind = min(ind+1, len(ht)-1)
            elif ((rnum > p*1/3) & (rnum <= p*2/3)):
                ind = max(ind-1, 0)
            elif ((rnum > p*2/3) & (rnum <= p*(2/3+1/6))):
                ind = min(ind+2, len(ht)-1)
            elif ((rnum > p*(2/3+1/6)) & (rnum <= p)):
                ind = max(ind - 2, 0)
            else:
                ind = ind

            bw = np.append(bw, np.random.normal(ht[ind], sigma) * 10 ** 6)
        mbw = np.append(mbw, [bw], axis=0)

    pickle.dump(mbw, open(filename, 'wb'))

        # plt.plot(bw)
        # plt.show()
    return mbw



def synthetic_bw(nrow, mu, file_name='synthetic'):
    bw=np.empty((0, 400), float)
    for i in range(nrow):
        sigma = np.random.uniform(0.05, 0.5)  # generate a random variance
        bw = np.append(bw, [np.random.normal(mu, sigma, 400) * 10 ** 6], axis=0)
    pickle.dump(bw, open(file_name, 'wb'))
  # plot(file_name)


if __name__ == '__main__':
    # markov_bw(p=0.5, nrow=50, ht=[0.5, 1, 2], filename='markov1')
    # markov_bw(p=0.5, nrow=50, ht=[2, 3, 4], filename='markov2')
    # markov_bw(p=0.5, nrow=50, ht=[4, 5, 6], filename='markov3')
    # markov_bw(p=0.5, nrow=50, ht=[6, 7, 8], filename='markov4')

    # synthetic_bw(nrow = 50, mu=1, file_name='synthetic_mean1_l50')
    # synthetic_bw(nrow = 50, mu=2, file_name='synthetic_mean2_l50')
    # synthetic_bw(nrow = 50, mu=3, file_name='synthetic_mean3_l50')
    # synthetic_bw(nrow = 50, mu=4, file_name='synthetic_mean4_l50')
    # synthetic_bw(nrow = 50, mu=5, file_name='synthetic_mean5_l50')
    # synthetic_bw(nrow = 50, mu=6, file_name='synthetic_mean6_l50')
    # synthetic_bw(nrow = 50, mu=7, file_name='synthetic_mean7_l50')
    # synthetic_bw(nrow = 50, mu=8, file_name='synthetic_mean8_l50')

    synthetic_bw(nrow = 50, mu=1, file_name='synthetic_mean1_l50')
    synthetic_bw(nrow = 50, mu=2, file_name='synthetic_mean2_l50')
    synthetic_bw(nrow = 50, mu=3, file_name='synthetic_mean3_l50')
    synthetic_bw(nrow = 50, mu=4, file_name='synthetic_mean4_l50')
    synthetic_bw(nrow = 50, mu=5, file_name='synthetic_mean5_l50')
    synthetic_bw(nrow = 50, mu=6, file_name='synthetic_mean6_l50')
    synthetic_bw(nrow = 50, mu=7, file_name='synthetic_mean7_l50')
    synthetic_bw(nrow = 50, mu=8, file_name='synthetic_mean8_l50')