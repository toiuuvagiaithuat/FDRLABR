import pickle
import matplotlib.pyplot as plt
import numpy as np
from numpy import arange, array, random
import pickle


def plot(pickle_file_name):
    bitrate_list = pickle.load(open(pickle_file_name, 'rb'))
    plt.figure(figsize=(15, 3))
    plt.plot(bitrate_list)
    plt.xlabel('time')
    plt.ylabel('bitrate')
    plt.title(pickle_file_name)
    plt.show()


def markov_bw(file_name='markovian_bw'):
    C = np.arange(0, 10, 0.25) * 10 ** 6
    cur_bitrate = random.choice(C)
    bitrate_list = [cur_bitrate]

    for j in range(2000):
        i = random.rand()
        if i < 0.5:
            cur_bitrate = random.choice(C)
        bitrate_list.append(cur_bitrate)

    pickle.dump(bitrate_list, open(file_name, 'wb'))
    plot(file_name)


def synthetic_bw(mu, sigma=0.1, file_name='synthetic'):
  bw = np.random.normal(mu, sigma, 2000) * 10 ** 6
  pickle.dump(bw, open(file_name, 'wb'))
  plot(file_name)


if __name__ == '__main__':
    # markov_bw()
    # synthetic_bw(mu=2.0, sigma=0.1, file_name='synthetic_mean20')
    # synthetic_bw(mu=1.5, sigma=0.1, file_name='synthetic_mean15')
    # synthetic_bw(mu=1.0, sigma=0.1, file_name='synthetic_mean10')
    # synthetic_bw(mu=1.6, sigma=0.1, file_name='synthetic_mean16')
    # synthetic_bw(mu=1.2, sigma=0.1, file_name='synthetic_mean12')
    # synthetic_bw(mu=0.8, sigma=0.1, file_name='synthetic_mean08')
    synthetic_bw(mu=0.5, sigma=0.1, file_name='synthetic_mean05')
    # synthetic_bw(mu=0.4, sigma=0.1, file_name='synthetic_mean04')
    # plot('bw/bitrate_list')
