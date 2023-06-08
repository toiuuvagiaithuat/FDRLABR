import numpy as np
import pickle


def get_fcc_test_data():
    fcc_test = np.asarray(pickle.load(open("./data/bw/fcc_test100kb.pickle", "rb")))
    fcc_test = np.repeat(fcc_test, 10, axis=1)
    return fcc_test

def get_lte_test_data():
    lte_test = np.asarray(pickle.load(open("./data/bw/LTE_test100kb.pickle", "rb")))
    return lte_test

def get_fcc_train_data():
    fcc_train = np.asarray(pickle.load(open("./data/bw/fcc_train100kb.pickle", "rb")))
    fcc_train = np.repeat(fcc_train, 10, axis=1)
    return fcc_train

def get_lte_train_data():
    lte_train = np.asarray(pickle.load(open("./data/bw/LTE_train100kb.pickle", "rb")))
    return lte_train

def get_lte_up40():
    return np.asarray(pickle.load(open("./data/bw/LTEUp40Train.pickle", "rb")))

def get_lte_low40():
    return np.asarray(pickle.load(open("./data/bw/LTELow40Train.pickle", "rb")))

def get_fcc_up40():
    fcc_up40 = np.asarray(pickle.load(open("./data/bw/FCCUp40Train.pickle", "rb")))
    fcc_up40 = np.repeat(fcc_up40, 10, axis=1)
    return fcc_up40

def get_fcc_low40():
    fcc_low40 = np.asarray(pickle.load(open("./data/bw/FCCLow40Train.pickle", "rb")))
    fcc_low40 = np.repeat(fcc_low40, 10, axis=1)
    return fcc_low40

def get_real_train_data_updown40():
    real_train1 = np.asarray(pickle.load(open("./data/bw/FCCUp40Train.pickle", "rb")))
    real_train1 = np.repeat(real_train1, 10, axis=1)

    real_train2 = np.asarray(pickle.load(open("./data/bw/FCCLow40Train.pickle", "rb")))
    real_train2 = np.repeat(real_train2, 10, axis=1)

    real_train3 = np.asarray(pickle.load(open("./data/bw/LTEUp40Train.pickle", "rb")))
    real_train4 = np.asarray(pickle.load(open("./data/bw/LTELow40Train.pickle", "rb")))
    real_train = np.concatenate((real_train1, real_train2, real_train3, real_train4), axis=0)
    return real_train

def get_real_valid_data_updown40():
    real_valid1 = np.asarray(pickle.load(open("./data/bw/FCCUp40Test.pickle", "rb")))
    real_valid1 = np.repeat(real_valid1, 10, axis=1)
    real_valid1 = real_valid1[:300]

    real_valid2 = np.asarray(pickle.load(open("./data/bw/FCCLow40Test.pickle", "rb")))
    real_valid2 = np.repeat(real_valid2, 10, axis=1)
    real_valid2 = real_valid2[:300]

    real_valid3 = np.asarray(pickle.load(open("./data/bw/LTEUp40Test.pickle", "rb")))
    real_valid3 = real_valid3[:100]

    real_valid4 = np.asarray(pickle.load(open("./data/bw/LTELow40Test.pickle", "rb")))
    real_valid4 = real_valid4[:100]

    real_validate = np.concatenate((real_valid1, real_valid2, real_valid3, real_valid4), axis=0)
    return real_validate

def get_real_valid_data_updown40():
    real_valid1 = np.asarray(pickle.load(open("./data/bw/FCCUp40Test.pickle", "rb")))
    real_valid1 = np.repeat(real_valid1, 10, axis=1)
    real_valid1 = real_valid1[:50]

    real_valid2 = np.asarray(pickle.load(open("./data/bw/FCCLow40Test.pickle", "rb")))
    real_valid2 = np.repeat(real_valid2, 10, axis=1)
    real_valid2 = real_valid2[:50]

    real_valid3 = np.asarray(pickle.load(open("./data/bw/LTEUp40Test.pickle", "rb")))
    real_valid3 = real_valid3[:50]

    real_valid4 = np.asarray(pickle.load(open("./data/bw/LTELow40Test.pickle", "rb")))
    real_valid4 = real_valid4[:50]

    real_validate = np.concatenate((real_valid1, real_valid2, real_valid3, real_valid4), axis=0)
    return real_validate

def get_real_test800_data_updown40():
    real_test1 = np.asarray(pickle.load(open("./data/bw/FCCUp40Test.pickle", "rb")))
    real_test1 = np.repeat(real_test1, 10, axis=1)
    real_test1 = real_test1[:200]

    real_test2 = np.asarray(pickle.load(open("./data/bw/FCCLow40Test.pickle", "rb")))
    real_test2 = np.repeat(real_test2, 10, axis=1)
    real_test2 = real_test2[:200]

    real_test3 = np.asarray(pickle.load(open("./data/bw/LTEUp40Test.pickle", "rb")))
    real_test3 = real_test3[:200]

    real_test4 = np.asarray(pickle.load(open("./data/bw/LTELow40Test.pickle", "rb")))
    real_test4 = real_test4[:200]

    real_test = np.concatenate((real_test1, real_test2, real_test3, real_test4), axis=0)
    return real_test

def get_real_test1600_data_updown40():
    real_test1 = np.asarray(pickle.load(open("./data/bw/FCCUp40Test.pickle", "rb")))
    real_test1 = np.repeat(real_test1, 10, axis=1)
    # real_test1 = real_test1[-300:]

    real_test2 = np.asarray(pickle.load(open("./data/bw/FCCLow40Test.pickle", "rb")))
    real_test2 = np.repeat(real_test2, 10, axis=1)
    # real_test2 = real_test2[-300:]

    real_test3 = np.asarray(pickle.load(open("./data/bw/LTEUp40Test.pickle", "rb")))
    # real_test3 = real_test3[-100:]

    real_test4 = np.asarray(pickle.load(open("./data/bw/LTELow40Test.pickle", "rb")))
    # real_test4 = real_test4[-100:]

    real_test = np.concatenate((real_test1, real_test2, real_test3, real_test4), axis=0)
    return real_test

# train mix real bandwidth
def get_real_training_data():
    fcc_train = np.asarray(pickle.load(open("./data/bw/fcc_train100kb.pickle", "rb")))
    fcc_train = np.repeat(fcc_train, 10, axis=1)
    lte_train = np.asarray(pickle.load(open("./data/bw/LTE_train100kb.pickle", "rb")))
    train = np.concatenate((fcc_train, lte_train), axis=0)
    return train

# test mix real bandwidth, FCC + LTE
def get_real_test_data():
    fcc_test = np.asarray(pickle.load(open("./data/bw/fcc_test100kb.pickle", "rb")))
    fcc_test = np.repeat(fcc_test, 10, axis=1)
    lte_test = np.asarray(pickle.load(open("./data/bw/LTE_test100kb.pickle", "rb")))
    test = np.concatenate((fcc_test, lte_test), axis=0)
    return test