import numpy as np

def split_x(dataset, size):
    aaa = []
    for i in range(len(dataset)-size+1):
        subset = dataset[i:(i+size)]
        aaa.append(subset)
    return np.array(aaa)