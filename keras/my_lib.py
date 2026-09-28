import numpy as np

def split_x(dataset, size):
    x = []
    y = []
    for i in range(len(dataset)-size-1):
        subset = dataset[i:(i+size+1)]
        x.append(subset[:size])
        y.append(subset[size:])
    return np.array(x), np.array(y)