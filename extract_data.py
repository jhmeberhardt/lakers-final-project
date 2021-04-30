import numpy as np

def pull_data_iphone(file_name):
    f = open(file_name + '.csv')
    f.readline()
    xs = []
    timestamps = []
    for line in f:
        value = line.split(',')
        timestamps.append(float(value[1]))
        xs.append(float(value[len(value)-1]))
    return np.array(xs), np.array(timestamps)