import numpy as np

def rdist(dis_l, avg):
    rl = []
    for _ in range(len(dis_l)):
        rl.append(np.sqrt(np.abs(np.sum(np.square(avg - dis_l[_])))))
    return rl

def dis_avg(lis):
    return np.average(np.concatenate(tuple(lis)),axis=0).reshape(1,128)

def rshap(lst):
    for _ in range(len(lst)):
        lst[_] = lst[_].reshape(1,128)
    return lst

def relative_dist(d1, d2):
    return np.sqrt(np.abs(np.sum(np.square(d1 - d2))))
