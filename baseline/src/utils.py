import numpy as np


def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3


def count_lists(l):
    c_m = -1
    c_um = -1
    c_list = np.zeros(len(l))
    for i, e in enumerate(l):
        if l[i] == 0:
            c_um = c_um+1
            c_list[i] = c_um
        else:
            c_m = c_m+1
            c_list[i] = c_m
    return c_list


def performance(ndarr, ratio = 0.1):
    ndarr_sorted = np.sort(ndarr)
    ndarr_trimed = ndarr_sorted[int(len(ndarr)*ratio):int(len(ndarr)*(1-ratio))]
    return ndarr_trimed.mean(), ndarr_trimed.std()