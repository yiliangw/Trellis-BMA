import numpy as np

def marker_encode(seq, markers, positions):
    res = ''
    lastp = 0
    for m, p in zip(markers, positions):
        res += seq[lastp:p]
        res += m
        lastp = p
    res += seq[lastp:]
    return res

def marker_decode(sample, ori_len, markers, positions):
    return ''