import numpy as _N

def spkISI(spk01s):
    M = spk01s.shape[0]

    isis = []
    for m in xrange(M):
        sts = _N.where(spk01s[m] == 1)[0]
        isis.extend(_N.diff(sts))

    return _N.std(isis) / _N.mean(isis)
