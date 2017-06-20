import numpy as _N

def interpret(dtys, rns, us):
    p    = 1/(1 + _N.exp(-us))

    ITRS = dtys.shape[0]
    
    FF   = _N.empty(ITRS)
    mn   = _N.empty(ITRS)

    for it in xrange(ITRS):
        FF[it] = 1-p[it] if (dtys[it] == 1) else 1 / (1 - p[it])
        mn[it] = rns[it]*p[it] if (dtys[it] == 1) else (rns[it]*p[it]) / (1 - p[it])

    return mn, FF
