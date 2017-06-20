import numpy as _N

def entropy(X, bins=None):
    """
    entropy of the probability distribution
    """
    _p, bins = _N.histogram(X, bins=bins)
    p  = _N.array(_p, dtype=_N.float)
    p /= _N.sum(p)
    H = -p*_N.log2(p)
    inds = _N.where(p==0)[0]
    H[inds] = 0
    return _N.sum(H)

def MI(X1, X2, x1bins=None, x2bins=None):
    """
    entropy of the probability distribution
    """
    jphHst, binsX1, binsX2 = _N.histogram2d(X1, X2, bins=[x1bins, x2bins])

    x1Bs = x1bins.shape[0] - 1
    x2Bs = x2bins.shape[0] - 1
    #  marginal distributions
    p1 = _N.sum(jphHst, axis=0)
    p2 = _N.sum(jphHst, axis=1)
    p1 /= _N.sum(p1)
    p2 /= _N.sum(p2)

    #  reshape so we can sum over X1 and X2
    p2 = p2.reshape(p2.shape[0], 1)
    p1 = p1.reshape(1, p1.shape[0])
    jphHst /= _N.sum(jphHst)
    ############  Find 
    arr2d = jphHst * _N.log2(jphHst/(p1*p2))
    nanInds = _N.isnan(arr2d)

    x, y = _N.where(nanInds == True)
    for i in xrange(len(x)):
        arr2d[x[i], y[i]] = 0

    return _N.sum(arr2d), jphHst
