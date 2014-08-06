import numpy as _N

def LL(B, *args):
    x        = args[1]
    y        = args[2]

    Bxi      = _N.dot(B, x.T)
    emBxi    = _N.exp(Bxi)   #  x is N x k, B is k-dim
#    LL       = _N.dot(y, Bxi) - _N.sum(_N.log(1 + emBxi))
    LLtbt    = y*Bxi - _N.log(1 + emBxi)
    LL       = _N.sum(LLtbt)
    return -LL

def jacb(B, *args):
    k        = args[0]
    x        = args[1]
    y        = args[2]

    mdrv      = _N.empty(k)
    Bxi      = _N.dot(B, x.T)
    emBxi    = _N.exp(Bxi)
    pn       = emBxi / (1 + emBxi)
    for j in xrange(k):
        #mdrv[j] = _N.dot(x[:, j], y - Bo1pB)
        tbt      = x[:, j]*(y - pn)
        mdrv[j] = _N.sum(tbt)
    return -mdrv

#################  the versions with abs refractory periods
def LLr(B, *args):
    x        = args[1]
    y        = args[2]
    lrn      = args[3]

    Bxi      = _N.dot(B, x.T)
    emBxi    = _N.exp(Bxi)   #  x is N x k, B is k-dim
    pn       = emBxi / (1 + emBxi)
    loglrn   = _N.log(lrn)
    logpn    = _N.log(pn)
    log1mpn  = _N.log(1 - lrn*pn)
    LL       = _N.sum(y * (loglrn + logpn) + (1 - y)*log1mpn)
    return -LL

def LLr2(B, *args):
    x        = args[1]
    y        = args[2]
    lrn      = args[3]
    offs= _N.log(lrn / (1 + (1 - lrn)*_N.exp(_N.dot(B, x.T))))

    Bxia     = _N.dot(B, x.T) + offs
    emBxia   = _N.exp(Bxia)   #  x is N x k, B is k-dim
    pn       = emBxia / (1 + emBxia)
    LLtbt    = y*Bxia - _N.log(1 + emBxia)
    LL       = _N.sum(LLtbt)
    return -LL


    return -LL

def build_lrn(N, y, absRf):
    lrn = _N.ones(N)
    for n in xrange(N):
        if y[n] == 1:
            untl = (n + absRf + 1) if (n + absRf + 1 < N) else N   # untl
            for m in xrange(n + 1, untl):
                lrn[m] = 0.0001

    return lrn
