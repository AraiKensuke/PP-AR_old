import scipy.misc as _sm
import scipy.stats as _ss
import numpy as _N

import warnings
warnings.filterwarnings("error")

def Llklhds(ks, n1, p1):
    return _N.sum(_N.log(_sm.comb(n1, ks)) + ks*_N.log(p1) + (n1-ks)*_N.log(1 - p1))

def trPoi(lmd, a):
    """
    a, b inclusive
    """
    ct = _N.random.poisson(lmd)
    while (ct < a):
        ct = _N.random.poisson(lmd)
    return ct

def MCMC(burn, NMC, cts, ns, us, order, pcdlog):
    Mk  = _N.mean(cts)
    iMk = 1./Mk
    sdk = _N.std(cts)
    cv = ((sdk*sdk) * iMk)
    nmin= _N.max(cts)

    stdu = 0.1
    stdu2= stdu**2
    istdu2= 1./ stdu2

    p0   = 0.2
    u0   = -_N.log(1/p0 - 1)   #  generate initial u0.  sample u's.
    n0    = int(Mk / p0)

    lFlB = _N.empty(2)
    n1n0 = _N.empty(2)

    rds  = _N.random.rand(burn+NMC)
    rdns = _N.random.randn(burn+NMC)

    print "cv is %.3f" % cv

    lFlB[1] = Llklhds(cts, n0, p0)
    rngs   = _N.arange(0, 20000)
    for it in xrange(burn + NMC):
        #############  n jump
        """
        uu  = -_N.log(n0 * iMk - 1) # mean of proposal density
        u1 = uu + stdu * rdns[it]
        p1 = 1 / (1 + _N.exp(-u1))
        lmd= Mk/p1
        n1 = trPoi(lmd, nmin)   #  mean is p0/Mk
        """
        #  if we pick p first, we can sometimes get large values of n
        #  which break Llklhds calc?  _N.comb with large n prob. problematic
        lmd= Mk/p0
        n1 = trPoi(lmd, nmin)   #  mean is p0/Mk
        uu  = -_N.log(n1 * iMk - 1) # mean of proposal density
        u1 = uu + stdu * rdns[it]
        p1 = 1 / (1 + _N.exp(-u1))

        #  based on n1, we pick

        lFlB[0] = Llklhds(cts, n1, p1)

        n1n0[0] = n1;                   n1n0[1] = n0
        lpPR = 0.5*istdu2*(-((u1 - uu)*(u1 - uu)) + ((u0 - uu)*(u0 - uu)))
        if n0 == n1:
            lnPR = 0  #  n0 == n1
        else:
            if n0 > n1: # range(n1+1, n0+1)
                #lnPR = (n1-n0) * _N.log(lmd) + _N.sum(pcdlog[range(n1+1, n0+1)])
                lnPR = (n1-n0) * _N.log(lmd) + _N.sum(pcdlog[rngs[n1+1:n0+1]])
            else:
                #lnPR = (n1-n0) * _N.log(lmd) - _N.sum(pcdlog[range(n0+1, n1+1)])
                lnPR = (n1-n0) * _N.log(lmd) - _N.sum(pcdlog[rngs[n0+1:n1+1]])
        lPR = lnPR + lpPR
        if lPR > 50:
            prRat = 1e+10
        else:
            prRat = _N.exp(lPR)

            #try:
        posRat = 1e+50 if (lFlB[0] - lFlB[1] > 100) else _N.exp(lFlB[0]-lFlB[1])

        #aln  = min(1, posRat*prRat)
        rat  = posRat*prRat
        aln  = rat if (rat < 1)  else 1

        if rds[it] < aln:
            u0 = u1
            n0 = n1
            p0 = p1
            lFlB[1] = lFlB[0]
        us[it] = u0
        ns[it] = n0
