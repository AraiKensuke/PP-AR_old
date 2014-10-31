import scipy.misc as _sm
import scipy.stats as _ss
import numpy as _N

import warnings
warnings.filterwarnings("error")

def Llklhds(ks, r1, p1):
    return _N.sum(_N.log(_sm.comb(ks + r1-1, ks)) + ks*_N.log(p1) + r1*_N.log(1 - p1))

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
    nmin= _N.min(cts)

    stdu = 0.1
    stdu2= stdu**2
    istdu2= 1./ stdu2

    p0   = 0.2
    u0   = -_N.log(1/p0)   #  generate initial u0.  sample u's.
    r0    = int(Mk * ((1-p0)/p0))

    lFlB = _N.empty(2)
    r1r0 = _N.empty(2)

    rds  = _N.random.rand(burn+NMC)
    rdns = _N.random.randn(burn+NMC)

    print "cv is %.3f" % cv

    lFlB[1] = Llklhds(cts, r0, p0)
    rngs   = _N.arange(0, 20000)

    bLargeP  = False    #  which proposal density to use for n
    for it in xrange(burn + NMC):
        #############  n jump
        uu  = -_N.log(r0 * iMk) # mean of proposal density
        u1 = uu + stdu * rdns[it]
        p1 = 1 / (1 + _N.exp(-u1))
        bLargeP = (p0 > 0.3) and (p1 > 0.3)
        if bLargeP:#    fairly large p.  Exact proposal ratio
            lmd= Mk*((1-0.5*(p0+p1))/(0.5*(p0+p1)))
        else:          #  fairly small p.  prop. ratio far from lower lim of n
            lmd1= Mk*((1-p1)/p1);  lmd0= Mk*((1-p0)/p0);   lmd= lmd1
        r1 = trPoi(lmd, nmin)   #  mean is p0/Mk

        #  if we pick p first, we can sometimes get large values of n
        #  which break Llklhds calc?  _N.comb with large n prob. problematic
        """
        lmd= Mk/p0
        r1 = trPoi(lmd, nmin)   #  mean is p0/Mk
        uu  = -_N.log(r1 * iMk - 1) # mean of proposal density
        u1 = uu + stdu * rdns[it]
        p1 = 1 / (1 + _N.exp(-u1))
        """

        #  based on r1, we pick

        lFlB[0] = Llklhds(cts, r1, p1)

        r1r0[0] = r1;                   r1r0[1] = r0
        lpPR = 0.5*istdu2*(-((u1 - uu)*(u1 - uu)) + ((u0 - uu)*(u0 - uu)))
        if not bLargeP:
             if r0 == r1:
                 lnPR = r1*(_N.log(lmd1) - _N.log(lmd0)) - (lmd1 - lmd0)  #  n0 == n1
             else:
                 if r0 > r1: # range(n1+1, n0+1)
                     lnPR = r1*_N.log(lmd1) - r0*_N.log(lmd0) + _N.sum(pcdlog[rngs[r1+1:r0+1]]) - (lmd1 - lmd0)
                 else:
                     lnPR = r1*_N.log(lmd1) - r0*_N.log(lmd0) - _N.sum(pcdlog[rngs[r0+1:r1+1]]) - (lmd1 - lmd0)
        else:
            if r0 == r1:
                lnPR = 0  #  r0 == r1
            else:
                if r0 > r1: # range(r1+1, r0+1)
                    lnPR = (r1-r0) * _N.log(lmd) + _N.sum(pcdlog[rngs[r1+1:r0+1]])
                else:
                    lnPR = (r1-r0) * _N.log(lmd) - _N.sum(pcdlog[rngs[r0+1:r1+1]])
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
            r0 = r1
            p0 = p1
            lFlB[1] = lFlB[0]
        us[it] = u0
        ns[it] = r0

