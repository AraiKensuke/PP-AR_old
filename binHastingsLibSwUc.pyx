import scipy.misc as _sm
import scipy.stats as _ss
import numpy as _N
cimport numpy as _N

cdef extern from "math.h":
    double log(double)
    double exp(double)
import warnings
warnings.filterwarnings("error")

dDTYPE = _N.double
ctypedef _N.double_t dDTYPE_t
iDTYPE = _N.int
ctypedef _N.int_t iDTYPE_t

__BNML__ = 0   # binomial
__NBML__ = 1   # negative binomial

cdef Llklhds(int type, _N.ndarray[iDTYPE_t, ndim=1] ks, int rn1, double p1):
    if type == __BNML__:
        return _N.sum(_N.log(_sm.comb(rn1, ks)) + ks*_N.log(p1) + (rn1-ks)*_N.log(1 - p1))
    elif type == __NBML__:
        return _N.sum(_N.log(_sm.comb(ks + rn1-1, ks)) + ks*_N.log(p1) + rn1*_N.log(1 - p1))

def trPoi(lmd, a):
    """
    a, b inclusive
    """
    ct = _N.random.poisson(lmd)
    while (ct < a):
        ct = _N.random.poisson(lmd)
    return ct

def MCMC(burn, NMC, cts, rns, us, dty, pcdlog, double p0=0.05, int dist=__BNML__):
    Mk  = _N.mean(cts)
    cdef double iMk = 1./Mk
    sdk = _N.std(cts)
    cv = ((sdk*sdk) * iMk)
    nmin= _N.max(cts)   #  if n too small, can't generate data
    rmin= 1

    stdu = 0.1
    stdu2= stdu**2
    istdu2= 1./ stdu2

    #  let's start it off from binomial
    cdef double p1
        
    cdef double u0   = -log(1/p0 - 1)   #  generate initial u0.  sample u's.
    cdef int rn0    = int(Mk / p0)
    if dist == __BNML__:  #  make sure initial distribution is capable of generating data
        while rn0 < nmin:
            rn0 += 1
    else:
        while rn0 < rmin:
            rn0 += 1

    pTH  = 0.03
    uTH    = log(pTH / (1 - pTH))

    lFlB = _N.empty(2, dtype=_N.double)
    rn1rn0 = _N.empty(2, dtype=_N.int)

    rds  = _N.random.rand(burn+NMC)
    rdns = _N.random.randn(burn+NMC)

    print "cv is %.3f" % cv

    cdef int todist
    lFlB[1] = Llklhds(dist, cts, rn0, p0)
    rngs   = _N.arange(0, 20000)

    cross  = False

    cdef double u1, lmd0, lmd1, lmd
    cdef int rn1
    cdef int it
    for it from 0 <= it < burn + NMC:
        if dist == __BNML__:
            uu1  = -log(rn0 * iMk - 1.) # mean of proposal density
            u1 = uu1 + stdu * rdns[it]

            if u1 > uTH:       ###########   Stay in Binomial ##########
                todist = __BNML__;    cross  = False
                p1 = 1 / (1 + exp(-u1))
                lmd= Mk/p1
                rn1 = trPoi(lmd, nmin)   #  mean is p0/Mk
                bLargeP = (p0 > 0.3) and (p1 > 0.3)
                if bLargeP:#    fairly large p.  Exact proposal ratio
                    lmd = Mk/(0.5*(p0+p1))
                else:          #  small p.  prop. ratio far from lower lim of n
                    lmd1= Mk/p1;     lmd0= Mk/p0;     lmd = lmd1
                uu0  = -log(rn1 * iMk - 1) # mean of proposal density
                # log of probability
                lpPR = 0.5*istdu2*(-((u1 - uu1)*(u1 - uu1)) + ((u0 - uu0)*(u0 - uu0)))  #  - (lnc0 - lnc1), lnc reciprocal of norm
            else:   ########  Switch to __NBML__  ####################
                todist = __NBML__;   cross  = True
                u1 = 2*uTH - u1  #  u1 now a parameter of NB distribution   
                p1 = 1 / (1 + exp(-u1))
                lmd = (1./p1 - 1)*Mk
                rn1 = trPoi(lmd, rmin)   #  mean is p0/Mk
                lmd1= Mk*((1-p1)/p1);  lmd0= Mk/p0;   lmd= lmd1
                uu0  = -log(rn1 * iMk) # mean of proposal density
                lpPR = 0.5*istdu2*(-(((uTH-u1) - uu1)*((uTH-u1) - uu1)) + (((uTH-u0) - uu0)*((uTH-u0) - uu0)))
        elif dist == __NBML__:
            uu1  = -log(rn0 * iMk) # mean of proposal density
            u1 = uu1 + stdu * rdns[it]
            if u1 > uTH:       ######   Stay in Negative binomial ######
                todist = __NBML__;    cross  = False
                p1 = 1 / (1 + exp(-u1))
                lmd = (1./p1 - 1)*Mk
                rn1 = trPoi(lmd, rmin)   #  mean is p0/Mk
                bLargeP = (p0 > 0.3) and (p1 > 0.3)
                if bLargeP:#    fairly large p.  Exact proposal ratio
                    lmd= Mk*((1-0.5*(p0+p1))/(0.5*(p0+p1)))
                else:          #  small p.  prop. ratio far from lower lim of n
                    lmd1= Mk*((1-p1)/p1);  lmd0= Mk*((1-p0)/p0);   lmd= lmd1
                uu0  = -log(rn1 * iMk) # mean of proposal density
                # log of probability
                lpPR = 0.5*istdu2*(-((u1 - uu1)*(u1 - uu1)) + ((u0 - uu0)*(u0 - uu0)))
            else:   ########  Switch to __BNML__  ####################
                todist = __BNML__;    cross  = True
                u1 = 2*uTH - u1  #  u in NB distribution
                p1 = 1 / (1 + exp(-u1))
                lmd = Mk/p1
                rn1 = trPoi(lmd, nmin)   #  mean is p0/Mk
                lmd1= Mk/p1;     lmd0= Mk*((1-p0)/p0);     lmd = lmd1
                uu0  = -log(rn1 * iMk - 1) # mean of proposal density
                lpPR = 0.5*istdu2*(-(((uTH-u1) - uu1)*((uTH-u1) - uu1)) + (((uTH-u0) - uu0)*((uTH-u0) - uu0)))
        
        lFlB[0] = Llklhds(todist, cts, rn1, p1)
        rn1rn0[0] = rn1;                   rn1rn0[1] = rn0

        ########  log of proposal probabilities

        if cross or not bLargeP:  #  lnPR can be calculated without regard to whether cross or not, because it is conditionally dependent on pN, pB
            #print "in cross or not bLargeP  %d" % it
            if rn0 == rn1:
                lnPR = rn1*(log(lmd1) - log(lmd0)) - (lmd1 - lmd0) 
            else:
                if rn0 > rn1: # range(n1+1, n0+1)
                    lnPR = rn1*log(lmd1) - rn0*log(lmd0) + _N.sum(pcdlog[rngs[rn1+1:rn0+1]]) - (lmd1 - lmd0)
                else:
                    lnPR = rn1*log(lmd1) - rn0*log(lmd0) - _N.sum(pcdlog[rngs[rn0+1:rn1+1]]) - (lmd1 - lmd0)
        else:
            if rn0 == rn1:
                lnPR = 0  #  r0 == r1
            else:
                if rn0 > rn1: # range(r1+1, r0+1)
                    lnPR = (rn1-rn0) * log(lmd) + _N.sum(pcdlog[rngs[rn1+1:rn0+1]])
                else:
                    lnPR = (rn1-rn0) * log(lmd) - _N.sum(pcdlog[rngs[rn0+1:rn1+1]])

        lPR = lnPR + lpPR

        if lPR > 50:
            prRat = 1e+10
        else:
            prRat = exp(lPR)
                
        posRat = 1e+50 if (lFlB[0] - lFlB[1] > 100) else exp(lFlB[0]-lFlB[1])

        rat  = posRat*prRat


        aln  = rat if (rat < 1)  else 1
        #print "%(it)d   %(aln).5f   posRat %(posRat).3f    lnPR %(lnPR).3f   lpPR %(lpPR).3f" % {"it" : it, "aln" : aln, "posRat" : posRat, "lnPR" : lnPR, "lpPR" : lpPR}
        if rds[it] < aln:   #  accept
            u0 = u1
            rn0 = rn1
            p0 = p1
            lFlB[1] = lFlB[0]
            #print "accepted  %d" % it
            dist = todist
        dty[it] = dist
        us[it] = u0
        rns[it] = rn0    #  rn0 is the newly sampled value if accepted
