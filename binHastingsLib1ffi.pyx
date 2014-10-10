import scipy.misc as _sm
import scipy.stats as _ss
import numpy as _N
cimport numpy as _N

dDTYPE = _N.double
ctypedef _N.double_t dDTYPE_t
iDTYPE = _N.int
ctypedef _N.int_t iDTYPE_t


cdef extern from "math.h":
    double exp(double)
    double log(double)
"""
cdef extern from "gsl/gsl_rng.h":
   ctypedef struct gsl_rng_type:
       pass
   ctypedef struct gsl_rng:
       pass
   gsl_rng_type *gsl_rng_mt19937
   gsl_rng *gsl_rng_alloc(gsl_rng_type * T)
  
cdef gsl_rng *rng = gsl_rng_alloc(gsl_rng_mt19937)

cdef extern from "gsl/gsl_randist.h":
   double poisson "gsl_ran_poisson"(gsl_rng * r,double)
"""

cdef double Llklhds(ks, int n1, double p1):
    return _N.sum(_N.log(_sm.comb(n1, ks)) + ks*log(p1) + (n1-ks)*log(1. - p1))

cdef int trPoi(double lmd, int a):
    """
    a, b inclusive
    """
    cdef int ct = _N.random.poisson(lmd)
    #cdef int ct = int(poisson(rng, lmd))
    while (ct < a):
        ct = _N.random.poisson(lmd)  #  _ss.poisson.rvs is slower
        #ct = int(poisson(rng, lmd))  #  _ss.poisson.rvs is slower
    return ct

#def MCMC(burn, NMC, _N.ndarray[iDTYPE_t, ndim=1] cts, _N.ndarray[iDTYPE_t, ndim=1] ns, _N.ndarray[dDTYPE_t, ndim=1] ps, order):
def MCMC(int burn, int NMC, cts, _N.ndarray[iDTYPE_t, ndim=1] ns, _N.ndarray[dDTYPE_t, ndim=1] ps, int order):
#def MCMC(burn, NMC, cts, ns, ps, order):
    cdef double p0, p1, pu
    cdef int    n0, n1
    pMin=0.01
    pMax=0.99
    cdef double Mk  = _N.mean(cts)
    iMk100 = int(Mk/pMin)   #  
    sdk = _N.std(cts)
    cv = ((sdk*sdk) / Mk)
    cdef int nmin= _N.max(cts)

    stdp = 0.02
    istdp= 1./stdp
    stdp2= stdp**2
    istdp2= 1/stdp2

    p0    = 0.2   #  for neural data, this seems a reasonable choice
    n0    = int(Mk / p0)

    cdef _N.ndarray[dDTYPE_t, ndim=1] lFlB = _N.empty(2, dtype=dDTYPE)
    cdef _N.ndarray[iDTYPE_t, ndim=1] n1n0 = _N.empty(2, dtype=iDTYPE)

    cdef _N.ndarray[dDTYPE_t, ndim=1] rds  = _N.empty(burn+NMC, dtype=dDTYPE)
    rds[:]  = _N.random.rand(burn+NMC)

    cdef int mL      = 5000
    cdef _N.ndarray[dDTYPE_t, ndim=1] pcdlog  = _N.empty(mL, dtype=dDTYPE)
    pcdlog[1:mL] = _N.log(_N.arange(1, mL))
    print "cv is %.3f" % cv

    lFlB[1] = Llklhds(cts, n0, p0)
    cdef int it
    cdef double lmd
    cdef double lFlBdf
    cdef double lPR, lnPR, lpPR
    cdef double posRat, prRat, aln

    for it in xrange(burn + NMC):
        if order == 1:
            lmd= Mk/p0
            n1 = trPoi(lmd, nmin)   #  mean is p0/Mk
            pu = Mk/n1
            p1 = pu + stdp*_ss.truncnorm.rvs((pMin-pu)*istdp, (pMax-pu)*istdp)
        else:  #  why is behavior
            pu = Mk/n0
            p1 = pu + stdp*_ss.truncnorm.rvs((pMin-pu)*istdp, (pMax-pu)*istdp)
            lmd= Mk/p1
            n1 = trPoi(lmd, nmin)   #  mean is p0/Mk

        lFlB[0] = Llklhds(cts, n1, p1)

        n1n0[0] = n1;                   n1n0[1] = n0
        # lpPR   p proposal ratio
        lpPR = 0.5*(-((p1 - pu)*(p1 - pu)) + ((p0 - pu)*(p0 - pu)))*istdp2
        if n0 == n1:
            lnPR = 0  #  n0 == n1
        else:
            if n0 > n1:
                lnPR = (n1-n0) * log(lmd) + _N.sum(pcdlog[n1+1:n0+1])
            else:
                lnPR = (n1-n0) * log(lmd) - _N.sum(pcdlog[n0+1:n1+1])
        lPR = lnPR + lpPR
        if lPR > 50:
            prRat = 1e+10
        else:
            prRat = exp(lPR)

        lFlBdf = lFlB[0] - lFlB[1]
        posRat = 1e+50 if (lFlBdf > 100.) else exp(lFlBdf)
            
        aln  = min(1., posRat*prRat)

        if rds[it] < aln:
            p0 = p1
            n0 = n1
            lFlB[1] = lFlB[0]
        ps[it] = p0
        ns[it] = n0

