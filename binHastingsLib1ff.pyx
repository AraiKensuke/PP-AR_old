import scipy.misc as _sm
import scipy.stats as _ss
import numpy as _N

def Llklhds(ks, n1, p1):
    return _N.sum(_N.log(_sm.comb(n1, ks)) + ks*_N.log(p1) + (n1-ks)*_N.log(1 - p1))

def trPoi(lmd, a, b):
    """
    a, b inclusive
    """
    ct = _N.random.poisson(lmd)
    while (ct < a):
        ct = _N.random.poisson(lmd)  #  _ss.poisson.rvs is slower
    return ct

def MCMC(burn, NMC, cts, ns, ps, order):
    pMin=0.01
    pMax=0.99
    Mk  = _N.mean(cts)
    iMk100 = int(Mk/pMin)   #  
    sdk = _N.std(cts)
    cv = ((sdk*sdk) / Mk)
    nmin= _N.max(cts)

    stdp = 0.02
    stdp2= stdp**2

    p0    = 0.2   #  for neural data, this seems a reasonable choice
    n0    = int(Mk / p0)

    lFlB = _N.empty(2)
    n1n0 = _N.empty(2)

    rds  = _N.random.rand(burn+NMC)

    mL      = 5000
    pcdlog  = _N.empty(mL)        #precomputed logs
    pcdlog[1:mL] = _N.log(_N.arange(1, mL))
    print "cv is %.3f" % cv

    lFlB[1] = Llklhds(cts, n0, p0)
    for it in xrange(burn + NMC):
        if order == 1:
            lmd= Mk/p0
            n1 = trPoi(lmd, a=nmin, b=iMk100)   #  mean is p0/Mk
            pu = Mk/n1
            p1 = pu + stdp*_ss.truncnorm.rvs((pMin-pu)/stdp, (pMax-pu)/stdp)
        else:  #  why is behavior
            pu = Mk/n0
            p1 = pu + stdp*_ss.truncnorm.rvs((pMin-pu)/stdp, (pMax-pu)/stdp)
            lmd= Mk/p1
            n1 = trPoi(lmd, a=nmin, b=iMk100)   #  mean is p0/Mk

        lFlB[0] = Llklhds(cts, n1, p1)

        n1n0[0] = n1;                   n1n0[1] = n0
        # lpPR   p proposal ratio
        lpPR = 0.5*(-((p1 - pu)*(p1 - pu)) + ((p0 - pu)*(p0 - pu)))/stdp2
        if n0 == n1:
            lnPR = 0  #  n0 == n1
        else:
            if n0 > n1:
                lnPR = (n1-n0) * _N.log(lmd) + _N.sum(pcdlog[range(n1+1, n0+1)])
            else:
                lnPR = (n1-n0) * _N.log(lmd) - _N.sum(pcdlog[range(n0+1, n1+1)])
        lPR = lnPR + lpPR
        if lPR > 50:
            prRat = 1e+10
        else:
            prRat = _N.exp(lPR)
        
        posRat = 1e+50 if (lFlB[0] - lFlB[1] > 100) else _N.exp(lFlB[0]-lFlB[1])
            
        aln  = min(1, posRat*prRat)

        if rds[it] < aln:
            p0 = p1
            n0 = n1
            lFlB[1] = lFlB[0]
        ps[it] = p0
        ns[it] = n0

