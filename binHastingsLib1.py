import scipy.misc as _sm
import scipy.stats as _ss
import numpy as _N

def Llklhds(ks, n1, p1, n2, p2, out):
    out[0] = _N.sum(_N.log(_sm.comb(n1, ks)) + ks*_N.log(p1) + (n1-ks)*_N.log(1 - p1))
    out[1] = _N.sum(_N.log(_sm.comb(n2, ks)) + ks*_N.log(p2) + (n2-ks)*_N.log(1 - p2))

def trPoi(lmd, a, b):
    """
    a, b inclusive
    """
    ct = _N.random.poisson(lmd)
    while (ct < a) or (ct > b):
        ct = _N.random.poisson(lmd)  #  _ss.poisson.rvs is slower
    return ct

def MCMC(burn, NMC, cts, ns, ps, order):
    pMin=0.001
    pMax=0.99
    Mk  = _N.mean(cts)
    iMk100 = int(100*Mk)
    sdk = _N.std(cts)
    cv = ((sdk*sdk) / Mk)
    nmin= _N.max(cts)

    stdp = 0.02
    stdp2= stdp**2

    p0    = 0.2
    n0    = int(Mk / p0)

    lFlB = _N.empty(2)
    dp   = _N.empty(2)
    n1n0 = _N.empty(2)

    rds  = _N.random.rand(burn+NMC)
    alv  = _N.ones(2)

    posRats = _N.empty(burn+NMC)
    prRats  = _N.empty(burn+NMC)
    n1s     = _N.empty(burn+NMC)
    print "cv is %.3f" % cv
    for it in xrange(burn + NMC):
        if order == 1:
            #############  n jump
            lmd= Mk/p0
            rv = _ss.poisson(lmd)

            #  proposal distribution is Poisson with mean (p0/Mk), truncated to a,b
            n1 = trPoi(lmd, a=nmin, b=iMk100)   #  mean is p0/Mk
            #  based on n1, we pick
            pu = Mk/n1
            p1 = pu + stdp*_ss.truncnorm.rvs((pMin-pu)/stdp, (pMax-pu)/stdp)
        else:  #  why is behavior
            pu = Mk/n0
            p1 = pu + stdp*_ss.truncnorm.rvs((pMin-pu)/stdp, (pMax-pu)/stdp)

            lmd= Mk/p1
            rv = _ss.poisson(lmd)

            #  proposal distribution is Poisson with mean (p0/Mk), truncated to a,b
            n1 = trPoi(lmd, a=nmin, b=iMk100)   #  mean is p0/Mk
            #  based on n1, we pick

        Llklhds(cts, n1, p1, n0, p0, lFlB)

        dp[0]   = (p1 - pu)*(p1 - pu);  dp[1]   = (p0 - pu)*(p0 - pu)
        n1n0[0] = n1;                   n1n0[1] = n0
        qFqB = rv.pmf(n1n0) * _N.exp(-0.5*dp/stdp2)

        posRat = 1e+50 if (lFlB[0] - lFlB[1] > 100) else _N.exp(lFlB[0]-lFlB[1])
        if (qFqB[1] == 0) and (qFqB[0] == 0):
            prRat  = 1
        elif (qFqB[0] == 0):
            prRat  = 1e+10
        else:
            prRat = qFqB[1]/qFqB[0]
            
        alv[1] = posRat*prRat
        aln  = min(1, posRat*prRat)
        posRats[it] = posRat
        prRats[it]  = prRat
        #aln  = _N.min(alv)

        if rds[it] < aln:
            p0 = p1
            n0 = n1
        n1s[it] = n1
        ps[it] = p0
        ns[it] = n0

    return posRats, prRats, n1s
