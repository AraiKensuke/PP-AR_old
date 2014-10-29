import scipy.misc as _sm
import scipy.stats as _ss
import numpy as _N

import warnings
warnings.filterwarnings("error")

__BNML__ = 0   # binomial
__NBML__ = 1   # negative binomial

def Llklhds(type, ks, rn1, p1):
    if type == __BNML__:
        return _N.sum(_N.log(_sm.comb(rn1, ks)) + ks*_N.log(p1) + (rn1-ks)*_N.log(1 - p1))
    if type == __NBML__:
        return _N.sum(_N.log(_sm.comb(ks + rn1-1, ks)) + ks*_N.log(p1) + rn1*_N.log(1 - p1))

def trPoi(lmd, a):
    """
    a, b inclusive
    """
    ct = _N.random.poisson(lmd)
    while (ct < a):
        ct = _N.random.poisson(lmd)
    return ct

def MCMC(burn, NMC, cts, rns, us, order, pcdlog, lfc):
    Mk  = _N.mean(cts)
    iMk = 1./Mk
    sdk = _N.std(cts)
    cv = ((sdk*sdk) * iMk)
    nmin= _N.max(cts)

    stdu = 0.1
    stdu2= stdu**2
    istdu2= 1./ stdu2

    #  let's start it off from binomial
    p0   = 0.1
    u0   = -_N.log(1/p0 - 1)   #  generate initial u0.  sample u's.
    rn0    = int(Mk / p0)

    pTH  = 0.001
    uTH    = _N.log(pTH / (1 - pTH))

    lFlB = _N.empty(2)
    rn1rn0 = _N.empty(2)

    rds  = _N.random.rand(burn+NMC)
    rdns = _N.random.randn(burn+NMC)

    print "cv is %.3f" % cv

    lFlB[1] = Llklhds(__BNML__, cts, rn0, p0)
    rngs   = _N.arange(0, 20000)

    cross  = False
    dist   = __BNML__
    for it in xrange(burn + NMC):
        if dist == __BNML__:
            uu1  = -_N.log(rn0 * iMk - 1) # mean of proposal density
            u1 = uu1 + stdu * rdns[it]

            if u1 > uTH:       ###########   Stay in Binomial ##########
                todist = __BNML__;    cross  = False
                lnc1     = lfc.trncNrmNrmlz(uTH, 100, uu1, stdu)
                p1 = 1 / (1 + _N.exp(-u1))
                lmd= Mk/p1
                rn1 = trPoi(lmd, nmin)   #  mean is p0/Mk
                bLargeP = (p0 > 0.3) and (p1 > 0.3)
                if bLargeP:#    fairly large p.  Exact proposal ratio
                    lmd = Mk/(0.5*(p0+p1))
                else:          #  small p.  prop. ratio far from lower lim of n
                    lmd1= Mk/p1;     lmd0= Mk/p0;     lmd = lmd1
                uu0  = -_N.log(rn1 * iMk - 1) # mean of proposal density
                lnc0     = lfc.trncNrmNrmlz(uTH, 100, uu0, stdu)
                # log of probability
                lpPR = 0.5*istdu2*(-((u1 - uu1)*(u1 - uu1)) + ((u0 - uu0)*(u0 - uu0))) - (lnc0 - lnc1)  #  - (lnc0 - lnc1), lnc reciprocal of norm
            else:   ########  Switch to __NBML__  ####################
                todist = __NBML__;   cross  = True
                lnc1     = lfc.trncNrmNrmlz(-100, uTH, uu1, stdu)  # bin param.
                u1 = uTH - u1  #  u1 now a parameter of NB distribution   
                p1 = 1 / (1 + _N.exp(-u1))
                lmd = (1./p1 - 1)*Mk
                rn1 = trPoi(lmd, rmin)   #  mean is p0/Mk
                lmd1= Mk*((1-p1)/p1);  lmd0= Mk*((1-p0)/p0);   lmd= lmd1
                uu0  = -_N.log(rn1 * iMk) # mean of proposal density
                lnc0     = lfc.trncNrmNrmlz(-100, uTH, uu0, stdu)  # bin param.
                lpPR = 0.5*istdu2*(-(((uTH-u1) - uu1)*((uTH-u1) - uu1)) + ((u0 - uu0)*(u0 - uu0))) - (lnc0 - lnc1)  #  - (lnc0 - lnc1), lnc reciprocal of norm
        if dist == __NBML__:
            uu1  = -_N.log(rn0 * iMk) # mean of proposal density
            u1 = uu + stdu * rdns[it]
            if u1 > uTH:       ######   Stay in Negative binomial ######
                todist = __NBML__;    cross  = False
                lnc1     = lfc.trncNrmNrmlz(uTH, 100, u1, stdu)
                p1 = 1 / (1 + _N.exp(-u1))
                lmd = (1./p1 - 1)*Mk
                rn1 = trPoi(lmd, rmin)   #  mean is p0/Mk
                bLargeP = (p0 > 0.3) and (p1 > 0.3)
                if bLargeP:#    fairly large p.  Exact proposal ratio
                    lmd= Mk*((1-0.5*(p0+p1))/(0.5*(p0+p1)))
                else:          #  small p.  prop. ratio far from lower lim of n
                    lmd1= Mk*((1-p1)/p1);  lmd0= Mk*((1-p0)/p0);   lmd= lmd1
                uu0  = -_N.log(rn1 * iMk) # mean of proposal density
                lnc0     = lfc.trncNrmNrmlz(uTH, 100, uu0, stdu)
                # log of probability
                lpPR = 0.5*istdu2*(-((u1 - uu1)*(u1 - uu1)) + ((u0 - uu0)*(u0 - uu0))) - (lnc0 - lnc1)  #  - (lnc0 - lnc1), lnc reciprocal of norm
            else:   ########  Switch to __BNML__  ####################
                todist = __BNML__;    cross  = True
                u1 = uTH - u1  #  u in NB distribution
                lnc1    = lfc.trncNrmNrmlz(-100, uTH, u1, stdu)
                p1 = 1 / (1 + _N.exp(-u1))
                lmd = Mk/p1
                rn1 = trPoi(lmd, nmin)   #  mean is p0/Mk
                lmd1= Mk/p1;     lmd0= Mk/p0;     lmd = lmd1
                uu0  = -_N.log(rn1 * iMk) # mean of proposal density
                lnc0     = lfc.trncNrmNrmlz(-100, uTH, uu0, stdu)
                lpPR = 0.5*istdu2*(-(((uTH-u1) - uu1)*((uTH-u1) - uu1)) + ((u0 - uu0)*(u0 - uu0))) - (lnc0 - lnc1)  #  - (lnc0 - lnc1), lnc reciprocal of norm

        
        lFlB[0] = Llklhds(todist, cts, rn1, p1)
        rn1rn0[0] = rn1;                   rn1rn0[1] = rn0

        ########  log of proposal probabilities

        if cross or not bLargeP:  #  lnPR can be calculated without regard to whether cross or not, because it is conditionally dependent on pN, pB
            if rn0 == rn1:
                lnPR = rn1*(_N.log(lmd1) - _N.log(lmd0)) - (lmd1 - lmd0) 
            else:
                if rn0 > rn1: # range(n1+1, n0+1)
                    lnPR = rn1*_N.log(lmd1) - rn0*_N.log(lmd0) + _N.sum(pcdlog[rngs[rn1+1:rn0+1]]) - (lmd1 - lmd0)
                else:
                    lnPR = rn1*_N.log(lmd1) - rn0*_N.log(lmd0) - _N.sum(pcdlog[rngs[rn0+1:rn1+1]]) - (lmd1 - lmd0)
        else:
            if rn0 == rn1:
                lnPR = 0  #  r0 == r1
            else:
                if rn0 > rn1: # range(r1+1, r0+1)
                    lnPR = (rn1-rn0) * _N.log(lmd) + _N.sum(pcdlog[rngs[rn1+1:rn0+1]])
                else:
                    lnPR = (rn1-rn0) * _N.log(lmd) - _N.sum(pcdlog[rngs[rn0+1:rn1+1]])

        lPR = lnPR + lpPR
        if lPR > 50:
            prRat = 1e+10
        else:
            prRat = _N.exp(lPR)
                
        posRat = 1e+50 if (lFlB[0] - lFlB[1] > 100) else _N.exp(lFlB[0]-lFlB[1])

        rat  = posRat*prRat
        aln  = rat if (rat < 1)  else 1

        if rds[it] < aln:
            u0 = u1
            rn0 = rn1
            p0 = p1
            lFlB[1] = lFlB[0]
        us[it] = u0
        rns[it] = rn0
