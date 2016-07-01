import numpy as _N
import scipy.misc as _sm
import commdefs as _cd
import matplotlib.pyplot as _plt

#  connection between 
pTH  = 0.01
uTH    = _N.log(pTH / (1 - pTH))

def Llklhds(type, ks, rn1, p1):
    l1gt = 0
    l1lt = 0
    gt = _N.where(ks > 10)[0]
    lt = _N.where(ks <= 10)[0]

    try:
        if type == _cd.__BNML__:
            # if p large, rn1-ksf small   we are not dealing with this region
            # if p small, ks small
            if len(gt) > 0:
                #print len(gt)
                ksf = ks[gt]
                p1f = p1[gt]
                top = 0.5*_N.log(2*_N.pi*rn1) + rn1*_N.log(rn1) - rn1
                b1  = 0.5*_N.log(2*_N.pi*ksf) + ksf*_N.log(ksf) - ksf
                b2  = 0.5*_N.log(2*_N.pi*(rn1-ksf)) + (rn1-ksf)*_N.log(rn1-ksf) - (rn1-ksf)
                l1gt  = _N.sum(top - b1 - b2 + ksf*_N.log(p1f) + (rn1-ksf)*_N.log(1 - p1f))
            if len(lt) > 0:
                ksf = ks[lt]
                p1f = p1[lt]
                l1lt  = _N.sum(_N.log(_sm.comb(rn1, ksf)) + ksf*_N.log(p1f) + (rn1-ksf)*_N.log(1 - p1f))

            return l1lt + l1gt
        if type == _cd.__NBML__:
            ksrn1m1 = ks+rn1-1   #  ks + rn1 -1 > ks
            rn1m1  = rn1-1

            if len(gt) > 0:  # if gt, ksrn1m1 will also be OK to use Stirling
                ksf = ks[gt]
                p1f = p1[gt]
                ksrn1m1f = ksrn1m1[gt]
                top = 0.5*_N.log(2*_N.pi*(ksrn1m1f)) + (ksrn1m1f)*_N.log(ksrn1m1f) - (ksrn1m1f)
                b1  = 0.5*_N.log(2*_N.pi*ksf) + ksf*_N.log(ksf) - ksf
                b2  =_N.sum(_N.log(_N.linspace(1, rn1m1, rn1m1)))  # log(fact)

                l1gt  = _N.sum(top - b1 - b2 + ksf*_N.log(p1f) + rn1*_N.log(1 - p1f))
            if len(lt) > 0:
                ksf = ks[lt]
                p1f = p1[lt]
                l1lt  = _N.sum(_N.log(_sm.comb(ksf + rn1-1, ksf)) + ksf*_N.log(p1f) + rn1*_N.log(1 - p1f))
            return l1lt + l1gt

    except Warning:
        print "Warning raised:   !!!!!!!!"
        print "type %d" % type
        print "rn1  %d" % rn1
        print ksf

        raise

def CtDistLogNorm(type, ks, rnM, LN):
    """
    Log of the normalization constant for count distributions
    """
    l1gt = 0
    l1lt = 0
    gt = _N.where(ks > 10)[0]
    lt = _N.where(ks <= 10)[0]

    try:
        if type == _cd.__BNML__:
            # if p large, rn1-ksf small   we are not dealing with this region
            # if p small, ks small
            if len(gt) > 0:
                #print len(gt)
                ksf = ks[gt]
                top = 0.5*_N.log(2*_N.pi*rnM) + rnM*_N.log(rnM) - rnM
                b1  = 0.5*_N.log(2*_N.pi*ksf) + ksf*_N.log(ksf) - ksf
                b2  = 0.5*_N.log(2*_N.pi*(rnM-ksf)) + (rnM-ksf)*_N.log(rnM-ksf) - (rnM-ksf)
                LN[gt]  = top - b1 - b2
            if len(lt) > 0:
                ksf = ks[lt]
                LN[lt]  = _N.log(_sm.comb(rnM, ksf))
        if type == _cd.__NBML__:
            ksrnMm1 = ks+rnM-1   #  ks + rn1 -1 > ks
            rnMm1  = rnM-1

            if len(gt) > 0:  # if gt, ksrn1m1 will also be OK to use Stirling
                ksf = ks[gt]
                ksrnMm1f = ksrnMm1[gt]
                top = 0.5*_N.log(2*_N.pi*(ksrnMm1f)) + (ksrnMm1f)*_N.log(ksrnMm1f) - (ksrnMm1f)
                b1  = 0.5*_N.log(2*_N.pi*ksf) + ksf*_N.log(ksf) - ksf
                b2  =  0.5*_N.log(2*_N.pi*rnMm1) + rnMm1*_N.log(rnMm1) - rnMm1

                LN[gt]  = top - b1 - b2
            if len(lt) > 0:
                ksf = ks[lt]
                LN[lt] = _N.log(_sm.comb(ksf + rnM-1, ksf))

    except Warning:
        print "!!!!!!!!"
        print "type %d" % type
        print "rn1  %d" % rn1

        raise


def CtDistLogNorm2(mdl, J, ks, rnM, LN):
    """
    Log of the normalization constant for count distributions
    High and low states can be separate distributions

    if counts are too high for a proposed binomial binomial model, 
    we will artifically set counts to be as low as necessary, and
    in step where we choose z occupancy, all those with too high
    counts will automatically be disqualified to be binomial
    """
    l1gt = 0
    l1lt = 0
    gt = _N.where(ks > 10)[0]
    lt = _N.where(ks <= 10)[0]

    try:
        for j in xrange(J):
            rn = rnM[j]
            if mdl[j] == _cd.__BNML__:
                # if p large, rn1-ksf small we are not dealing with this region
                # if p small, ks small
                if len(gt) > 0:
                    #print len(gt)
                    ksf = ks[gt]   #  returns new array
                    notBNML = _N.where(ksf >= rn)[0]   # ct must 1 less than n
                    ksf[notBNML] = rn-1 #  this term not used in the end anyway
                    top = 0.5*_N.log(2*_N.pi*rn) + rn*_N.log(rn) - rn
                    b1  = 0.5*_N.log(2*_N.pi*ksf) + ksf*_N.log(ksf) - ksf
                    b2  = 0.5*_N.log(2*_N.pi*(rn-ksf)) + (rn-ksf)*_N.log(rn-ksf) - (rn-ksf)
                    LN[gt, j]  = top - b1 - b2
                if len(lt) > 0:
                    ksf = ks[lt]
                    notBNML = _N.where(ksf >= rn)[0]   #  
                    ksf[notBNML] = rn-1 #  this term not used in the end anyway
                    LN[lt, j]  = _N.log(_sm.comb(rn, ksf))
            if mdl[j] == _cd.__NBML__:
                ksrnMm1 = ks+rn-1   #  ks + rn1 -1 > ks
                rnMm1  = rn-1

                if len(gt) > 0:  # if gt, ksrn1m1 will also be OK to use Stirling
                    ksf = ks[gt]
                    ksrnMm1f = ksrnMm1[gt]
                    top = 0.5*_N.log(2*_N.pi*(ksrnMm1f)) + (ksrnMm1f)*_N.log(ksrnMm1f) - (ksrnMm1f)
                    b1  = 0.5*_N.log(2*_N.pi*ksf) + ksf*_N.log(ksf) - ksf
                    b2  =  0.5*_N.log(2*_N.pi*rnMm1) + rnMm1*_N.log(rnMm1) - rnMm1

                    LN[gt, j]  = top - b1 - b2
                if len(lt) > 0:
                    ksf = ks[lt]
                    LN[lt, j] = _N.log(_sm.comb(ksf + rn-1, ksf))
    except Warning:
        print "!!!!!!!!"
        print "type %s" % str(mdl)
        print "rnM  %s" % str(rnM)

        raise

def startingValues(cts, fillsmpx=None, cv0=None, trials=None):
    if trials is not None:  # fillsmpx[trials] couldn't be passed as a pointer
        cts = cts[trials]   # fillsmpx = fillsmpx[trials] creates new array
    else:                   # so we must use fillsmpx[trials] = ...  for assign
        trials = _N.arange(len(cts))
    Nss = len(cts)  #  N same state
    Mk = _N.mean(cts)  #  1comp if nWins=1, 2comp
    iMk = 1./Mk
    sdk = _N.std(cts)
    cv = ((sdk*sdk) * iMk)
    nmin= _N.max(cts)   #  if n too small, can't generate data
    rmin= 1

    epS  = 20   #  in 20 trial segments

    ##########################  estimate cv0, p0, u0
    Npcs = (Nss) / epS   #  20 trials each
    cvs  = _N.empty(Npcs)
    mns  = _N.empty(Npcs)
    rns  = _N.empty(Npcs)
    rn1s = _N.empty(Npcs)

    for ep in xrange(Npcs):
        mns[ep]  = _N.mean(cts[ep*epS:(ep+1)*epS]) # OK even if overshoot 
        cvs[ep]       = _N.std(cts[ep*epS:(ep+1)*epS])**2 / mns[ep]
    if cv0 is None:
        cv0   = _N.mean(cvs)   #  only set this if not passed in

    mdl, p0 = (_cd.__BNML__, 1 - cv0) if cv0 < 1 else (_cd.__NBML__, 1 - 1/cv0)
    #mdl, p0 = (_cd.__BNML__, 1 - cv0) 
    #if p0 < 0: p0 = 0.99
    a1o0 = 1 if (mdl == _cd.__BNML__) else 0   #  add 1 or 0

    u0 = -_N.log(1/p0 - 1)
    ###  estimate rn0
    for ep in xrange(Npcs):
        rns[ep]  = (mns[ep] / p0) if mdl == _cd.__BNML__ else (1/p0 - 1)*mns[ep]

    rn0 = int(_N.mean(rns))

    if mdl == _cd.__BNML__:
        rn0 = (max(cts)+1) if rn0 <= max(cts) else rn0
    print "rn0  %d" % rn0

    ##########################  estimate xn
    xn     = _N.array(mns)
    print "cts*******"
    print cts
    print "mns-------"
    print mns

    xoff   = _N.empty(Nss)

    xoff[0:Npcs*epS] = _N.repeat(xn, epS)
    xoff[Npcs*epS:] = xn[Npcs-1]   #  if theres some values left at end

    xoff   -= _N.mean(xoff)
    xn        -= _N.mean(xn)

    xoff   /= _N.std(xoff)   #  normalized

    #  for 
    magStps = 30
    lls  = _N.empty(magStps)
    mags = _N.linspace(0, 1, magStps)
    j    = 0

    for mag in _N.linspace(0, 1, magStps):
        print "mag %f" % mag
        p1x = 1 / (1 + _N.exp(-(u0 + mag*xoff)))
        lls[j] = Llklhds(mdl, cts, rn0, p1x)
        j += 1

    maxJ = _N.where(_N.max(lls) == lls)[0][0]

    xoff *= mags[maxJ]
    fillsmpx[trials] = xoff

    # print "hereA   %d" % len(fillsmpx)
    # print fillsmpx
    # print "hereB"
    xn   *= mags[maxJ]
    p1x = 1 / (1 + _N.exp(-(u0 + xoff)))

    llsV = _N.empty(40)
    print "printing cts"
    print cts
    bestRN = bestrn(mdl, cts, rn0, llsV, p1x)
    fig = _plt.figure()
    _plt.hist(cts, bins=_N.linspace(0, 50, 51))
    FF  = _N.std(cts)**2/_N.mean(cts)
    _plt.suptitle("bestRN %(br)d   FF %(FF).3f" % {"br" : bestRN, "FF" : FF})
    
    # print "FF %.3f" % FF
    #print bestRN
    # if bestRN < 10:
    #     bestRN = 40
    #     u0     -= .5

    return u0, bestRN, mdl
    #  For cv~1, r~n 

def startingValuesM(cts, J, zs, fillsmpx=None, indLH=False):
    epS  = 20   #  in 20 trial segments
    Nss = len(cts)  #  N same state

    ##########################  estimate cv0, p0, u0
    Npcs = Nss / epS   #  20 trials each

    mns= _N.empty(Npcs)
    for ep in xrange(Npcs):
        mns[ep]  = _N.mean(cts[ep*epS:(ep+1)*epS]) # OK even if overshoot 
        loInds = (_N.where(cts[ep*epS:(ep+1)*epS] < mns[ep])[0])  + ep*epS
        hiInds = (_N.where(cts[ep*epS:(ep+1)*epS] >= mns[ep])[0]) + ep*epS
        zs[loInds, 0] = 1
        zs[hiInds, 1] = 1

    cv0s = _N.empty(J)
    u0s  = _N.empty(J)
    bestRNs = _N.empty(J, dtype=_N.int)
    models = _N.empty(J, dtype=_N.int)

    for j in xrange(J):
        trls = _N.where(zs[:, j] == 1)[0]

        #  1 / (1-p) = c      1/c = 1-p   p = 1 - 1/c
        u0, bestRN, mdl = startingValues(cts, fillsmpx=fillsmpx, trials=trls)
        p0    = 1 / (1 + _N.exp(-u0))
        cv0s[j] = (1 - p0) if (mdl == _cd.__BNML__) else 1 / (1 - p0)
        print cv0s[j]
        cv0s[j] *= float(len(trls)) / Nss    # weighted cv0
        bestRNs[j] = bestRN
        models[j]  = mdl
        u0s[j] = u0

    if indLH:
        return u0s, bestRNs, models
    cv0 = _N.sum(cv0s)

    for j in xrange(J):
        trls = _N.where(zs[:, j] == 1)[0]

        #  1 / (1-p) = c      1/c = 1-p   p = 1 - 1/c
        u0, bestRNs[j], mdl = startingValues(cts, fillsmpx=fillsmpx, cv0=cv0, trials=trls)
    return u0, bestRNs, mdl

# def startingValuesMw(cts, J, zs, fillsmpx=None, indLH=False):
#     epS  = 20   #  in 20 trial segments
#     Nss = cts.shape[0]
#     WNS   = cts.shape[1]
#     print "WNS   %d" % WNS

#     ##########################  estimate cv0, p0, u0
#     Npcs = Nss / epS   #  20 trials each

#     mns= _N.empty((Npcs, WNS))
#     zsW= _N.zeros((WNS, Nss, J), dtype=_N.int)

#     for ep in xrange(Npcs):
#         mns[ep]  = _N.mean(cts[ep*epS:(ep+1)*epS], axis=0) # overshoot OK
#         for w in xrange(WNS):
#             loInds = (_N.where(cts[ep*epS:(ep+1)*epS,w] < mns[ep,w])[0])  + ep*epS
#             hiInds = (_N.where(cts[ep*epS:(ep+1)*epS,w] >= mns[ep,w])[0]) + ep*epS

#             if J > 1:
#                 zsW[w, loInds, 0] = 1
#                 zsW[w, hiInds, 1] = 1
#             else:
#                 zsW[w, :, 0] = 1

#     if J > 1:
#         loInds = _N.where(_N.mean(zsW[:, :, 0], axis=0) >= 0.5)[0]
#         hiInds = _N.where(_N.mean(zsW[:, :, 0], axis=0) <  0.5)[0]
#         zs[loInds, 0] = 1
#         zs[hiInds, 1] = 1
#     else:
#         zs[:, 0] = 1

#     cv0s = _N.empty((WNS, J))
#     u0s  = _N.empty((WNS, J))
#     bestRNs = _N.empty((WNS, J), dtype=_N.int)
#     models = _N.empty((WNS, J), dtype=_N.int)

#     fs   = _N.zeros((WNS, Nss))
#     for w in xrange(WNS-1, -1, -1):
#         for j in xrange(J):
#             print "j is %d" % j
#             trls = _N.where(zs[:, j] == 1)[0]
#             print trls

#             #  1 / (1-p) = c      1/c = 1-p   p = 1 - 1/c
#             u0, bestRN, mdl = startingValues(cts[:, w], fillsmpx=fs[w], trials=trls)
#             #print "mean cts %.3f" % _N.mean(cts[trls, w])
#             p0    = 1 / (1 + _N.exp(-u0))
#             cv0s[w, j] = (1 - p0) if (mdl == _cd.__BNML__) else 1 / (1 - p0)
#             #print cv0s[w, j]
#             cv0s[w, j] *= float(len(trls)) / Nss    # weighted cv0
#             bestRNs[w, j] = bestRN
#             models[w, j]  = mdl
#             u0s[w, j] = u0
#             #_plt.plot(fs[w])
#     _N.mean(fs, axis=0, out=fillsmpx)
    
#     return u0s, bestRNs, models


def startingValuesMw(cts, J, zs, fillsmpx=None, indLH=False):
    epS  = 20   #  in 20 trial segments
    Nss = cts.shape[0]
    WNS   = cts.shape[1]
    print "WNS   %d" % WNS

    ##########################  estimate cv0, p0, u0
    Npcs = Nss / epS   #  20 trials each

    mns= _N.empty((Npcs, WNS))
    zsW= _N.zeros((WNS, Nss, J), dtype=_N.int)

    for w in xrange(WNS):
        for ep in xrange(Npcs):
            mns[ep]  = _N.mean(cts[ep*epS:(ep+1)*epS], axis=0) # overshoot OK
            loInds = (_N.where(cts[ep*epS:(ep+1)*epS,w] < 0.6*mns[ep,w])[0])  + ep*epS
            hiInds = (_N.where(cts[ep*epS:(ep+1)*epS,w] >= 0.6*mns[ep,w])[0]) + ep*epS

            if J > 1:
                zsW[w, loInds, 0] = 1
                zsW[w, hiInds, 1] = 1
            else:
                zsW[w, :, 0] = 1

    if J > 1:  # consensus of both windows
        loInds = _N.where(_N.mean(zsW[:, :, 0], axis=0) >= 0.5)[0]
        hiInds = _N.where(_N.mean(zsW[:, :, 0], axis=0) <  0.5)[0]
        zs[loInds, 0] = 1
        zs[hiInds, 1] = 1
    else:
        zs[:, 0] = 1

    cv0s = _N.empty((WNS, J))
    u0s  = _N.empty((WNS, J))
    bestRNs = _N.empty((WNS, J), dtype=_N.int)
    models = _N.empty((WNS, J), dtype=_N.int)

    fs   = _N.zeros((WNS, Nss))
    for w in xrange(WNS-1, -1, -1):
        for j in xrange(J):
            print "j is %d" % j
            trls = _N.where(zs[:, j] == 1)[0]
            print trls

            #  1 / (1-p) = c      1/c = 1-p   p = 1 - 1/c
            u0, bestRN, mdl = startingValues(cts[:, w], fillsmpx=fs[w], trials=trls)
            #print "mean cts %.3f" % _N.mean(cts[trls, w])
            p0    = 1 / (1 + _N.exp(-u0))
            cv0s[w, j] = (1 - p0) if (mdl == _cd.__BNML__) else 1 / (1 - p0)
            #print cv0s[w, j]
            cv0s[w, j] *= float(len(trls)) / Nss    # weighted cv0
            bestRNs[w, j] = bestRN
            models[w, j]  = mdl
            u0s[w, j] = u0
            #_plt.plot(fs[w])
    _N.mean(fs, axis=0, out=fillsmpx)
    
    return u0s, bestRNs, models


def startingValuesW(cts, fillsmpx=None):
    epS  = 20   #  in 20 trial segments
    Nss = cts.shape[0]
    WNS   = cts.shape[1]
    print "WNS   %d" % WNS

    ##########################  estimate cv0, p0, u0
    Npcs = Nss / epS   #  20 trials each

    mns= _N.empty((Npcs, WNS))

    cv0s = _N.empty(WNS)
    u0s  = _N.empty(WNS)
    bestRNs = _N.empty((WNS), dtype=_N.int)
    models = _N.empty((WNS), dtype=_N.int)

    fs   = _N.zeros((WNS, Nss))
    for w in xrange(WNS-1, -1, -1):
        #  1 / (1-p) = c      1/c = 1-p   p = 1 - 1/c
        u0, bestRN, mdl = startingValues(cts[:, w], fillsmpx=fs[w])
        #print "mean cts %.3f" % _N.mean(cts[trls, w])
        p0    = 1 / (1 + _N.exp(-u0))
        cv0s[w] = (1 - p0) if (mdl == _cd.__BNML__) else 1 / (1 - p0)
        bestRNs[w] = bestRN
        models[w]  = mdl
        u0s[w] = u0
    _N.mean(fs, axis=0, out=fillsmpx)
    
    return u0s, bestRNs, models

def bestrn(dist, cnt, lmd0, llsV, p1x):
    dn   = int(lmd0*0.02)
    dn   = 1 if dn == 0 else dn

    n0   = lmd0 - 20*dn
    n1   = lmd0 + 20*dn
    n0   = 1 if n0 < 1 else n0
    if dist == _cd.__BNML__:
        n0min = _N.max(cnt) + 1
        n0   = n0min if n0 < n0min else n0
    
    candRNs = _N.arange(n0, n1, dn)
    #print len(candRNs)
    #print len(llsV)
    j    = 0

    #llsV  = _N.empty(len(candRNs))
    for tryRN in candRNs:
        llsV[j] = Llklhds(dist, cnt, tryRN, p1x)
        j += 1
    for ij in xrange(len(candRNs), 40):
        llsV[ij] = llsV[ij-1]
                     
    #fig = _plt.figure()
    maxI = _N.where(_N.max(llsV) == llsV)[0][0]
    # print dn
    # print len(llsV)
    # print len(candRNs)
    # print "maxI   %d" % maxI
    return candRNs[maxI]



def cntmdlMCMCOnly(GibbsIter, iters, u0, rn0, dist, cts, rns, us, dty, xn, stdu=0.5):
    """
    We need starting values for rn, u0, model
    """
    # u0, rn0, dist = startingValues(cts, xoff=xn)
    # print "starting values"
    # print "$$$$$$$$$$$$$$$$$$$$$$$$"
    # print "rn=%d" % rn0
    # print "us=%.3f" % u0
    # print "md=%d" % dist
    # print "$$$$$$$$$$$$$$$$$$$$$$$$"
    # print "in cntmdlMCMCOnly"
    # print rn0
    # print dist
    # print cts
    #  proposal parameters
    stdu2= stdu**2;
    istdu2= 1./ stdu2

    Mk = _N.mean(cts)  #  1comp if nWins=1, 2comp
    iMk = 1./Mk
    nmin= _N.max(cts)   #  if n too small, can't generate data
    rmin= 1

    lFlB = _N.empty(2)
    rn1rn0 = _N.empty(2)

    rds  = _N.random.rand(iters)
    rdns = _N.random.randn(iters)

    p0  = 1 / (1 + _N.exp(-u0))
    p0x = 1 / (1 + _N.exp(-(u0+xn)))
    lFlB[1] = Llklhds(dist, cts, rn0, p0x)
    lBg = lFlB[1]

    cross  = False
    lls   = []
    accptd = 0

    llsV = _N.empty(40)
    #rn0 = bestrn(dist, cts, rn0, llsV, p0x)

    #print "uTH is %.3e" % uTH
    for it in xrange(iters):
        #
        if dist == _cd.__BNML__:
            uu1  = -_N.log(rn0 * iMk - 1) # mean of proposal density
            u1 = uu1 + stdu * rdns[it]
            #print "BNML   uu1  %(uu1).3e    u1  %(u1).3e" % {"uu1" : uu1, "u1" : u1}

            if u1 > uTH:       ###########   Stay in Binomial ##########
                todist = _cd.__BNML__;    cross  = False
                p1 = 1 / (1 + _N.exp(-u1))
                p1x = 1 / (1 + _N.exp(-(u1+xn)))
                lmd0= int(Mk/p1)
                rn1 = bestrn(todist, cts, lmd0, llsV, p1x)
                #print "%(1)d   %(2)d" % {"1" : lmd0, "2" : rn1}

                uu0  = -_N.log(rn1 * iMk - 1) # mean of proposal density
                # log of probability
                #print "d1  %(1) .3f    %(2) .3f" % {"1" : (u1 - uu1), "2" : (u0 - uu0)}
                #lpPR = 0.5*istdu2*(-((u1 - uu1)*(u1 - uu1)) + ((u0 - uu0)*(u0 - uu0)))  #  - (lnc0 - lnc1), lnc reciprocal of norm
                lpPR = 0.5*istdu2*(((u1 - uu1)*(u1 - uu1)) - ((u0 - uu0)*(u0 - uu0)))  #  - (lnc0 - lnc1), lnc reciprocal of norm
            else:   ########  Switch to __NBML__  ####################
                print "switch 2 NBML"
                todist = _cd.__NBML__;   cross  = True
                u1 = 2*uTH - u1  #  u1 now a parameter of NB distribution   
                p1 = 1 / (1 + _N.exp(-u1))
                p1x = 1 / (1 + _N.exp(-(u1+xn)))
                lmd0 = int((1./p1 - 1)*Mk)
                rn1 = bestrn(todist, cts, lmd0, llsV, p1x)

                uu0  = -_N.log(rn1 * iMk) # mean of proposal density
                #lpPR = 0.5*istdu2*(-(((uTH-u1) - uu1)*((uTH-u1) - uu1)) + (((uTH-u0) - uu0)*((uTH-u0) - uu0)))
                lpPR = 0.5*istdu2*((((uTH-u1) - uu1)*((uTH-u1) - uu1)) - (((uTH-u0) - uu0)*((uTH-u0) - uu0)))
        elif dist == _cd.__NBML__:
            uu1  = -_N.log(rn0 * iMk) # mean of proposal density
            u1 = uu1 + stdu * rdns[it]
            #print "NBML   uu1  %(uu1).3e    u1  %(u1).3e" % {"uu1" : uu1, "u1" : u1}

            if u1 > uTH:       ######   Stay in Negative binomial ######
                todist = _cd.__NBML__;    cross  = False
                p1 = 1 / (1 + _N.exp(-u1))
                p1x = 1 / (1 + _N.exp(-(u1+xn)))
                lmd0 = int((1./p1 - 1)*Mk)
                #print "lmd0   %d" % lmd0
                rn1 = bestrn(todist, cts, lmd0, llsV, p1x)

                #rn1 = trPoi(lmd, rmin)   #  mean is p0/Mk
                # bLargeP = (p0 > 0.3) and (p1 > 0.3)
                # if bLargeP:#    fairly large p.  Exact proposal ratio
                #     lmd= Mk*((1-0.5*(p0+p1))/(0.5*(p0+p1)))
                # else:          #  small p.  prop. ratio far from lower lim of n
                #     lmd1= Mk*((1-p1)/p1);  lmd0= Mk*((1-p0)/p0);   lmd= lmd1
                uu0  = -_N.log(rn1 * iMk) # mean of proposal density
                # log of probability

                #lpPR = 0.5*istdu2*(-((u1 - uu1)*(u1 - uu1)) + ((u0 - uu0)*(u0 - uu0)))
                lpPR = 0.5*istdu2*(((u1 - uu1)*(u1 - uu1)) - ((u0 - uu0)*(u0 - uu0)))
            else:   ########  Switch to __BNML__  ####################
                print "switch 2 BNML"
                todist = _cd.__BNML__;    cross  = True
                u1 = 2*uTH - u1  #  u in NB distribution
                p1 = 1 / (1 + _N.exp(-u1))
                p1x = 1 / (1 + _N.exp(-(u1+xn)))
                lmd0= int(Mk/p1)
                rn1 = bestrn(todist, cts, lmd0, llsV, p1x)

                #lmd = Mk/p1
                #rn1 = trPoi(lmd, nmin)   #  mean is p0/Mk
                #lmd1= Mk/p1;     lmd0= Mk*((1-p0)/p0);     lmd = lmd1
                uu0  = -_N.log(rn1 * iMk - 1) # mean of proposal density
                lpPR = 0.5*istdu2*(-(((uTH-u1) - uu1)*((uTH-u1) - uu1)) + (((uTH-u0) - uu0)*((uTH-u0) - uu0)))
                #lpPR = 0.5*istdu2*((((uTH-u1) - uu1)*((uTH-u1) - uu1)) - (((uTH-u0) - uu0)*((uTH-u0) - uu0)))

        lFlB[0] = Llklhds(todist, cts, rn1, p1x)
        #print "proposed state  ll  %(1).3e   old state  ll  %(2).3e     new-old  %(3).3e" % {"1" : lFlB[0], "2" : lFlB[1], "3" : (lFlB[0] - lFlB[1])}

        rn1rn0[0] = rn1;                   rn1rn0[1] = rn0

        ########  log of proposal probabilities

        lnPR = 0    #  we have the log part set to 1.  No change
        lPR = lnPR + lpPR
        lposRat = lFlB[0] - lFlB[1]
        lrat = lPR + lposRat
        # if lPR > 100:
        #     prRat = 2.7e+43
        # else:
        #     prRat = _N.exp(lPR)

        #  lFlB[0] - lFlB[1] >> 0  -->  new state has higher likelihood
        #posRat = 1.01e+200 if (lFlB[0] - lFlB[1] > 500) else _N.exp(lFlB[0]-lFlB[1])

        #print "posRat %(1).3e     prRat %(2).3e" % {"1" : posRat, "2" : prRat}
        #print "lrat is %f" % lrat
        #rat  = _N.exp(lrat)

        aln   = 1 if (lrat > 0) else _N.exp(lrat)
        #aln  = rat if (rat < 1)  else 1   #  if aln == 1, always accept
        if rds[it] < aln:   #  accept
            accptd += 1
            u0 = u1
            rn0 = rn1
            p0 = p1
            lFlB[1] = lFlB[0]
            #lls.append(lFlB[1])
            #print "accepted  %d" % it
            dist = todist
        lls.append(lFlB[1])

        dty[it] = dist
        us[it] = u0
        rns[it] = rn0    #  rn0 is the newly sampled value if accepted

    print "accepted %d" % accptd
    # fig = _plt.figure()
    # _plt.plot(llsV)
    # _plt.suptitle(accptd)
    # _plt.savefig("llsV%d" % GibbsIter)
    # _plt.close()
    lEn = lFlB[0]

    #print "ll Bg %(b).3e   ll En %(e).3e" % {"b" : lBg, "e" : lEn}
    return u0, rn0, dist

def cntmdlMCMCOnlyM(iters, J, zs, u0, rn0, dist, cts, rns, us, dty, xn):
    """
    We need starting values for rn, u0, model
    """
    #  if one of the mix categories has 0 occupancy, default to cntmdlMCMCOnly
    emptyCats = _N.where(_N.sum(zs, axis=0) == 0)[0]
    if len(emptyCats) > 0:
        print "empty cats"
        nonEmpty = 1 - emptyCats[0]
        ru0, rrn0, rdist = cntmdlMCMCOnly(iters, u0, rn0[nonEmpty], dist, cts, rns[:, nonEmpty], us, dty, xn)
        rn0M= _N.empty(J, dtype=_N.int)
        rn0M[emptyCats] = rn0[emptyCats]
        rn0M[nonEmpty] = rn0[nonEmpty]
        return ru0, rn0M, rdist

    llsV = _N.empty(20)

    #  Pick mu.  Optimize rn for both states
    lFlB = _N.zeros(2)

    p0  = 1 / (1 + _N.exp(-u0))

    Mks = _N.empty(J)
    iMks= _N.empty(J)
    rn0M= _N.empty(J, dtype=_N.int)
    rn1M= _N.empty(J, dtype=_N.int)
    nmin= _N.empty(J, dtype=_N.int)
    uu0M= _N.empty(J)
    uu1M= _N.empty(J)
    wgM = _N.empty(J)
    rmin= 1
    rds  = _N.random.rand(iters)
    rdns = _N.random.randn(iters)

    trls = []
    lFlB[1] = 0
    for j in xrange(J):
        trls.append(_N.where(zs[:, j] == 1)[0])
        p0x = 1 / (1 + _N.exp(-(u0+xn[trls[j]])))
        Mks[j] = _N.mean(cts[trls[j]])  #  1comp if nWins=1, 2comp
        iMks[j] = 1./Mks[j]
        nmin[j]= _N.max(cts[trls[j]])   #  if n too small, can't generate data
        rn0M[j] = rn0[j]
        wgM[j]  = float(len(trls[j])) / len(cts)

        lFlB[1] += Llklhds(dist, cts[trls[j]], rn0M[j], p0x)
    lBg = lFlB[1]

    cross  = False
    lls   = []

    for it in xrange(iters):
        #
        if dist == _cd.__BNML__:
            _N.log(rn0M * iMks - 1, out=uu1M) # mean of proposal density
            uu1M *= -1
            #print uu1M
            uu1 = _N.sum(uu1M * wgM)
            #print uu1
            u1 = uu1 + stdu * rdns[it]

            if u1 > uTH:       ###########   Stay in Binomial ##########
                todist = _cd.__BNML__;    cross  = False
                p1 = 1 / (1 + _N.exp(-u1))

                for j in xrange(J):
                    p1x = 1 / (1 + _N.exp(-(u1+xn[trls[j]])))
                    lmd0= int(Mks[j]/p1)  #  bestrn for lo and hi state
                    rn1M[j] = bestrn(todist, cts[trls[j]], lmd0, llsV, p1x)

                _N.log(rn1M * iMks - 1, out=uu0M) # mean of proposal density
                uu0M*=-1
                uu0 = _N.sum(uu0M*wgM)
                #print "--  %(uu0).3f   %(u0).3f" % {"uu0" : uu0, "u0" : u0}
                # log of probability
                lpPR = 0.5*istdu2*(((u1 - uu1)*(u1 - uu1)) - ((u0 - uu0)*(u0 - uu0)))  #  - (lnc0 - lnc1), lnc reciprocal of norm
            else:   ########  Switch to __NBML__  ####################
                todist = _cd.__NBML__;   cross  = True
                u1 = 2*uTH - u1  #  u1 now a parameter of NB distribution   
                p1 = 1 / (1 + _N.exp(-u1))
                for j in xrange(J):
                    p1x = 1 / (1 + _N.exp(-(u1+xn[trls[j]])))
                    lmd0= int((1./p1 - 1)*Mks[j])  #  bestrn for lo and hi state
                    rn1M[j] = bestrn(todist, cts[trls[j]], lmd0, llsV, p1x)

                _N.log(rn1M * iMks - 1, out=uu0M) # mean of proposal density
                uu0M*=-1
                uu0 = _N.sum(uu0M*wgM)

                lpPR = 0.5*istdu2*((((uTH-u1) - uu1)*((uTH-u1) - uu1)) - (((uTH-u0) - uu0)*((uTH-u0) - uu0)))
        elif dist == _cd.__NBML__:
            _N.log(rn0M * iMks, out=uu1M) # mean of proposal density
            uu1M *= -1
            #print uu1M
            uu1 = _N.sum(uu1M * wgM)
            #print uu1
            u1 = uu1 + stdu * rdns[it]

            if u1 > uTH:       ######   Stay in Negative binomial ######
                todist = _cd.__NBML__;    cross  = False
                p1 = 1 / (1 + _N.exp(-u1))
                for j in xrange(J):
                    p1x = 1 / (1 + _N.exp(-(u1+xn[trls[j]])))
                    lmd0= int((1./p1 - 1)*Mks[j])  #  bestrn for lo and hi state
                    rn1M[j] = bestrn(todist, cts[trls[j]], lmd0, llsV, p1x)

                _N.log(rn1M * iMks, out=uu0M) # mean of proposal density
                uu0M*=-1
                uu0 = _N.sum(uu0M*wgM)
                #print "--  %(uu0).3f   %(u0).3f" % {"uu0" : uu0, "u0" : u0}

                lpPR = 0.5*istdu2*(((u1 - uu1)*(u1 - uu1)) - ((u0 - uu0)*(u0 - uu0)))
            else:   ########  Switch to __BNML__  ####################
                #print "here"
                todist = _cd.__BNML__;    cross  = True
                u1 = 2*uTH - u1  #  u in NB distribution
                p1 = 1 / (1 + _N.exp(-u1))
                for j in xrange(J):
                    p1x = 1 / (1 + _N.exp(-(u1+xn[trls[j]])))
                    lmd0= int(Mks[j]/p1)  #  bestrn for lo and hi state
                    rn1M[j] = bestrn(todist, cts[trls[j]], lmd0, llsV, p1x)

                _N.log(rn1M * iMks - 1, out=uu0M) # mean of proposal density
                uu0M*=-1
                uu0 = _N.sum(uu0M*wgM)

                lpPR = 0.5*istdu2*(-(((uTH-u1) - uu1)*((uTH-u1) - uu1)) + (((uTH-u0) - uu0)*((uTH-u0) - uu0)))

        lFlB[0] = 0
        for j in xrange(J):
            p1x = 1 / (1 + _N.exp(-(u1+xn[trls[j]])))
            lFlB[0] += Llklhds(todist, cts[trls[j]], rn1M[j], p1x)

        ########  log of proposal probabilities

        lnPR = 0
        lPR = lnPR + lpPR

        if lPR > 50:
            prRat = 1e+10
        else:
            prRat = _N.exp(lPR)

        #  lFlB[0] - lFlB[1] >> 0  -->  new state has higher likelihood
        posRat = 1.01e+200 if (lFlB[0] - lFlB[1] > 500) else _N.exp(lFlB[0]-lFlB[1])
        rat  = posRat*prRat

        aln  = rat if (rat < 1)  else 1   #  if aln == 1, always accept
        if rds[it] < aln:   #  accept
            u0 = u1
            rn0M[:] = rn1M
            p0 = p1
            lFlB[1] = lFlB[0]
            dist = todist
        lls.append(lFlB[1])

        dty[it] = dist
        us[it] = u0
        rns[it] = rn0M    #  rn0 is the newly sampled value if accepted

    lEn = lFlB[0]

    
    iInd = 0


    print "ll Bg %(b).3e   ll En %(e).3e" % {"b" : lBg, "e" : lEn}
    return u0, rn0M, dist



