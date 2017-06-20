import numpy as _N
import scipy.misc as _sm
import commdefs as _cd
import matplotlib.pyplot as _plt
import time as _tm
import scipy.stats as _ss

#  connection between 
pTH  = 0.01
uTH    = _N.log(pTH / (1 - pTH))
logfact= None

ints = _N.arange(5000)

def Llklhds(typ, ks, rn1, p1):
    global logfact
    N = len(ks)
    if typ == _cd.__BNML__:
        return N*logfact[rn1]-_N.sum(logfact[ks]+logfact[rn1-ks]-ks*_N.log(p1) - (rn1-ks)*_N.log(1 - p1))
    else:
        return _N.sum(logfact[ks+rn1-1]-logfact[ks]  + ks*_N.log(p1) + rn1*_N.log(1 - p1))-N*logfact[rn1-1]


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

    llsV = _N.empty(16)
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
            mns[ep]  = _N.mean(cts[ep*epS:(ep+1)*epS,w], axis=0) # overshoot OK
            loInds = (_N.where(cts[ep*epS:(ep+1)*epS,w] < mns[ep,w])[0])  + ep*epS
            hiInds = (_N.where(cts[ep*epS:(ep+1)*epS,w] >= mns[ep,w])[0]) + ep*epS

            if J > 1:
                zsW[w, loInds, 0] = 1
                zsW[w, hiInds, 1] = 1
            else:
                zsW[w, :, 0] = 1

    if J > 1:  # consensus of both windows
        # if W > 1:
        #     loInds = _N.where(_N.mean(zsW[:, :, 0], axis=0) >= 0.5)[0]
        #     hiInds = _N.where(_N.mean(zsW[:, :, 0], axis=0) <  0.5)[0]
        # else:
        loInds = _N.where(zsW[0, :, 0] >= 0.5)[0]
        hiInds = _N.where(zsW[0, :, 0] <  0.5)[0]

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


def bestrn(dist, cnt, lmd0, llsV, p1x):
    ##  Conditioned on p1x's, what is the best rn?  Search around the
    ##  vicinity of lmd0
    ##
    dn   = int(lmd0*0.02)
    dn   = 1 if dn == 0 else dn

    n0   = lmd0 - 8*dn
    n1   = lmd0 + 8*dn
    n0   = 1 if n0 < 1 else n0
    if dist == _cd.__BNML__:
        n0min = _N.max(cnt) + 1
        if n0 < n0min:
            n0 = n0min
            n1 = n0min + 16
            dn = 1
    
    candRNs = _N.arange(n0, n1, dn)
    #print len(candRNs)
    #print len(llsV)
    j    = -1

    #llsV  = _N.empty(len(candRNs))
    L      = len(candRNs)
    if L == 0:
        print "------------   L is 0"
        print "n0  %(n0)d   n1  %(n1)d     dn  %(dn)d    lmd0*0.02  %(lmd0002)d" % {"n0" : n0, "n1" : n1, "dn" : dn, "lmd0002" : int(lmd0*0.02)}
        if dist == _cd.__BNML__:
            print "n0min is %d" % n0min
    for tryRN in candRNs:
        j += 1
        llsV[j] = Llklhds(dist, cnt, tryRN, p1x)

    #for ij in xrange(len(candRNs), 40):
    #    llsV[ij] = llsV[ij-1]
                     
    #fig = _plt.figure()
    maxI = _N.where(_N.max(llsV[0:L]) == llsV[0:L])[0][0]
    # print dn
    # print len(llsV)
    # print len(candRNs)
    # print "maxI   %d" % maxI
    #print "time in bestrn %.4f" % (tt1-tt0)
    return candRNs[maxI]



def cntmdlMCMCOnly(GibbsIter, iters, u0, rn0, dist, cts, rns, us, dty, xn, stdu=0.5):
    global ints, logfact
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

    Mk = _N.mean(cts) if len(cts) > 0 else 0  #  1comp if nWins=1, 2comp
    if Mk == 0:
        return u0, rn0, dist   # no data assigned to this 
    iMk = 1./Mk   #  if Mk == 0, just make it a small number?
    nmin= _N.max(cts)+1 if len(cts) > 0 else 0   #  if n too small, can't generate data
    rmin= 1

    lFlB = _N.empty(2)
    rn1rn0 = _N.empty(2)

    rds  = _N.random.rand(iters)
    rdns = _N.random.randn(iters)

    p0  = 1 / (1 + _N.exp(-u0))
    p0x = 1 / (1 + _N.exp(-(u0+xn)))

    lFlB[1] = Llklhds(dist, cts, rn0, p0x)
    lBg = lFlB[1]   #  

    cross  = False
    lls   = []
    accptd = 0

    #rn0 = bestrn(dist, cts, rn0, llsV, p0x)

    #print "uTH is %.3e" % uTH

    #dbtt21 = 0
    #dbtt32 = 0
    #dbtt43 = 0
    #dbtt54 = 0

    #  the poisson distribution needs to be truncated

    for it in xrange(iters):
        bDone = False
        #
        #dbtt1 = _tm.time()
        if dist == _cd.__BNML__:
            uu1  = -_N.log(rn0 * iMk - 1) # mean of proposal density
            u1 = uu1 + stdu * rdns[it]    #  **PROPOSED** u1

            if u1 > uTH:       ###########   Stay in Binomial ##########
                todist = _cd.__BNML__;    cross  = False
                p1 = 1 / (1 + _N.exp(-u1))
                p1x = 1 / (1 + _N.exp(-(u1+xn)))
                lmd1= Mk/p1

                llmd1 = _N.log(lmd1)
                trms = _N.exp(ints[0:nmin]*llmd1 - logfact[0:nmin] - lmd1)
                lC1 = _N.log(1 - _N.sum(trms))    #  multiple by which to multiply pmf(k) to get pmf fo trunc

                while not bDone:
                    rnds = _ss.poisson.rvs(lmd1, size=5)
                    pois = _N.where(rnds >= nmin)[0]
                    if len(pois) > 0:
                        bDone = True
                        rn1   = rnds[pois[0]]   # bn parameter

                #print rn1
                lpm1 = rn1 * llmd1 - logfact[rn1] - lmd1 - lC1  # pmf

                #  mean 

                if rn1*iMk-1 <= 0:
                    print "woa.  %(rn1)d   %(imk).3f"   % {"rn1" : rn1, "imk" : iMk}
                uu0  = -_N.log(rn1 * iMk - 1) # mean of reverse proposal density
                lmd0= Mk/p0
                llmd0 = _N.log(lmd0)
                trms = _N.exp(ints[0:nmin]*llmd0 - logfact[0:nmin] - lmd0)
                lC0 = _N.log(1 - _N.sum(trms))    #  multiple by which to multiply pmf(k) to get pmf fo trunc
                lpm0 = rn0 * llmd0 - logfact[rn0] - lmd0 - lC0  # pmf

                # print C1
                # print lmd1
                # print C0
                # print lmd0
                # print "A  pm0  %(0).3e   pm1  %(1).3e" % {"0" : pm0, "1" : pm1} 
                

                # print u1
                # print uu1
                # print u0
                # print uu0
                lpPR = 0.5*istdu2*(((u1 - uu1)*(u1 - uu1)) - ((u0 - uu0)*(u0 - uu0))) + lpm0 - lpm1   #, lnc reciprocal of norm
                #print "---------%(1).4e   %(2).4e" % {"1" : _N.log(pm1/pm0), "2" : lpPR}
            else:   ########  Switch to __NBML__  ####################
                #print "switch 2 NBML"
                todist = _cd.__NBML__;   cross  = True
                u1 = 2*uTH - u1  #  u1 now a parameter of NB distribution   
                p1 = 1 / (1 + _N.exp(-u1))
                p1x = 1 / (1 + _N.exp(-(u1+xn)))
                lmd1 = (1./p1 - 1)*Mk

                llmd1 = _N.log(lmd1)
                trms = _N.exp(ints[0:rmin]*llmd1 - logfact[0:rmin] - lmd1)
                lC1 = _N.log(1 - _N.sum(trms))    #  multiple by which to multiply pmf(k) to get pmf fo trunc

                while not bDone:
                    rnds = _ss.poisson.rvs(lmd1, size=5)
                    pois = _N.where(rnds >= rmin)[0]  
                    if len(pois) > 0:
                        bDone = True
                        rn1   = rnds[pois[0]]   #  nb parameter
                lpm1 = rn1 * llmd1 - logfact[rn1] - lmd1 - lC1  # pmf

                lmd0= (1./pTH - 1) * Mk
                llmd0 = _N.log(lmd0)   #  param is BN parameter
                trms = _N.exp(ints[0:nmin]*llmd0 - logfact[0:nmin] - lmd0)
                lC0 = _N.log(1 - _N.sum(trms))    #  multiple by which to multiply pmf(k) to get pmf fo trunc
                lpm0 = rn0 * llmd0 - logfact[rn0] - lmd0 - lC0  # pmf

                #print "B  pm0  %(0).3e   pm1  %(1).3e" % {"0" : pm0, "1" : pm1}

                uu0  = -_N.log(rn1 * iMk) # mean of proposal density

                lpPR = 0.5*istdu2*((((uTH-u1) - uu1)*((uTH-u1) - uu1)) - (((uTH-u0) - uu0)*((uTH-u0) - uu0))) + lpm0 - lpm1
        elif dist == _cd.__NBML__:
            uu1  = -_N.log(rn0 * iMk) # mean of proposal density
            u1 = uu1 + stdu * rdns[it]
            #print "NBML   uu1  %(uu1).3e    u1  %(u1).3e" % {"uu1" : uu1, "u1" : u1}

            if u1 > uTH:       ######   Stay in Negative binomial ######
                todist = _cd.__NBML__;    cross  = False
                p1 = 1 / (1 + _N.exp(-u1))
                p1x = 1 / (1 + _N.exp(-(u1+xn)))
                lmd1 = (1./p1 - 1)*Mk

                llmd1 = _N.log(lmd1)
                trms = _N.exp(ints[0:rmin]*llmd1 - logfact[0:rmin] - lmd1)
                lC1 = _N.log(1 - _N.sum(trms))    #  multiple by which to multiply pmf(k) to get pmf fo trunc

                while not bDone:
                    rnds = _ss.poisson.rvs(lmd1, size=5)
                    pois = _N.where(rnds >= rmin)[0]
                    if len(pois) > 0:
                        bDone = True
                        rn1   = rnds[pois[0]]   # bn parameter

                lpm1 = rn1 * llmd1 - logfact[rn1] - lmd1 - lC1  # pmf

                #     lmd1= Mk*((1-p1)/p1);  lmd0= Mk*((1-p0)/p0);   lmd= lmd1
                uu0  = -_N.log(rn1 * iMk) # mean of proposal density
                lmd0 = (1./p0 - 1)*Mk
                llmd0 = _N.log(lmd0)
                trms = _N.exp(ints[0:rmin]*llmd0 - logfact[0:rmin] - lmd0)
                lC0 = _N.log(1 - _N.sum(trms))    #  multiple by which to multiply pmf(k) to get pmf fo trunc
                lpm0 = rn0 * llmd0 - logfact[rn0] - lmd0 - lC0  # pmf

                #print "C  pm0  %(0).3e   pm1  %(1).3e" % {"0" : pm0, "1" : pm1}
                # log of probability

                lpPR = 0.5*istdu2*(((u1 - uu1)*(u1 - uu1)) - ((u0 - uu0)*(u0 - uu0)))+ lpm0 - lpm1
            else:   ########  Switch to __BNML__  ####################
                #print "switch 2 BNML"
                todist = _cd.__BNML__;    cross  = True
                u1 = 2*uTH - u1  #  u in NB distribution
                p1 = 1 / (1 + _N.exp(-u1))
                p1x = 1 / (1 + _N.exp(-(u1+xn)))
                lmd1= Mk/p1

                #print "%(gi)d   %(lmd1).4f" % {"gi" : GibbsIter, "lmd1" : lmd1}
                #print "%(Mk)d   %(p1).4e" % {"Mk" : Mk, "p1" : p1}
                llmd1 = _N.log(lmd1)
                trms = _N.exp(ints[0:nmin]*llmd1 - logfact[0:nmin] - lmd1)
                lC1 = _N.log(1 - _N.sum(trms))    #  multiple by which to multiply pmf(k) to get pmf fo trunc

                while not bDone:
                    rnds = _ss.poisson.rvs(lmd1, size=5)
                    pois = _N.where(rnds >= nmin)[0]  
                    if len(pois) > 0:
                        bDone = True
                        rn1   = rnds[pois[0]]   #  nb parameter
                lpm1 = rn1 * llmd1 - logfact[rn1] - lmd1 - lC1  # pmf

                lmd0 = Mk/pTH
                llmd0 = _N.log(lmd0)
                trms = _N.exp(ints[0:rmin]*llmd0 - logfact[0:rmin] - lmd0)
                lC0 = _N.log(1 - _N.sum(trms))    #  multiple by which to multiply pmf(k) to get pmf fo trunc
                lpm0 = rn0 * llmd0 - logfact[rn0] - lmd0 - lC0  # pmf

                #print "D  pm0  %(0).3e   pm1  %(1).3e" % {"0" : pm0, "1" : pm1}
                uu0  = -_N.log(rn1 * iMk - 1) # mean of proposal density
                lpPR = 0.5*istdu2*(-(((uTH-u1) - uu1)*((uTH-u1) - uu1)) + (((uTH-u0) - uu0)*((uTH-u0) - uu0)))+ lpm0 - lpm1


        #dbtt2 = _tm.time()
        lFlB[0] = Llklhds(todist, cts, rn1, p1x)
        #print "proposed state  ll  %(1).3e   old state  ll  %(2).3e     new-old  %(3).3e" % {"1" : lFlB[0], "2" : lFlB[1], "3" : (lFlB[0] - lFlB[1])}
        #dbtt3 = _tm.time()
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

        #dbtt4 = _tm.time()
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
        #dbtt5 = _tm.time()
        #dbtt21 += #dbtt2-#dbtt1
        #dbtt32 += #dbtt3-#dbtt2
        #dbtt43 += #dbtt4-#dbtt3
        #dbtt54 += #dbtt5-#dbtt4

    # print "#timing start"
    # print "t2t1+=%.4e" % #dbtt21
    # print "t3t2+=%.4e" % #dbtt32
    # print "t4t3+=%.4e" % #dbtt43
    # print "t5t4+=%.4e" % #dbtt54
    # print "#timing end"

    #print "accepted %d" % accptd
    # fig = _plt.figure()
    # _plt.plot(llsV)
    # _plt.suptitle(accptd)
    # _plt.savefig("llsV%d" % GibbsIter)
    # _plt.close()
    lEn = lFlB[0]

    #print "ll Bg %(b).3e   ll En %(e).3e" % {"b" : lBg, "e" : lEn}
    return u0, rn0, dist




