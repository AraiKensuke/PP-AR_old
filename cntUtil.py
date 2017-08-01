import numpy as _N
import scipy.misc as _sm
import commdefs as _cd
import matplotlib.pyplot as _plt
import time as _tm
import scipy.stats as _ss
import os

#  connection between 

logfact= None

ks = _N.arange(2000000)

uTH1= -6.5
uTH2= -1

uTHa= -6.5
uTHb= -1

ln2pi= 1.8378770664093453
ks = _N.arange(2000000)
maxrn = -1

mn_u = 0
#iu_sd2 = (1/1.2)*(1/1.2)
iu_sd2 = (1/1.4)*(1/1.4)


def _init(lf):
    global logfact, maxrn
    logfact = lf
    maxrn   = len(logfact) - 1000

def Llklhds(typ, ks, rn1, p1):
    global logfact
    N = len(ks)
    if typ == _cd.__BNML__:
        return N*logfact[rn1]-_N.sum(logfact[ks]+logfact[rn1-ks]-ks*_N.log(p1) - (rn1-ks)*_N.log(1 - p1))
    else:
        return _N.sum(logfact[ks+rn1-1]-logfact[ks]  + ks*_N.log(p1) + rn1*_N.log(1 - p1))-N*logfact[rn1-1]

def startingValues(cts, w, lh, fillsmpx=None, cv0=None, trials=None):
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
        if mns[ep] > 0:
            cvs[ep]       = _N.std(cts[ep*epS:(ep+1)*epS])**2 / mns[ep]
        else:
            cvs[ep]       = 0
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
    bestRN = bestrn(mdl, cts, rn0, llsV, p1x)
    fig = _plt.figure(figsize=(5, 8))
    ax  = fig.add_subplot(2, 1, 1)
    _plt.hist(cts, bins=_N.linspace(0, 50, 51))
    ax  = fig.add_subplot(2, 1, 2)
    _plt.plot(llsV)
    FF  = _N.std(cts)**2/_N.mean(cts)
    _plt.suptitle("w %(w)d  lh %(lh)d    bestRN %(br)d   FF %(FF).3f" % {"br" : bestRN, "FF" : FF, "w" : w, "lh" : lh})
    
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
        if WNS > 1:
            loInds = _N.where(_N.mean(zsW[:, :, 0], axis=0) >= 0.55)[0]
            hiInds = _N.where(_N.mean(zsW[:, :, 0], axis=0) <  0.45)[0]
        else:
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
            print "w is %(w)d    j is %(j)d" % {"w" : w, "j" : j}
            trls = _N.where(zs[:, j] == 1)[0]

            #  1 / (1-p) = c      1/c = 1-p   p = 1 - 1/c
            u0, bestRN, mdl = startingValues(cts[:, w], w, j, fillsmpx=fs[w], trials=trls)
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
    n0   = 1 if n0 < 1 else n0
    n1   = n0 + 16*dn
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



#def BNorNB(GibbsIter, iters, w, j, u0, rn0, dist, cts, rns, us, dty, xn, jxs, mvtyps, lls, accpts, llklhd0, llklhd1, lpru0s, lpru1s, lpriorpmf1s, lpriorpmf0s, ljacs, lrats, lpprs0, lpprs1, stdu=0.03, propratm=0):
def BNorNB(iters, w, j, u0, rn0, dist, cts, rns, us, dty, xn, stdu=0.03):
    global logfact, ks, maxrn
    #  if accptd too small, increase stdu and try again
    #  if accptd is 0, the sampled params returned for conditional posterior
    #  are not representative of the conditional posterior
    stdu2= stdu**2
    istdu2= 1./ stdu2
    #stdp2 = 0.25*0.25
    #stdp = 0.002
    stdp = 0.05
    istdp= 1./stdp
    istdp2= istdp*istdp

    Mk = _N.mean(cts) if len(cts) > 0 else 0  #  1comp if nWins=1, 2comp
    if Mk == 0:
        return u0, rn0, dist   # no data assigned to this 
    rnmin = _N.array([-1, _N.max(cts)+1, 1], dtype=_N.int)

    rdis= _N.random.rand(iters)  #
    rdns = _N.random.randn(iters)
    ran_accpt  = _N.random.rand(iters)

    p0  = 1./(1 + _N.exp(-u0))
    p0x  = 1./(1 + _N.exp(-(u0+xn)))

    ll0 = Llklhds(dist, cts, rn0, p0x)
    #  the poisson distribution needs to be truncated
    # prop_us = _N.empty(iters)
    # prop_rns = _N.empty(iters, dtype=_N.int)
    # prop_dty = _N.empty(iters, dtype=_N.int)

    zr2rnmins  = _N.array([None, _N.arange(rnmin[1]), _N.arange(rnmin[2])]) # rnmin is a valid value for n

    u_m     = (uTH1+uTH2)*0.5
    mag     = 4./(uTH2 - uTH1)

    #  TO DO:  Large cnts usually means large rn.  
    #  for transitions with small ps, the rn we need to transition on the
    #  other side might be very large where the prior prob. is very small
    r_prior = 1
    p_prior = 0.9996   

    accptd  = 0

    lpriorpmf = logfact[ks + r_prior - 1] - logfact[r_prior-1] - logfact[ks] + r_prior*_N.log(1-p_prior) + ks*_N.log(p_prior)   #  prior

    for it in xrange(iters):
        us[it] = u0
        rns[it] = rn0    #  rn0 is the newly sampled value if accepted
        dty[it] = dist

        ut    = (u0 - u_m)*mag
        jx    = 0.9 / (1 + _N.exp(2*ut))
        #
        #dbtt1 = _tm.time()

        if it % 1000 == 0:
            print it

        if _N.random.rand() < jx:  #  JUMP
            todist = _cd.__NBML__ if dist == _cd.__BNML__ else _cd.__BNML__
            mv = 0
            #  jump
            #print "here  %(it)d   %(jx).3f" % {"it" : it, "jx" : jx}
            u1 = (uTHa + uTHb) - u0
            # prop_us[it] = u1
            # prop_dty[it]= todist
            ip1 = 1 + _N.exp(-u1)
            p0  = 1 / (1 + _N.exp(-u0))
            p1  = 1 / (1 + _N.exp(-u1))
            p1x = 1 / (1 + _N.exp(-(u1+xn)))

            #rr   = rn0*p0/p1 
            rr   = ((rn0*p0) / (p1*(1-p0))) if dist == _cd.__NBML__ else ((rn0*p0*(1-p1))/p1)
            irr  = int(rr)
            rmdr = rr-irr
            rn1   = int(rr)

            if _N.random.rand() < rmdr:
                rn1 += 1
            #prop_rns[it] = rn1

            p1x = 1 / (1 + _N.exp(-(u1+xn)))
            #lpPR   = _N.log((uTH1 - u0) / (uTH1 - u1))         #  deterministic crossing.  Jac = 1
            utr = (u1 - u_m)*mag
            #utr = (uTH1 + uTH2)-u1
            jxr = 0.9 / (1 + _N.exp(2*utr))
            #lpPR   = _N.log(jxr/jx)
            if todist == _cd.__NBML__: #  r'p'/(1-p')=np, r' = np x (1-p')/p'
                ljac = _N.log((p0*(1-p1)) / p1)
            else:   #  n'p'=rp/(1-p),  n' = rp/[p'(1-p)]
                ljac = _N.log(p0 / (p1*(1-p0)))                

            # if dist == _cd.__BNML__:
            #     ljac = _N.log((jxr/jx) * (p0 / (p1*(1-p1))))
            # else:
            #     ljac = _N.log((jxr/jx) * ((p0*(1-p1)) / p1))
            lprop0 = lprop1 = 0  #

            lpru0   = -0.5*(u0 - mn_u)*(u0 - mn_u)*iu_sd2
            lpru1   = -0.5*(u1 - mn_u)*(u1 - mn_u)*iu_sd2

            #ljac   = _N.log((jxr/jx) * (p0/p1))
        else:    #  ########   DIFFUSION    ############
            todist = dist
            #prop_dty[it]= todist

            zr2rnmin = zr2rnmins[dist]

            ##  propose an rn1 
            m2          = 1./rn0 + 0.9      # rn0 is the mean for proposal for rn1
            p_prp_rn1        = 1 - 1./(rn0*m2)  # param p for proposal for rn1
            r_prp_rn1        = rn0 / (rn0*m2-1) # param r for proposal for rn1
            ir_prp_rn1 = int(r_prp_rn1)

            bGood = False   #  rejection sampling of rn1
            while not bGood:
                rn1 = _N.random.negative_binomial(ir_prp_rn1, 1-p_prp_rn1)
                if (rn1 >= rnmin[dist]) and (rn1 < maxrn):
                    bGood = True  # rn1 < maxrn  - just throw away huge rn
            #prop_rns[it] = rn1

            #########  log proposal density for rn1
            ir_prp_rn1 = int(r_prp_rn1)
            ltrms = logfact[zr2rnmin+ir_prp_rn1-1]  - logfact[ir_prp_rn1-1] - logfact[zr2rnmin] + ir_prp_rn1*_N.log(1-p_prp_rn1) + zr2rnmin*_N.log(p_prp_rn1)
            lCnb1        = _N.log(1 - _N.sum(_N.exp(ltrms)))  #  nrmlzation 4 truncated pmf

            lpmf1       = logfact[rn1+ir_prp_rn1-1]  - logfact[ir_prp_rn1-1] - logfact[rn1] + r_prp_rn1*_N.log(1-p_prp_rn1) + rn1*_N.log(p_prp_rn1) - lCnb1

            #########  log proposal density for rn0
            ##  rn1
            m2          = 1./rn1 + 0.9      # rn0 is the mean for proposal for rn1
            p_prp_rn0        = 1 - 1./(rn1*m2)  # param p for proposal for rn1
            r_prp_rn0        = rn1 / (rn1*m2-1.) # param r for proposal for rn1
            ir_prp_rn0 = int(r_prp_rn0)

            ltrms = logfact[zr2rnmin+ir_prp_rn0-1]  - logfact[ir_prp_rn0-1] - logfact[zr2rnmin] + ir_prp_rn0*_N.log(1-p_prp_rn0) + zr2rnmin*_N.log(p_prp_rn0)
            smelt = _N.sum(_N.exp(ltrms))
            # print "------"
            # print "%.5f" % smelt
            # print dist
            # print todist
            # print ir_prp_rn0
            # print p_prp_rn0
            # print zr2rnmin
            lCnb0        = _N.log(1 - _N.sum(_N.exp(ltrms)))  #  nrmlzation 4 truncated 

            lpmf0       = logfact[rn0+ir_prp_rn0-1]  - logfact[ir_prp_rn0-1] - logfact[rn0] + r_prp_rn0*_N.log(1-p_prp_rn0) + rn0*_N.log(p_prp_rn0) - lCnb0

            ###################################################
            #  propose p1
            #  sample using p from [0, 0.75]  mean 0.25, sqrt(variance) = 0.25
            #  p1 x rn1 = p0 x rn0      --> p1 = p0 x rn0/rn1   BINOMIAL
            #  

            try:
                mn_u1 = -_N.log(rn1 * (1+_N.exp(-u0))/rn0 - 1) if dist == _cd.__BNML__ else (u0 - _N.log(float(rn1)/rn0))
            except Warning:
                #  if rn0 >> rn1, (rn0/rn1) >> 1.   p0 x (rn0/rn1) could be > 1
                print "restart  %(it)d   todist   %(to)d   dist %(fr)d   rn0 %(rn0)d >> rn1 %(rn1)d" % {"to" : todist, "fr" : dist, "it" : it, "rn0" : rn0, "rn1" : rn1}
                mn_u1  = 0


            u1          = mn_u1 + stdu*rdns[it]
            p1x = 1 / (1 + _N.exp(-(u1+xn)))
            p1  = 1/(1 + _N.exp(-u1))

            utr = (u1 - u_m)*mag
            jxr = 0.9 / (1 + _N.exp(2*utr))

            #prop_us[it] = u1
            mn_u0 = -_N.log(rn0 * (1+_N.exp(-u1))/rn1 - 1) if dist == _cd.__BNML__ else (u1 - _N.log(float(rn0)/rn1))

            lprop1 = -0.5*(u1 - mn_u1)*(u1-mn_u1)*istdu2 + lpmf1 + _N.log(1-jx) #  forward
            lprop0 = -0.5*(u0 - mn_u0)*(u0-mn_u0)*istdu2 + lpmf0 + _N.log(1-jxr) #  backwards

            if _cd.__BNML__:
                ratn    = float(rn1)/rn0
                bot     = (ratn*(1+_N.exp(-u0))-1)
                if bot == 0:
                    ljac = -30
                else:
                    logme   = _N.abs((ratn*_N.exp(-u0)) / bot)
                    ljac    = _N.log(logme)
            else:
                ljac    = 0

            lpru0   = -0.5*(u0 - mn_u)*(u0 - mn_u)*iu_sd2
            lpru1   = -0.5*(u1 - mn_u)*(u1 - mn_u)*iu_sd2
        ll1 = Llklhds(todist, cts, rn1, p1x)  #  forward

        lrat = ll1 - ll0 + lpru1 - lpru0 + lprop0 - lprop1 + ljac

        aln   = 1 if (lrat > 0) else _N.exp(lrat)
        accpts = ran_accpt[it] < aln
        if accpts:   #  accept
            u0 = u1
            rn0 = rn1
            ll0 = ll1
            p0  = p1
            dist=todist
            accptd += 1

    return u0, rn0, dist, accptd
