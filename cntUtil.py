import numpy as _N
import scipy.misc as _sm
import commdefs as _cd
import matplotlib.pyplot as _plt

#  proposal parameters
stdu = 0.05;
stdu2= stdu**2;   
istdu2= 1./ stdu2

#  connection between 
pTH  = 0.01
uTH    = _N.log(pTH / (1 - pTH))

def Llklhds(type, ks, rn1, p1):
    try:
        if type == _cd.__BNML__:
            #return _N.sum(_N.log(_sm.comb(rn1, ks)) + ks*_N.log(p1) + (rn1-ks)*_N.log(1 - p1))
            #top = 0.5*_N.log(2*_N.pi*rn1) + rn1*_N.log(rn1) - rn1 
            #b1  = 0.5*_N.log(2*_N.pi*ks) + ks*_N.log(ks) - ks
            #b2  = 0.5*_N.log(2*_N.pi*(rn1-ks)) + (rn1-ks)*_N.log(rn1-ks) - (rn1-ks)
            #l1  = _N.sum(top - b1 - b2 + ks*_N.log(p1) + (rn1-ks)*_N.log(1 - p1))
            l2  = _N.sum(_N.log(_sm.comb(rn1, ks)) + ks*_N.log(p1) + (rn1-ks)*_N.log(1 - p1))
            #print "l1  %(1).3f   %(2).3f" % {"1" : l1, "2" : l2}
            return l2
        if type == _cd.__NBML__:
            ksrn1m1 = ks+rn1-1
            rn1m1  = rn1-1

            #top = 0.5*_N.log(2*_N.pi*(ksrn1m1)) + (ksrn1m1)*_N.log(ksrn1m1) - (ksrn1m1)
            #b1  = 0.5*_N.log(2*_N.pi*ks) + ks*_N.log(ks) - ks
            #if rn1m1 > 0:
            #    b2  = 0.5*_N.log(2*_N.pi*(rn1m1)) + (rn1m1)*_N.log(rn1m1) - (rn1m1)
            #else:
            #    b2  = 0
            #l1  = _N.sum(top - b1 - b2 + ks*_N.log(p1) + rn1*_N.log(1 - p1))
            l2  = _N.sum(_N.log(_sm.comb(ks + rn1-1, ks)) + ks*_N.log(p1) + rn1*_N.log(1 - p1))
            #print "l1  %(1).3f   %(2).3f" % {"1" : l1, "2" : l2}
            return l2
    except Warning:
        print "!!!!!!!!"
        print "type %d" % type
        print "rn1  %d" % rn1

        raise

def startingValues2(cts, trials=None, fillsmpx=None):
    #  starting values always calculated
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
    cv0   = _N.mean(cvs)
    print cv0

    mdl, p0 = (_cd.__BNML__, 1 - cv0) if cv0 < 1 else (_cd.__NBML__, 1 - 1/cv0)
    a1o0 = 1 if (mdl == _cd.__BNML__) else 0   #  add 1 or 0

    u0 = -_N.log(1/p0 - 1)
    ###  estimate rn0
    for ep in xrange(Npcs):
        rns[ep]  = (mns[ep] / p0) if mdl == _cd.__BNML__ else (1/p0 - 1)*mns[ep]

    rn0 = int(_N.mean(rns))

    ##########################  estimate xn
    xn     = _N.array(mns)
    #print rn0

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
    fillsmpx[:] = xoff
    xn   *= mags[maxJ]
    p1x = 1 / (1 + _N.exp(-(u0 + xoff)))

    llsV = _N.empty(20)
    bestRN = bestrn(mdl, cts, rn0, llsV, p1x)

    return u0, bestRN, mdl
    #  For cv~1, r~n 

def startingValues(cts, xoff=None, trials=None, fillsmpx=None):
    #  starting values always calculated
    Nss = len(cts)  #  N same state
    Mk = _N.mean(cts)  #  1comp if nWins=1, 2comp
    iMk = 1./Mk
    sdk = _N.std(cts)
    cv = ((sdk*sdk) * iMk)
    nmin= _N.max(cts)   #  if n too small, can't generate data
    rmin= 1

    epS  = 20
    ####  guess initial value
    #  estimate u0
    Npcs = (Nss) / epS   #  20 trials each
    cvs  = _N.empty(Npcs)
    mns  = _N.empty(Npcs)
    rn0s = _N.empty(Npcs)
    rn1s = _N.empty(Npcs)

    for ep in xrange(Npcs):
        mp       = _N.mean(cts[ep*epS:(ep+1)*epS]) # OK even if overshoot 
        cp       = _N.std(cts[ep*epS:(ep+1)*epS])**2 / mp
        mns[ep]  = mp
        cvs[ep]  = cp

    #print cvs
    cv0   = _N.mean(cvs)
    #print cv0

    mdl, p0 = (_cd.__BNML__, 1 - cv0) if cv0 < 1 else (_cd.__NBML__, 1 - 1/cv0)
    u0 = -_N.log(1/p0 - 1)

    xn     = _N.empty(Npcs)

    if xoff is None:
        xoff   = _N.empty(Nss)

        for ep in xrange(Npcs):
            xn[ep] = _N.mean(cts[ep*epS:(ep+1)*epS])
            xoff[ep*epS:(ep+1)*epS] = xn[ep]
        xoff[Npcs*epS:] = xn[Npcs-1]   #  if theres some values left at end

        xoff   -= _N.mean(xoff)
        xoff   /= 3*_N.std(xoff) #  xOff initially set to be 0 mean, std = 1
        xn        -= _N.mean(xn)
        xn       /= 3*_N.std(xn)    #  xOff initially set to be 0 mean, std = 1
        fillsmpx[:] = xoff
    else:  #  we were handed a latent state
        for ep in xrange(Npcs):
            xn[ep] = _N.mean(xoff[ep*epS:(ep+1)*epS])

    a1o0 = 1 if (mdl == _cd.__BNML__) else 0   #  add 1 or 0
    for ep in xrange(Npcs):
        rn0s[ep] = _N.mean(cts[ep*epS:(ep+1)*epS]) * (a1o0 + _N.exp(-(u0 + xn[ep])))

    rn0    = int(_N.mean(rn0s))

    #  For cv~1, r~n 

    #  let's start it off from binomial
    p0x  = 1 / (1 + _N.exp(-(u0 + xoff)))  #  p0 
    if mdl == _cd.__BNML__:  #  make sure initial distribution is capable of generating data
        while rn0 < nmin:
            rn0 += 1
    else:
        while rn0 < rmin:
            rn0 += 1

    #print "model   is  %(m)d   rn0 is %(r)d" % {"m" : mdl, "r" : rn0}
    rnLo = int(rn0 * 0.9)
    rnHi = int(rn0 * 1.1)
    candRNs = range(rnLo, rnHi)
    lls = _N.empty(rnHi-rnLo)
    for tryRN in candRNs:
        lls[tryRN-rnLo] = Llklhds(mdl, cts, tryRN, p0x)
    maxI = _N.where(_N.max(lls) == lls)[0][0]
    return u0, candRNs[maxI], mdl

def bestrn(dist, cnt, lmd0, llsV, p1x):
    #print "bestrn    lmd0 %d" % lmd0
    dn   = int(lmd0*0.02)
    dn   = 1 if dn == 0 else dn

    n0   = lmd0 - 10*dn
    n1   = lmd0 + 10*dn
    n0   = 1 if n0 < 1 else n0
    
    candRNs = range(n0, n1, dn)
    
    j    = 0

    for tryRN in candRNs:
        llsV[j] = Llklhds(dist, cnt, tryRN, p1x)
        j += 1
    maxI = _N.where(_N.max(llsV) == llsV)[0][0]
    #print "maxI   %d" % maxI
    return candRNs[maxI]


def cntmdlMCMCOnlyU(iters, u0, rn0, dist, cts, rns, us, dty, pcdlog, xn):
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

    llsV = _N.empty(20)
    rn0 = bestrn(dist, cts, rn0, llsV, p0x)

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
                #print "here"
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

        lnPR = 0
        lPR = lnPR + lpPR

        if lPR > 50:
            prRat = 1e+10
        else:
            prRat = _N.exp(lPR)

        #  lFlB[0] - lFlB[1] >> 0  -->  new state has higher likelihood
        posRat = 1.01e+200 if (lFlB[0] - lFlB[1] > 500) else _N.exp(lFlB[0]-lFlB[1])

        #print "posRat %(1).3e     prRat %(2).3e" % {"1" : posRat, "2" : prRat}
        rat  = posRat*prRat

        aln  = rat if (rat < 1)  else 1   #  if aln == 1, always accept
        if rds[it] < aln:   #  accept
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

    lEn = lFlB[0]

    print "ll Bg %(b).3e   ll En %(e).3e" % {"b" : lBg, "e" : lEn}
    return u0, rn0, dist

