import scipy.misc as _sm
import scipy.stats as _ss
import numpy as _N
import matplotlib.pyplot as _plt

import warnings
warnings.filterwarnings("error")

__BNML__ = 0   # binomial
__NBML__ = 1   # negative binomial

epS      = 5

#  proposal parameters
stdu = 0.3;    
stdu2= stdu**2;   
istdu2= 1./ stdu2

#  connection between 
pTH  = 0.01
uTH    = _N.log(pTH / (1 - pTH))

def Llklhds(type, ks, rn1, p1):
    try:
        if type == __BNML__:
            return _N.sum(_N.log(_sm.comb(rn1, ks)) + ks*_N.log(p1) + (rn1-ks)*_N.log(1 - p1))
        if type == __NBML__:
            return _N.sum(_N.log(_sm.comb(ks + rn1-1, ks)) + ks*_N.log(p1) + rn1*_N.log(1 - p1))
    except Warning:
        print "!!!!!!!!"
        print type
        print rn1
        print ks
        print p1

        raise

def trPoi(lmd, a):
    """
    a, b inclusive
    """
    ct = _N.random.poisson(lmd)
    while (ct < a):  #  accept ct == a.
        ct = _N.random.poisson(lmd)
    return ct

def MCMC(burn, NMC, cts, rns, us, dty, pcdlog, xn=None, cv0=0.99):
    N   = cts.shape[0]
    Mk  = _N.mean(cts)
    iMk = 1./Mk
    sdk = _N.std(cts)
    cv = ((sdk*sdk) * iMk)
    nmin= _N.max(cts)   #  if n too small, can't generate data
    rmin= 1

    ####  guess initial value
    #  estimate u0
    Npcs = N / epS   #  20 trials each
    cvs  = _N.empty(Npcs)
    mns  = _N.empty(Npcs)
    rn0s = _N.empty(Npcs)
    rn1s = _N.empty(Npcs)

    for ep in xrange(Npcs):
        mp       = _N.mean(cts[ep*epS:(ep+1)*epS])
        cp       = _N.std(cts[ep*epS:(ep+1)*epS])**2 / mp
        mns[ep]  = mp
        cvs[ep]  = cp

    cv0   = _N.mean(cvs)

    if cv0 < 1:
        dist = __BNML__
        p0 = 1 - cv0
    else:
        dist = __NBML__
        p0 = 1 - 1/cv0
    #p0 = 0.5
    u0 = -_N.log(1/p0 - 1)
    #xOff   = _N.empty(Npcs)   #  used only to find good initial value
    xOff   = _N.empty(N)   #  used only to find good initial value

    print "Initial distribution is %d" % dist
    print "cv0 is %f" % cv0
    print "p0 is %f" % p0

    if xn is None:  #  once we get going, xn will 
        xnEp   = _N.empty(Npcs)
        for ep in xrange(Npcs):
            xnEp[ep] = _N.mean(cts[ep*epS:(ep+1)*epS])

        estSTDxn  = _N.std(mns) / _N.mean(mns)   #  the approximate 
        # xOff *= estSTDxn / _N.std(xOff)
        # xn   = _N.empty(N)
        # for ep in xrange(Npcs):
        #     xn[ep*epS:(ep+1)*epS] = xOff[ep]
        # xn[Npcs*epS:] = xOff[Npcs-1]
    else:
        #for ep in xrange(Npcs):
        #    xOff[ep] = _N.mean(xn[ep*epS:(ep+1)*epS])
        for n in xrange(N):
            xOff[n] = xn[n]

    if dist == __BNML__:
        for ep in xrange(Npcs):
            rn0s[ep] = _N.mean(cts[ep*epS:(ep+1)*epS]) * (1 + _N.exp(-(u0 + xOff[ep])))
    else:
        for ep in xrange(Npcs):
            rn0s[ep] = _N.mean(cts[ep*epS:(ep+1)*epS]) * _N.exp(-(u0 + xOff[ep]))
    rn0    = int(_N.mean(rn0s))
    #  For cv~1, r~n 

    #  let's start it off from binomial
    p0x  = 1 / (1 + _N.exp(-(u0 + xn)))  #  p0 
    if dist == __BNML__:  #  make sure initial distribution is capable of generating data
        while rn0 < nmin:
            rn0 += 1
    else:
        while rn0 < rmin:
            rn0 += 1
    p0x = _N.exp(u0 + xOff) / (1 + _N.exp(u0+xOff))
    lls = _N.empty(25)
    for i in xrange(-12, 13):
        lls[i+12] = Llklhds(dist, cts, rn0+i, p0x)
    maxI = _N.where(_N.max(lls) == lls)[0][0]
    rn0 += -12 + maxI


    lFlB = _N.empty(2)
    rn1rn0 = _N.empty(2)

    rds  = _N.random.rand(burn+NMC)
    rdns = _N.random.randn(burn+NMC)

    lFlB[1] = Llklhds(dist, cts, rn0, p0x)

    cross  = False
    print "nmin is %d" %  nmin
    lls   = []
    print "rn0 is %d" % rn0

    print "uTH is %.3e" % uTH
    for it in xrange(burn + NMC):
        #print "------------  it %d" % it
        if dist == __BNML__:
            #print "BNML"
            uu1  = -_N.log(rn0 * iMk - 1) # mean of proposal density
            u1 = uu1 + stdu * rdns[it]
            #print "BNML   uu1  %(uu1).3e    u1  %(u1).3e" % {"uu1" : uu1, "u1" : u1}

            if u1 > uTH:       ###########   Stay in Binomial ##########
                todist = __BNML__;    cross  = False
                p1 = 1 / (1 + _N.exp(-u1))
                p1x = 1 / (1 + _N.exp(-(u1+xn)))
                lmd= Mk/p1
                rn1 = trPoi(lmd, nmin)   #  mean is p0/Mk
                if rn1 < nmin:
                    "rn1 is less than nmin.  Why?"
                bLargeP = (p0 > 0.3) and (p1 > 0.3)
                if bLargeP:#    fairly large p.  Exact proposal ratio
                    lmd = Mk/(0.5*(p0+p1))
                else:          #  small p.  prop. ratio far from lower lim of n
                    lmd1= Mk/p1;     lmd0= Mk/p0;     lmd = lmd1
                uu0  = -_N.log(rn1 * iMk - 1) # mean of proposal density
                # log of probability
                lpPR = 0.5*istdu2*(-((u1 - uu1)*(u1 - uu1)) + ((u0 - uu0)*(u0 - uu0)))  #  - (lnc0 - lnc1), lnc reciprocal of norm
            else:   ########  Switch to __NBML__  ####################
                todist = __NBML__;   cross  = True
                u1 = 2*uTH - u1  #  u1 now a parameter of NB distribution   
                p1 = 1 / (1 + _N.exp(-u1))
                p1x = 1 / (1 + _N.exp(-(u1+xn)))
                lmd = (1./p1 - 1)*Mk
                rn1 = trPoi(lmd, rmin)   #  mean is p0/Mk
                lmd1= Mk*((1-p1)/p1);  lmd0= Mk/p0;   lmd= lmd1
                uu0  = -_N.log(rn1 * iMk) # mean of proposal density
                lpPR = 0.5*istdu2*(-(((uTH-u1) - uu1)*((uTH-u1) - uu1)) + (((uTH-u0) - uu0)*((uTH-u0) - uu0)))
        elif dist == __NBML__:
            uu1  = -_N.log(rn0 * iMk) # mean of proposal density
            u1 = uu1 + stdu * rdns[it]
            #print "NBML   uu1  %(uu1).3e    u1  %(u1).3e" % {"uu1" : uu1, "u1" : u1}

            if u1 > uTH:       ######   Stay in Negative binomial ######
                todist = __NBML__;    cross  = False
                p1 = 1 / (1 + _N.exp(-u1))
                p1x = 1 / (1 + _N.exp(-(u1+xn)))

                lmd = (1./p1 - 1)*Mk
                rn1 = trPoi(lmd, rmin)   #  mean is p0/Mk
                bLargeP = (p0 > 0.3) and (p1 > 0.3)
                if bLargeP:#    fairly large p.  Exact proposal ratio
                    lmd= Mk*((1-0.5*(p0+p1))/(0.5*(p0+p1)))
                else:          #  small p.  prop. ratio far from lower lim of n
                    lmd1= Mk*((1-p1)/p1);  lmd0= Mk*((1-p0)/p0);   lmd= lmd1
                uu0  = -_N.log(rn1 * iMk) # mean of proposal density
                # log of probability
                lpPR = 0.5*istdu2*(-((u1 - uu1)*(u1 - uu1)) + ((u0 - uu0)*(u0 - uu0)))
            else:   ########  Switch to __BNML__  ####################
                #print "here"
                todist = __BNML__;    cross  = True
                u1 = 2*uTH - u1  #  u in NB distribution
                p1 = 1 / (1 + _N.exp(-u1))
                p1x = 1 / (1 + _N.exp(-(u1+xn)))
                lmd = Mk/p1
                rn1 = trPoi(lmd, nmin)   #  mean is p0/Mk
                lmd1= Mk/p1;     lmd0= Mk*((1-p0)/p0);     lmd = lmd1
                uu0  = -_N.log(rn1 * iMk - 1) # mean of proposal density
                lpPR = 0.5*istdu2*(-(((uTH-u1) - uu1)*((uTH-u1) - uu1)) + (((uTH-u0) - uu0)*((uTH-u0) - uu0)))
        
        lFlB[0] = Llklhds(todist, cts, rn1, p1x)
        rn1rn0[0] = rn1;                   rn1rn0[1] = rn0

        ########  log of proposal probabilities

        if cross or not bLargeP:  #  lnPR can be calculated without regard to whether cross or not, because it is conditionally dependent on pN, pB
            #print "in cross or not bLargeP  %d" % it
            if rn0 == rn1:
                lnPR = rn1*(_N.log(lmd1) - _N.log(lmd0)) - (lmd1 - lmd0) 
            else:
                if rn0 > rn1: # range(n1+1, n0+1)
                    lnPR = rn1*_N.log(lmd1) - rn0*_N.log(lmd0) + _N.sum(pcdlog[rn1+1:rn0+1]) - (lmd1 - lmd0)
                else:
                    lnPR = rn1*_N.log(lmd1) - rn0*_N.log(lmd0) - _N.sum(pcdlog[rn0+1:rn1+1]) - (lmd1 - lmd0)
        else:
            if rn0 == rn1:
                lnPR = 0  #  r0 == r1
            else:
                if rn0 > rn1: # range(r1+1, r0+1)
                    lnPR = (rn1-rn0) * _N.log(lmd) + _N.sum(pcdlog[rn1+1:rn0+1])
                else:
                    lnPR = (rn1-rn0) * _N.log(lmd) - _N.sum(pcdlog[rn0+1:rn1+1])

        lPR = lnPR + lpPR

        if lPR > 50:
            prRat = 1e+10
        else:
            prRat = _N.exp(lPR)

        posRat = 1.01e+304 if (lFlB[0] - lFlB[1] > 700) else _N.exp(lFlB[0]-lFlB[1])

        rat  = posRat*prRat

        aln  = rat if (rat < 1)  else 1
        # if dist == __BNML__:
        #     cvI = 1 - p0
        # else:
        #     cvI = 1 / (1 - p0)
        # if todist == __BNML__:
        #     cvF = 1 - p1
        # else:
        #     cvF = 1 / (1 - p1)

        # if cvF < cvI:
        #print "cvI %(cvi).2f  cvF  %(cvf).2f  %(it)d   %(aln).5f   posRat %(posRat).3f    lnPR %(lnPR).3f   lpPR %(lpPR).3f" % {"it" : it, "aln" : aln, "posRat" : posRat, "lnPR" : lnPR, "lpPR" : lpPR, "cvi" : cvI, "cvf" : cvF}
        #print "%(it)d   %(aln).5f   posRat %(posRat).3f    lnPR %(lnPR).3f   lpPR %(lpPR).3f" % {"it" : it, "aln" : aln, "posRat" : posRat, "lnPR" : lnPR, "lpPR" : lpPR}
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
    return lls

