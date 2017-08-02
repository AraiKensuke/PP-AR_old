import numpy as _N
cimport numpy as _N
import scipy.misc as _sm
import commdefs as _cd
import matplotlib.pyplot as _plt
import time as _tm
import scipy.stats as _ss
import os
from libc.math cimport exp, sqrt, log, abs

logfact= None
cdef double[::1] v_logfact
cdef double* p_logfact

cdef double uTH1= -6.5
cdef double uTH2= -1
cdef double uTH1_pl_uTH2 = uTH1 + uTH2

cdef double uTHa= -6.5
cdef double uTHb= -1

cdef double ln2pi= 1.8378770664093453
cdef int maxrn = -1

cdef double mn_u = 0
#iu_sd2 = (1/1.2)*(1/1.2)
cdef double iu_sd2 = (1/1.4)*(1/1.4)


def _init(lf):
    global logfact, maxrn, p_logfact, v_logfact
    logfact = lf
    v_logfact = logfact
    p_logfact = &v_logfact[0]
    maxrn   = len(logfact) - 1000

def Llklhds(long typ, long N, ks, long rn1, p1):
    global logfact, p_logfact

    if typ == _cd.__BNML__:
        return N*p_logfact[rn1]-_N.sum(logfact[ks]+logfact[rn1-ks]-ks*_N.log(p1) - (rn1-ks)*_N.log(1 - p1))
    else:
        return _N.sum(logfact[ks+rn1-1]-logfact[ks]  + ks*_N.log(p1) + rn1*_N.log(1 - p1))-N*p_logfact[rn1-1]

def BNorNB(long iters, long w, long j, double u0, long rn0, long dist, cts, xn, double stdu):
    global logfact, ks, maxrn, uTH1, uTH2, iu_sd2, uTH1_pl_uTH2
    #  if accptd too small, increase stdu and try again
    #  if accptd is 0, the sampled params returned for conditional posterior
    #  are not representative of the conditional posterior
    cdef double stdu2= stdu**2
    cdef double istdu2= 1./ stdu2
    cdef long N = len(cts)

    cdef double Mk = _N.mean(cts) if len(cts) > 0 else 0  #  1comp if nWins=1, 2comp
    if Mk == 0:
        return u0, rn0, dist   # no data assigned to this 
    rnmin = _N.array([-1, _N.max(cts)+1, 1], dtype=_N.int)

    rdns = _N.random.randn(iters)      #  prop u1
    rjxs = _N.random.rand(iters)      #  which move type
    rrmdr = _N.random.rand(iters)      #  which move type
    ran_accpt  = _N.random.rand(iters)

    cdef double[::1] v_rjxs = rjxs
    cdef double[::1] v_rdns = rdns
    cdef double[::1] v_ran_accpt = ran_accpt
    cdef double[::1] v_rrmdr = rrmdr

    cdef double p0  = 1./(1 + exp(-u0))
    p0x  = 1./(1 + _N.exp(-(u0+xn)))    #  array

    cdef double ll0 = Llklhds(dist, N, cts, rn0, p0x)
    #  the poisson distribution needs to be truncated

    zr2rnmins  = _N.array([None, _N.arange(rnmin[1]), _N.arange(rnmin[2])]) # rnmin is a valid value for n

    cdef double u_m     = (uTH1+uTH2)*0.5
    cdef double mag     = 4./(uTH2 - uTH1)

    #  TO DO:  Large cnts usually means large rn.  
    #  for transitions with small ps, the rn we need to transition on the
    #  other side might be very large where the prior prob. is very small
    cdef long accptd  = 0
    
    cdef double ut
    cdef double jx, jxr
    cdef long it
    cdef double u1
    cdef double p1, ip0, ip1
    cdef long rn1
    cdef double lprop0, lprop1, ljac, ll1, lpru0, lpru1

    for it in xrange(iters):
        # us[it] = u0
        # rns[it] = rn0    #  rn0 is the newly sampled value if accepted
        # dty[it] = dist

        ut    = (u0 - u_m)*mag
        jx    = 0.9 / (1 + exp(2*ut))
        #
        #dbtt1 = _tm.time()

        if it % 1000 == 0:
            print it

        if v_rjxs[it] < jx:  #  JUMP
            todist = _cd.__NBML__ if dist == _cd.__BNML__ else _cd.__BNML__
            #  jump
            #print "here  %(it)d   %(jx).3f" % {"it" : it, "jx" : jx}
            u1 = (uTH1_pl_uTH2) - u0
            ip1 = 1 + exp(-u1)
            p0  = 1 / (1 + exp(-u0))
            p1  = 1 / (1 + exp(-u1))
            p1x = 1 / (1 + _N.exp(-(u1+xn)))

            rr   = ((rn0*p0) / (p1*(1-p0))) if dist == _cd.__NBML__ else ((rn0*p0*(1-p1))/p1)
            irr  = int(rr)
            rmdr = rr-irr
            rn1   = int(rr)

            if v_rrmdr[it] < rmdr:
                rn1 += 1

            #lpPR   = _N.log((uTH1 - u0) / (uTH1 - u1))         #  deterministic crossing.  Jac = 1
            utr = (u1 - u_m)*mag
            #utr = (uTH1_pl_uTH2)-u1
            jxr = 0.9 / (1 + _N.exp(2*utr))
            #lpPR   = _N.log(jxr/jx)
            if todist == _cd.__NBML__: #  r'p'/(1-p')=np, r' = np x (1-p')/p'
                ljac = log((p0*(1-p1)) / p1)
            else:   #  n'p'=rp/(1-p),  n' = rp/[p'(1-p)]
                ljac = log(p0 / (p1*(1-p0)))                

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
                #  if rn0 >> rn1, (rn0/rn1) >> 1.   p0 x (rn0/rn1) could be > 1.
                print "restart  %(it)d   todist   %(to)d   dist %(fr)d   rn0 %(rn0)d >> rn1 %(rn1)d" % {"to" : todist, "fr" : dist, "it" : it, "rn0" : rn0, "rn1" : rn1}
                mn_u1  = 0


            u1          = mn_u1 + stdu*rdns[it]
            p1x = 1 / (1 + _N.exp(-(u1+xn)))
            p1  = 1/(1 + exp(-u1))

            utr = (u1 - u_m)*mag
            jxr = 0.9 / (1 + exp(2*utr))

            #prop_us[it] = u1
            mn_u0 = -log(rn0 * (1+exp(-u1))/rn1 - 1) if dist == _cd.__BNML__ else (u1 - log(float(rn0)/rn1))

            lprop1 = -0.5*(u1 - mn_u1)*(u1-mn_u1)*istdu2 + lpmf1 + log(1-jx) #  forward
            lprop0 = -0.5*(u0 - mn_u0)*(u0-mn_u0)*istdu2 + lpmf0 + log(1-jxr) #  backwards

            if _cd.__BNML__:
                ratn    = float(rn1)/rn0
                bot     = (ratn*(1+exp(-u0))-1)
                if bot == 0:
                    ljac = -30
                else:
                    logme   = abs((ratn*exp(-u0)) / bot)
                    ljac    = log(logme)
            else:
                ljac    = 0

            lpru0   = -0.5*(u0 - mn_u)*(u0 - mn_u)*iu_sd2
            lpru1   = -0.5*(u1 - mn_u)*(u1 - mn_u)*iu_sd2
        ll1 = Llklhds(todist, N, cts, rn1, p1x)  #  forward

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
