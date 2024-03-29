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

def BN(long iters, long w, long j, double u0, long rn0, cts, xn, double stdu):
    global logfact, ks, maxrn, uTH1, uTH2, iu_sd2, uTH1_pl_uTH2
    #  if accptd too small, increase stdu and try again
    #  if accptd is 0, the sampled params returned for conditional posterior
    #  are not representative of the conditional posterior
    cdef double stdu2= stdu**2
    cdef double istdu2= 1./ stdu2
    cdef long N = len(cts)
    cdef long dist = _cd.__BNML__

    cdef double Mk = _N.mean(cts) if len(cts) > 0 else 0  #  1comp if nWins=1, 2comp
    if Mk == 0:
        return u0, rn0, dist, 0   # no data assigned to this 
    rnmin = _N.array([-1, _N.max(cts)+1, 1], dtype=_N.int)
    cdef long[::1] v_rnmin = rnmin
    cdef long* p_rnmin     = &v_rnmin[0]

    rdns = _N.random.randn(iters)      #  prop u1
    rrmdr = _N.random.rand(iters)      #  which move type
    ran_accpt  = _N.random.rand(iters)

    cdef double[::1] v_rdns = rdns
    cdef double[::1] v_ran_accpt = ran_accpt
    cdef double[::1] v_rrmdr = rrmdr
    cdef double* p_rdns     = &v_rdns[0]
    cdef double* p_rrmdr     = &v_rrmdr[0]
    cdef double* p_ran_accpt     = &v_ran_accpt[0]

    cdef double p0  = 1./(1 + exp(-u0))
    p0x  = 1./(1 + _N.exp(-(u0+xn)))    #  array

    cdef double ll0 = Llklhds(dist, N, cts, rn0, p0x)
    #  the poisson distribution needs to be truncated

    zr2rnmins  = _N.array([None, _N.arange(rnmin[1]), _N.arange(rnmin[2])]) # rnmin is a valid value for n
    lf_zr2rnmins  = _N.array([None, logfact[_N.arange(rnmin[1])], logfact[_N.arange(rnmin[2])]]) # rnmin is a valid value for n

    cdef double u_m     = (uTH1_pl_uTH2)*0.5
    cdef double mag     = 4./(uTH2 - uTH1)

    #  TO DO:  Large cnts usually means large rn.  
    #  for transitions with small ps, the rn we need to transition on the
    #  other side might be very large where the prior prob. is very small
    cdef long accptd  = 0
    
    cdef double ut, utr
    cdef double jx, jxr
    cdef long it
    cdef double u1
    cdef double p1, ip0, ip1
    cdef long rn1
    cdef double irn0
    cdef double lprop0, lprop1, ljac, ll1, lpru0, lpru1
    cdef double e_mu0   # exp(-u0)

    #  initialize these.  Won't need to recalculate until proposal accept
    ut    = (u0 - u_m)*mag

    lpru0   = -0.5*(u0 - mn_u)*(u0 - mn_u)*iu_sd2
    p0  = 1 / (1 + exp(-u0))
    irn0=1./rn0

    smp_us   = _N.empty(iters)
    smp_rns  = _N.empty(iters)
    smp_dtys = _N.empty(iters)
    

    for it in xrange(iters):
        todist = dist

        zr2rnmin = zr2rnmins[dist]
        lf_zr2rnmin = lf_zr2rnmins[dist]

        ##  propose an rn1 
        m2          = irn0 + 0.9      # rn0 is the mean for proposal for rn1
        p_prp_rn1        = 1 - 1./(rn0*m2)  # param p for proposal for rn1
        r_prp_rn1        = rn0 / (rn0*m2-1) # param r for proposal for rn1
        ir_prp_rn1 = int(r_prp_rn1)

        bGood = False   #  rejection sampling of rn1
        while not bGood:
            rn1 = _N.random.negative_binomial(ir_prp_rn1, 1-p_prp_rn1)
            if (rn1 >= p_rnmin[dist]) and (rn1 < maxrn):
                bGood = True  # rn1 < maxrn  - just throw away huge rn

        #print "prop rn1 %d" % rn1

        #########  log proposal density for rn1
        ltrms = logfact[zr2rnmin+ir_prp_rn1-1]  - p_logfact[ir_prp_rn1-1] - lf_zr2rnmin + ir_prp_rn1*log(1-p_prp_rn1) + zr2rnmin*log(p_prp_rn1)
        lCnb1        = log(1 - _N.sum(_N.exp(ltrms)))  #  nrmlzation 4 truncated pmf

        lpmf1       = p_logfact[rn1+ir_prp_rn1-1]  - p_logfact[ir_prp_rn1-1] - p_logfact[rn1] + r_prp_rn1*log(1-p_prp_rn1) + rn1*log(p_prp_rn1) - lCnb1

        #########  log proposal density for rn0
        ##  rn1
        m2          = 1./rn1 + 0.9      # rn0 is the mean for proposal for rn1
        p_prp_rn0        = 1 - 1./(rn1*m2)  # param p for proposal for rn1
        r_prp_rn0        = rn1 / (rn1*m2-1.) # param r for proposal for rn1
        ir_prp_rn0 = int(r_prp_rn0)

        ltrms = logfact[zr2rnmin+ir_prp_rn0-1]  - p_logfact[ir_prp_rn0-1] - lf_zr2rnmin + ir_prp_rn0*log(1-p_prp_rn0) + zr2rnmin*log(p_prp_rn0)
        smelt = _N.sum(_N.exp(ltrms))
        lCnb0        = log(1 - _N.sum(_N.exp(ltrms)))  #  nrmlzation 4 truncated 

        lpmf0       = p_logfact[rn0+ir_prp_rn0-1]  - p_logfact[ir_prp_rn0-1] - p_logfact[rn0] + r_prp_rn0*log(1-p_prp_rn0) + rn0*log(p_prp_rn0) - lCnb0

        ###################################################
        #  propose p1
        #  sample using p from [0, 0.75]  mean 0.25, sqrt(variance) = 0.25
        #  p1 x rn1 = p0 x rn0      --> p1 = p0 x rn0/rn1   BINOMIAL
        #  

        # try:
        #     mn_u1 = -_N.log(rn1 * (1+e_mu0)/rn0 - 1) if dist == _cd.__BNML__ else (u0 - _N.log(rn1*irn0))
        # except Warning:
        #     #  if rn0 >> rn1, (rn0/rn1) >> 1.   p0 x (rn0/rn1) could be > 1.
        #     print "restart  %(it)d   todist   %(to)d   dist %(fr)d   rn0 %(rn0)d >> rn1 %(rn1)d" % {"to" : todist, "fr" : dist, "it" : it, "rn0" : rn0, "rn1" : rn1}
        #     mn_u1  = 0
        if dist == _cd.__BNML__:

            mn_u1 = -_N.log(rn1 * (1+exp(-u0))/rn0 - 1) if dist == _cd.__BNML__ else (u0 - _N.log(rn1*irn0))

            #  if rn0 >> rn1, (rn0/rn1) >> 1.   p0 x (rn0/rn1) could be > 1.
            #print "restart  %(it)d   todist   %(to)d   dist %(fr)d   rn0 %(rn0)d >> rn1 %(rn1)d" % {"to" : todist, "fr" : dist, "it" : it, "rn0" : rn0, "rn1" : rn1}
            #mn_u1  = 0


        u1          = mn_u1 + stdu*p_rdns[it]
        #print "mn_u1  %(m).4e    stdu %(s).4e" % {"m" : mn_u1, "s" : stdu}
        #print "prop u1 %.4e" % u1
        p1x = 1 / (1 + _N.exp(-(u1+xn)))
        p1  = 1/(1 + exp(-u1))

        utr = (u1 - u_m)*mag

        mn_u0 = -log(rn0 * (1+exp(-u1))/rn1 - 1) if dist == _cd.__BNML__ else (u1 - log(float(rn0)/rn1))

        lprop1 = -0.5*(u1 - mn_u1)*(u1-mn_u1)*istdu2 + lpmf1 #  forward
        lprop0 = -0.5*(u0 - mn_u0)*(u0-mn_u0)*istdu2 + lpmf0 #  backwards

        if _cd.__BNML__:
            ratn    = rn1*irn0
            bot     = (ratn*(1+exp(-u0)-1))
            logme   = abs((ratn*exp(-u0)) / bot)

        lpru1   = -0.5*(u1 - mn_u)*(u1 - mn_u)*iu_sd2
        ll1 = Llklhds(todist, N, cts, rn1, p1x)  #  forward

        #lrat = ll1 - ll0 + lpru1 - lpru0 + lprop0 - lprop1
        lrat = ll1 - ll0 + lprop1 - lprop0

        aln   = 1 if (lrat > 0) else exp(lrat)
        #print aln
        if p_ran_accpt[it] < aln:   #  accept
            u0 = u1
            rn0 = rn1
            irn0=1./rn0
            ll0 = ll1
            p0  = p1
            dist=todist

            e_mu0 = exp(-u0)
            ut    = (u0 - u_m)*mag

            lpru0   = -0.5*(u0 - mn_u)*(u0 - mn_u)*iu_sd2

            accptd += 1

        smp_us[it] = u0
        smp_rns[it] = rn0

    #return u0, rn0, dist, accptd
    return smp_us, smp_rns, accptd
