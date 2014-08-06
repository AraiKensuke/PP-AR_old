import numpy as _N
import kfARlib as _kfar
from ARcfSmpl import ARcfSmpl, FilteredTimeseries
import commdefs as _cd
import LogitWrapper as lw
import numpy.polynomial.polynomial as _Npp
from ARcfSmplFuncs import ampAngRep, buildLims, FfromLims, dcmpcff, initF
import scipy.stats as _ss

def build_lrn(N, y, absRf):
    lrn = _N.ones(N)
    for n in xrange(N):
        if y[n] == 1:
            untl = (n + absRf + 1) if (n + absRf + 1 < N) else N   # untl
            for m in xrange(n + 1, untl):
                lrn[m] = 0.0001

    return lrn

def gibbsSamp(burn, NMC, AR2lims, vF_alfa_rep, R, Cs, Cn, TR, rn, _d, u, q2, uts, wts, kp, ws, smpx, Bsmpx, smp_u, smp_q2, allalfas, fs, amps, ranks, priors, ARo, bMW=True, prior=_cd.__COMP_REF__, absRef=0):  ##################################
    x00         = _N.array(smpx[:, 2])
    V00         = _N.zeros((TR, _d.k, _d.k))

    u_u         = priors["u_u"]
    s2_u        = priors["s2_u"]
    a_q2        = priors["a_q2"]
    B_q2        = priors["B_q2"]
    u_x00       = priors["u_x00"]
    s2_x00      = priors["s2_x00"]
    F_alfa_rep = vF_alfa_rep.tolist()
    alpR   = F_alfa_rep[0:R]
    alpC   = F_alfa_rep[R:]
    #alpC.reverse()
    F_alfa_rep = alpR + alpC
    C          = Cs + Cn
    N            = _d.N
    k            = _d.k
    F0          = (-1*_Npp.polyfromroots(F_alfa_rep)[::-1].real)[1:]
        
    it    = 0

#    ARo   = _N.empty((TR, _d.N+1))
    lrn   = _N.empty((TR, _d.N+1))
    for tr in xrange(_d.TR):
        lrn[tr] = build_lrn(_d.N+1, _d.y[tr], absRef)

    while (it < NMC + burn - 1):
        it += 1
        if (it % 10) == 0:
            print it
        #  generate latent AR state
        _d.f_x[:, 0, :, 0]     = x00
        if it == 1:
            for m in xrange(TR):
                _d.f_V[m, 0, :, :]     = s2_x00
        else:
            _d.f_V[:, 0, :, :]     = _d.f_V[:, 1, :, :]

        #  Before doing devroye, we need to also sample, we need to set
        #  offset due to absolute refractory period
        
        #if not bGaussian:
        for m in xrange(_d.TR):
            _N.log(lrn[m] / (1 + (1 - lrn[m])*_N.exp(smpx[m, 2:, 0] + u[m])), out=ARo[m])
            lw.rpg_devroye(rn, smpx[m, 2:, 0] + u[m] + ARo[m], num=(N + 1), out=ws[m, :])
        if TR == 1:
            ws   = ws.reshape(1, _d.N+1)
        
        #  ws(it+1)    using u(it), F0(it), smpx(it)
        for m in xrange(TR):
            _d.y[m, :]             = kp[m, :]/ws[m, :] - u[m] - ARo[m]
        _d.copyParams(F0, q2)
        _d.Rv[:, :] =1 / ws[:, :]   #  time dependent noise

        for m in xrange(TR):
            A    = 0.5*(s2_u + _N.sum(ws[m]))
            B    = u_u/s2_u + _N.sum(kp[m] - ws[m]*(smpx[m, 2:, 0] + ARo[m]))
            u[m] = B/(2*A) + _N.sqrt(1/(2*A))*_N.random.randn()

        for m in xrange(TR):
            smpx[m, 2:] = _kfar.armdl_FFBS_1itr(_d, m=m, ffast=True, fast=True)
            smpx[m, 1, 0:k-1]   = smpx[m, 2, 1:]
            smpx[m, 0, 0:k-2]   = smpx[m, 2, 2:]
            Bsmpx[m, it, 2:]    = smpx[m, 2:, 0]

        # sample F0
        # for mh in xrange(50):
        #  F0(it+1)    using ws(it+1), u(it+1), smpx(it+1), ws(it+1)
    
        if bMW:
            #  wt.shape = (TR, C, _d.N+1+2, 1)
            #  wts.shape = (TR, burn+NMC, C, _d.N+1+2, 1)
            #  ut.shape = (TR, C, _d.N+1+1, 1)
            ARcfSmpl(N+1, k, AR2lims, smpx[:, 1:, 0:k], smpx[:, :, 0:k-1], q2, R, Cs, Cn, alpR, alpC, _d, prior=prior, accepts=10)
            F_alfa_rep = alpR + alpC   #  new constructed
            prt, rank, f, amp = ampAngRep(F_alfa_rep, f_order=True)
            ut, wt = FilteredTimeseries(N+1, k, smpx[:, 1:, 0:k], smpx[:, :, 0:k-1], q2, R, Cs, Cn, alpR, alpC, _d)
            ranks[it]    = rank
            allalfas[it] = F_alfa_rep

            for m in xrange(TR):
                wts[m, it, :, :]   = wt[m, :, :, 0]
                uts[m, it, :, :]   = ut[m, :, :, 0]
                amps[it, :]  = amp
                fs[it, :]    = f

            F0          = (-1*_Npp.polyfromroots(F_alfa_rep)[::-1].real)[1:]
        else:   #  Simple
            F0 = ARcfSimple(N+1, k, smpx[2:], q2)

        print prt
        #  sample u     WE USED TO Do this after smpx
        #  u(it+1)    using ws(it+1), F0(it), smpx(it+1), ws(it+1)

        for m in xrange(TR):
            #####################    sample q2
            a = a_q2 + 0.5*N  #  N + 1 - 1
            rsd_stp = smpx[m, 3:,0] - _N.dot(smpx[m, 2:-1], F0).T
            BB = B_q2 + 0.5 * _N.dot(rsd_stp, rsd_stp.T)
            q2[m] = _ss.invgamma.rvs(a, scale=BB)
            x00[m]      = smpx[m, 2]*0.1

            smp_u[m, it] = u[m]
            smp_q2[m, it]= q2[m]
    return _N.array(F_alfa_rep)
