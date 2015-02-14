from kflib import createDataAR
#import matplotlib.pyplot as _plt
import numpy as _N
import re as _re
from filter import bpFilt, lpFilt, gauKer

import scipy.stats as _ss
from kassdirs import resFN, datFN

from   mcmcARpPlot import plotFigs, plotARcomps, plotQ2
from mcmcARpFuncs import loadL2, runNotes
import kfardat as _kfardat

import utilities as _U

import numpy.polynomial.polynomial as _Npp
import time as _tm
import ARlib as _arl
import kfARlibMPmv as _kfar
import LogitWrapper as lw
from ARcfSmpl import ARcfSmpl, FilteredTimeseries

import commdefs as _cd

from ARcfSmplFuncs import ampAngRep, buildLims, FfromLims, dcmpcff, initF
from multiprocessing import Pool

import os

import mcmcARp as mARp

class mcmcARpBM(mARp.mcmcARp):
    ###########  TEMP
    ss0           = None; ss1 = None;
    THR           = None
    startZ        = 0

    #  binomial states
    nStates       = 2
    s             = None   #  coupling M x 2
    z             = None   #  state index M x 2  [(1, 0), (0, 1), (0, 1), ...]
    m             = None   #  dim 2
    sd            = None
    
    #  Dirichlet priors
    alp          = None

    def initGibbs(self):   ################################ INITGIBBS
        oo   = self

        mARp.mcmcARp.initGibbs(oo)

        oo.smp_zs    = _N.zeros((oo.TR, oo.burn + oo.NMC, oo.nStates))
        oo.smp_ms    = _N.zeros((oo.burn + oo.NMC, oo.nStates))
        oo.Z         = _N.zeros((oo.TR, oo.nStates), dtype=_N.int)
        oo.s         = _N.array([0.01, 1])
        oo.sd        = _N.zeros((oo.TR, oo.TR))
        oo.m         = _N.array([0.5, 0.5])
        oo.alp       = _N.array([1, 1])

    def gibbsSamp(self):  ###########################  GIBBSSAMP
        oo          = self
        ooTR        = oo.TR
        ook         = oo.k
        ooNMC       = oo.NMC
        ooN         = oo.N
        oo.x00         = _N.array(oo.smpx[:, 2])
        oo.V00         = _N.zeros((ooTR, ook, ook))

        ARo     = _N.empty((ooTR, oo._d.N+1))
        ARo01   = _N.empty((oo.nStates, ooTR, oo._d.N+1))
        
        kpOws = _N.empty((ooTR, ooN+1))
        lv_f     = _N.zeros((ooN+1, ooN+1))
        lv_u     = _N.zeros((ooTR, ooTR))
        alpR   = oo.F_alfa_rep[0:oo.R]
        alpC   = oo.F_alfa_rep[oo.R:]
        Bii    = _N.zeros((ooN+1, ooN+1))
        
        #alpC.reverse()
        #  F_alfa_rep = alpR + alpC  already in right order, no?

        Wims         = _N.empty((ooTR, ooN+1, ooN+1))
        Oms          = _N.empty((ooTR, ooN+1))
        smWimOm      = _N.zeros(ooN + 1)
        smWinOn      = _N.zeros(ooTR)
        bConstPSTH = False
        D_f          = _N.diag(_N.ones(oo.B.shape[0])*oo.s2_a)   #  spline
        iD_f = _N.linalg.inv(D_f)
        D_u  = _N.diag(_N.ones(oo.TR)*oo.s2_u)   #  This should 
        iD_u = _N.linalg.inv(D_u)
        iD_u_u_u = _N.dot(iD_u, _N.ones(oo.TR)*oo.u_u)
        BDB  = _N.dot(oo.B.T, _N.dot(D_f, oo.B))
        DB   = _N.dot(D_f, oo.B)
        BTua = _N.dot(oo.B.T, oo.u_a)

        it    = 0

        oo.lrn   = _N.empty((ooTR, ooN+1))
        if oo.l2 is None:
            oo.lrn[:] = 1
        else:
            for tr in xrange(ooTR):
                oo.lrn[tr] = oo.build_lrnLambda2(tr)

        ###############################  MCMC LOOP  ########################
        ###  need pointer to oo.us, but reshaped for broadcasting to work
        ###############################  MCMC LOOP  ########################
        oous_rs = oo.us.reshape((ooTR, 1))   #  done for broadcasting rules
        lrnBadLoc = _N.empty(oo.N+1, dtype=_N.bool)

        lows = [3, ]
        for tr in xrange(oo.TR):
            oo.Z[tr, 0] = 0;                oo.Z[tr, 1] = 1
            try:
                lows.index(tr)
                oo.Z[tr, 0] = 1;                oo.Z[tr, 1] = 0
            except ValueError:
                pass

        sd01   = _N.zeros((oo.nStates, oo.TR, oo.TR))
        _N.fill_diagonal(sd01[0], oo.s[0])
        _N.fill_diagonal(sd01[1], oo.s[1])
        isd01  = _N.zeros((oo.TR, oo.TR))

        smpx01 = _N.zeros((oo.nStates, oo.TR, oo.N+1))
        ARo01  = _N.empty((oo.nStates, oo.TR, oo.N+1))
        zsmpx  = _N.empty((oo.TR, oo.N+1))

        #  zsmpx created
        #  PG

        zd     = _N.zeros((oo.TR, oo.TR))
        izd    = _N.zeros((oo.TR, oo.TR))
        ll    = _N.empty(oo.nStates)
        Bp    = _N.empty((oo.nStates, oo.N+1))

        for m in xrange(ooTR):
            oo._d.f_V[m, 0]     = oo.s2_x00
            oo._d.f_V[m, 1]     = oo.s2_x00


        THR = _N.empty(oo.TR)
        dirArgs = _N.empty(oo.nStates)  #  dirichlet distribution args
        while (it < ooNMC + oo.burn - 1):
            t1 = _tm.time()
            it += 1
            print it
            BaS = _N.dot(oo.B.T, oo.aS)

            if it > oo.startZ:
                ######  Z
                for tryZ in xrange(oo.nStates):
                    _N.dot(sd01[tryZ], oo.smpx[..., 2:, 0], out=smpx01[tryZ])
                    oo.build_addHistory(ARo01[tryZ], smpx01[tryZ, m], BaS, oo.us, lrnBadLoc)

                for m in xrange(oo.TR):
                    for tryZ in xrange(oo.nStates):

                        #  calculate p0, p1  p0 = m_0 x PROD_n Ber(y_n | Z_j)
                        #                       = m_0 x _N.exp(_N.log(  ))
                        #  p0, p1 not normalized
                        ll[tryZ] = 0
                        #  Ber(0 | ) and Ber(1 | )
                        expT = _N.exp(smpx01[tryZ, m] + BaS + ARo01[tryZ, m] + oo.us[m])
                        Bp[0]   = 1 / (1 + expT)
                        Bp[1]   = expT / (1 + expT)
                        #   z[:, 1]   is state label

                        for n in xrange(oo.N+1):
                            ll[tryZ] += _N.log(Bp[oo.y[m, n], n])

                    ofs = _N.min(ll)
                    nc = oo.m[0]*_N.exp(ll[0] - ofs) + oo.m[1]*_N.exp(ll[1] - ofs)

                    oo.Z[m, 0] = 0;  oo.Z[m, 1] = 1
                    THR[m] = (oo.m[0]*_N.exp(ll[0] - ofs) / nc)
                    if _N.random.rand() < THR[m]:
                        oo.Z[m, 0] = 1;  oo.Z[m, 1] = 0
                    oo.smp_zs[m, it] = oo.Z[m]

                print THR
                #  Z  set
                _N.fill_diagonal(zd, oo.s[oo.Z[:, 1]])
                _N.fill_diagonal(izd, 1./oo.s[oo.Z[:, 1]])            
                _N.dot(zd, oo.smpx[..., 2:, 0], out=zsmpx)
                ######  sample m's
                _N.add(oo.alp, _N.sum(oo.Z, axis=0), out=dirArgs)
                oo.m[:] = _N.random.dirichlet(dirArgs)
                oo.smp_ms[it] = oo.m
            else:
                _N.fill_diagonal(zd, 1.)
                _N.fill_diagonal(izd, 1.)
                _N.dot(zd, oo.smpx[..., 2:, 0], out=zsmpx)
            oo.build_addHistory(ARo, zsmpx, BaS, oo.us, lrnBadLoc)

            ######  PG generate
            for m in xrange(ooTR):
                lw.rpg_devroye(oo.rn, zsmpx[m] + oo.us[m] + BaS + ARo[m], out=oo.ws[m])  ######  devryoe  ####TRD change
            _N.divide(oo.kp, oo.ws, out=kpOws)

            ########     per trial offset sample
            Ons  = kpOws - zsmpx - ARo - BaS
            _N.einsum("mn,mn->m", oo.ws, Ons, out=smWinOn)  #  sum over trials
            ilv_u  = _N.diag(_N.sum(oo.ws, axis=1))  #  var  of LL
            #  diag(_N.linalg.inv(Bi)) == diag(1./Bi).  Bii = inv(Bi)
            _N.fill_diagonal(lv_u, 1./_N.diagonal(ilv_u))
            lm_u  = _N.dot(lv_u, smWinOn)  #  nondiag of 1./Bi are inf, mean LL
            #  now sample
            iVAR = ilv_u + iD_u
            VAR  = _N.linalg.inv(iVAR)  #
            Mn    = _N.dot(VAR, _N.dot(ilv_u, lm_u) + iD_u_u_u)
            oo.us[:]  = _N.random.multivariate_normal(Mn, VAR, size=1)[0, :]
            oo.smp_u[:, it] = oo.us

            ####  Sample latent state
            oo._d.y = _N.dot(izd, kpOws - BaS - ARo - oous_rs)
            oo._d.copyParams(oo.F0, oo.q2)
            #  (MxM)  (MxN) = (MxN)  (Rv is MxN)
            _N.dot(_N.dot(izd, izd), 1. / oo.ws, out=oo._d.Rv)
            oo._d.f_x[:, 0, :, 0]     = oo.x00
            oo._d.f_V[:, 0]     =       oo._d.f_V[:, 1]
            #  _d.F, _d.N, _d.ks, 
            tpl_args = zip(oo._d.y, oo._d.Rv, oo._d.Fs, oo.q2, oo._d.Ns, oo._d.ks, oo._d.f_x[:, 0], oo._d.f_V[:, 0])

            for m in xrange(ooTR):
                oo.smpx[m, 2:], oo._d.f_x[m], oo._d.f_V[m] = _kfar.armdl_FFBS_1itrMP(tpl_args[m])
                oo.smpx[m, 1, 0:ook-1]   = oo.smpx[m, 2, 1:]
                oo.smpx[m, 0, 0:ook-2]   = oo.smpx[m, 2, 2:]
                oo.Bsmpx[m, it, 2:]    = oo.smpx[m, 2:, 0]

            #######  Sample AR coefficient
            if not oo.bFixF:   
                ARcfSmpl(oo.lfc, ooN+1, ook, oo.AR2lims, oo.smpx[:, 1:, 0:ook], oo.smpx[:, :, 0:ook-1], oo.q2, oo.R, oo.Cs, oo.Cn, alpR, alpC, oo._d, prior=oo.use_prior, accepts=30, aro=oo.ARord)

                oo.F_alfa_rep = alpR + alpC   #  new constructed
                prt, rank, f, amp = ampAngRep(oo.F_alfa_rep, f_order=True)
                print prt
            ut, wt = FilteredTimeseries(ooN+1, ook, oo.smpx[:, 1:, 0:ook], oo.smpx[:, :, 0:ook-1], oo.q2, oo.R, oo.Cs, oo.Cn, alpR, alpC, oo._d)
            oo.allalfas[it] = oo.F_alfa_rep

            for m in xrange(ooTR):
                oo.wts[m, it, :, :]   = wt[m, :, :, 0]
                oo.uts[m, it, :, :]   = ut[m, :, :, 0]
                if not oo.bFixF:
                    oo.amps[it, :]  = amp
                    oo.fs[it, :]    = f
            oo.F0          = (-1*_Npp.polyfromroots(oo.F_alfa_rep)[::-1].real)[1:]

            oo.a2 = oo.a_q2 + 0.5*(ooTR*ooN + 2)  #  N + 1 - 1
            BB2 = oo.B_q2
            for m in xrange(ooTR):
                #   set x00 
                oo.x00[m]      = oo.smpx[m, 2]*0.1

                #####################    sample q2
                rsd_stp = oo.smpx[m, 3:,0] - _N.dot(oo.smpx[m, 2:-1], oo.F0).T
                BB2 += 0.5 * _N.dot(rsd_stp, rsd_stp.T)
            oo.q2[:] = _ss.invgamma.rvs(oo.a2, scale=BB2)

            oo.smp_q2[:, it]= oo.q2

            ########     PSTH sample  Do PSTH after we generate zs
            if oo.bpsth:
                Oms  = kpOws - zsmpx - ARo - oous_rs
                _N.einsum("mn,mn->n", oo.ws, Oms, out=smWimOm)   #  sum over 
                ilv_f  = _N.diag(_N.sum(oo.ws, axis=0))
                _N.fill_diagonal(lv_f, 1./_N.diagonal(ilv_f))
                lm_f  = _N.dot(lv_f, smWimOm)  #  nondiag of 1./Bi are inf
                #  now sample
                iVAR = _N.dot(oo.B, _N.dot(ilv_f, oo.B.T)) + iD_f
                VAR  = _N.linalg.inv(iVAR)  #  knots x knots
                iBDBW = _N.linalg.inv(BDB + lv_f)   # BDB not diag
                Mn    = oo.u_a + _N.dot(DB, _N.dot(iBDBW, lm_f - BTua))
                oo.aS   = _N.random.multivariate_normal(Mn, VAR, size=1)[0, :]
                oo.smp_aS[it, :] = oo.aS
            else:
                oo.aS[:]   = 0

        t2 = _tm.time()
        print "gibbs iter %.3f" % (t2-t1)

