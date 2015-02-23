from kflib import createDataAR
import numpy as _N
import patsy
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

import logerfc as _lfc
import commdefs as _cd

from ARcfSmplFuncs import ampAngRep, buildLims, FfromLims, dcmpcff, initF
from multiprocessing import Pool

import os
import mcmcARp as mARp

class mcmcARpETM(mARp.mcmcARp):
    smp_gam       = None
    bFixgam       = False
    #  Current values of params and state
    s             = 0.01        #  trend.  TRm  = median trial
    GAM           = None;   gam = None
    dfGAM         = None;
    
    # psth spline priors.  diagnal + sub,superdiag matrix
    etme_u          = 0.5;   etme_s2_a         = 1.; etme_s2_b         = 0.1
    etme_is2_a         = None; etme_is2_b         = None

    def initGibbs(self):   ################################ INITGIBBS
        oo   = self

        mARp.mcmcARp.initGibbs(oo)

        if oo.dfGAM is None:
            oo.dfGAM = 9
        oo.GAM = patsy.bs(_N.linspace(0, (oo.t1 - oo.t0)*oo.dt, (oo.t1-oo.t0)), df=oo.dfGAM, include_intercept=True, degree=1)    #  Spline basis for modulation strength

        oo.dfGAM = oo.GAM.shape[1]
        if oo.gam is None:
            oo.gam = _N.ones(oo.dfGAM)
        oo.GAM2= oo.GAM*oo.GAM

        oo.TRm  = 0.5*(oo.TR - 1)
        oo.etme = _N.zeros((oo.N+1, oo.N+1))

        cv = _N.identity(oo.dfGAM) * oo.etme_s2_a
        _N.fill_diagonal(cv[1:, 0:-1], oo.etme_s2_b)
        _N.fill_diagonal(cv[0:-1, 1:], oo.etme_s2_b)
        icv = _N.linalg.inv(cv)
        oo.etme_is2_a = cv[0, 0]
        oo.etme_is2_b = cv[1, 0]

        oo.smp_gam    = _N.zeros((oo.burn + oo.NMC, oo.dfGAM))


    def gibbsSamp(self):  ###########################  GIBBSSAMP
        oo          = self
        ooTR        = oo.TR
        ook         = oo.k
        ooNMC       = oo.NMC
        ooN         = oo.N
        oo.x00         = _N.array(oo.smpx[:, 2])
        oo.V00         = _N.zeros((ooTR, ook, ook))

        ARo   = _N.empty((ooTR, oo._d.N+1))
        
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

        #pool = Pool(processes=oo.processes)
        ###############################  MCMC LOOP  ########################
        ###  need pointer to oo.us, but reshaped for broadcasting to work
        ###############################  MCMC LOOP  ########################
        oous_rs = oo.us.reshape((ooTR, 1))   #  done for broadcasting rules
        lrnBadLoc = _N.empty(oo.N+1, dtype=_N.bool)

        ietme = _N.zeros((oo.N+1, oo.N+1))  #  ietme
        tempetme = _N.ones(oo.N+1)

        while (it < ooNMC + oo.burn - 1):
            _N.fill_diagonal(oo.etme, _N.dot(oo.GAM, oo.gam))
            #_N.fill_diagonal(oo.etme, tempetme)
            _N.fill_diagonal(ietme, 1/_N.diag(oo.etme))
            etmeSMPX = _N.dot(oo.smpx[..., 2:, 0], oo.etme)
            t1 = _tm.time()
            it += 1
            print it
            if (it % 10) == 0:
                print it
            #  generate latent AR state
            oo._d.f_x[:, 0, :, 0]     = oo.x00
            if it == 1:
                for m in xrange(ooTR):
                    oo._d.f_V[m, 0]     = oo.s2_x00
            else:
                oo._d.f_V[:, 0]     = oo._d.f_V[:, 1]

            BaS = _N.dot(oo.B.T, oo.aS)
            ###  PG latent variable sample

            #tPG1 = _tm.time()
            for m in xrange(ooTR):
                _N.log(oo.lrn[m] / (1 + (1 - oo.lrn[m])*_N.exp(etmeSMPX[m] + BaS + oo.us[m])), out=ARo[m])   #  history Offset   ####TRD change
                nani = _N.isnan(ARo[m], out=lrnBadLoc)
                locs = _N.where(lrnBadLoc == True)
                if locs[0].shape[0] > 0:
                    L = locs[0].shape[0]
                    print "ARo locations bad tr %(m)d  %(L) d" % {"m" : m, "L" : L}
                    for l in xrange(L):  #  fill with reasonable value
                        ARo[m, locs[0][l]] = ARo[m, locs[0][l] - 1]

                #lw.rpg_devroye(oo.rn, oo.smpx[m, 2:, 0] + oo.us[m] + BaS + ARo[m], out=oo.ws[m])  ######  devryoe
                lw.rpg_devroye(oo.rn, etmeSMPX[m] + oo.us[m] + BaS + ARo[m], out=oo.ws[m])  ######  devryoe  ####TRD change
                    
            if ooTR == 1:
                oo.ws   = oo.ws.reshape(1, ooN+1)
            _N.divide(oo.kp, oo.ws, out=kpOws)

            #  Now that we have PG variables, construct Gaussian timeseries
            #  ws(it+1)    using u(it), F0(it), smpx(it)

            #oo._d.y = kpOws - BaS - ARo - oous_rs     ####TRD change
            oo._d.y = _N.dot(kpOws - BaS - ARo - oous_rs, ietme)
            #print _N.std(oo._d.y, axis=1)
            oo._d.copyParams(oo.F0, oo.q2)
            #oo._d.Rv[:, :] =1 / oo.ws[:, :]   #  time dependent noise
            #  (MxM)  (MxN) = (MxN)  (Rv is MxN)
            _N.dot(1 / oo.ws, _N.dot(ietme, ietme), out=oo._d.Rv)

            #  cov matrix, prior of aS 

            ########     per trial offset sample
            #Ons  = kpOws - oo.smpx[..., 2:, 0] - ARo - BaS
            Ons  = kpOws - etmeSMPX - ARo - BaS  ####TRD change
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

            ########     PSTH sample
            if oo.bpsth:
                #Oms  = kpOws - oo.smpx[..., 2:, 0] - ARo - oous_rs
                Oms  = kpOws - etmeSMPX - ARo - oous_rs  ####TRD change
                #Oms  = kpOws - oo.smpx[..., 2:, 0] - ARo
                _N.einsum("mn,mn->n", oo.ws, Oms, out=smWimOm)   #  sum over 
                ilv_f  = _N.diag(_N.sum(oo.ws, axis=0))
                #  diag(_N.linalg.inv(Bi)) == diag(1./Bi).  Bii = inv(Bi)
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

            #######  DATA AUGMENTATION.  If we update 's' before, we need to update _d.y right after, _d.y depends on 's'
            #  _d.F, _d.N, _d.ks, 
            tpl_args = zip(oo._d.y, oo._d.Rv, oo._d.Fs, oo.q2, oo._d.Ns, oo._d.ks, oo._d.f_x[:, 0], oo._d.f_V[:, 0])

            for m in xrange(ooTR):
                oo.smpx[m, 2:], oo._d.f_x[m], oo._d.f_V[m] = _kfar.armdl_FFBS_1itrMP(tpl_args[m])
                oo.smpx[m, 1, 0:ook-1]   = oo.smpx[m, 2, 1:]
                oo.smpx[m, 0, 0:ook-2]   = oo.smpx[m, 2, 2:]
                oo.Bsmpx[m, it, 2:]    = oo.smpx[m, 2:, 0]


            ######################################
            etmeSMPX = _N.dot(oo.smpx[..., 2:, 0], oo.etme)
            for m in xrange(ooTR):
                _N.log(oo.lrn[m] / (1 + (1 - oo.lrn[m])*_N.exp(etmeSMPX[m] + BaS + oo.us[m])), out=ARo[m])   #  history Offset   ####TRD change
                nani = _N.isnan(ARo[m], out=lrnBadLoc)
                locs = _N.where(lrnBadLoc == True)
                if locs[0].shape[0] > 0:
                    L = locs[0].shape[0]
                    print "ARo locations bad tr %(m)d  %(L) d" % {"m" : m, "L" : L}
                    for l in xrange(L):  #  fill with reasonable value
                        ARo[m, locs[0][l]] = ARo[m, locs[0][l] - 1]
            ######################################

            ###  TREND coefficient
            ###  ARo calculated using old value smpx x (m-m0).  
            ###  while new value of oo.smpx itself used.  Is this OK?

            if not oo.bFixgam:
                tmpGAM = _N.array(oo.gam)   #  coefficients
                igs = _U.shuffle(range(oo.dfGAM))
                iaa = oo.etme_is2_a
                ibb = oo.etme_is2_b
                ug  = oo.etme_u
                for ig in igs:
                    tmpGAM[:] = oo.gam
                    tmpGAM[ig] = 0
                    #  smpx MxN, GAM[i] is N, x * GAM[i] broadcasted to be MxN
                    #  chi_i is MxN
                    chi_ig = oo.smpx[:, 2:, 0] * oo.GAM.T[ig]
                    A = 0.5*_N.sum(oo.ws*chi_ig*chi_ig)
                    #  Now add the contribution of the prior
                    xEi = oo.smpx[:, 2:, 0] * _N.dot(oo.GAM, tmpGAM)  # broadcasted
                    B = _N.sum(oo.ws * (kpOws - xEi - BaS - oous_rs - ARo) * chi_ig)

                    A += iaa  #  common for all ig
                    if ig == 0:
                        B += 2*(ibb*(tmpGAM[1] - ug) - iaa*ug)
                    elif ig == oo.dfGAM - 1:
                        B += 2*(ibb*(tmpGAM[oo.dfGAM-2] - ug) - iaa*ug)
                    else:
                        B += 2*(ibb*(tmpGAM[ig-1] + tmpGAM[ig+1] - 2*ug) - iaa*ug)

                    A2 = 2*A
                    u  = B/A2
                    sd = _N.sqrt(1/A2)

                    a  = 0
                    b  = 100

                    a = (a - u) / sd
                    b = (b - u) / sd

                    oo.gam[ig] = u + sd*_ss.truncnorm.rvs(a, b)

                oo.smp_gam[it] = oo.gam
                print oo.gam

            if not oo.bFixF:   
                ARcfSmpl(oo.lfc, ooN+1, ook, oo.AR2lims, oo.smpx[:, 1:, 0:ook], oo.smpx[:, :, 0:ook-1], oo.q2, oo.R, oo.Cs, oo.Cn, alpR, alpC, oo._d, prior=oo.use_prior, accepts=30, aro=oo.ARord)  
                oo.F_alfa_rep = alpR + alpC   #  new constructed
                prt, rank, f, amp = ampAngRep(oo.F_alfa_rep, f_order=True)
                print prt
            ut, wt = FilteredTimeseries(ooN+1, ook, oo.smpx[:, 1:, 0:ook], oo.smpx[:, :, 0:ook-1], oo.q2, oo.R, oo.Cs, oo.Cn, alpR.tolist(), alpC.tolist(), oo._d)
            #ranks[it]    = rank
            oo.allalfas[it] = oo.F_alfa_rep

            for m in xrange(ooTR):
                oo.wts[m, it, :, :]   = wt[m, :, :, 0]
                oo.uts[m, it, :, :]   = ut[m, :, :, 0]
                if not oo.bFixF:
                    oo.amps[it, :]  = amp
                    oo.fs[it, :]    = f

            oo.F0          = (-1*_Npp.polyfromroots(oo.F_alfa_rep)[::-1].real)[1:]

            #  sample u     WE USED TO Do this after smpx
            #  u(it+1)    using ws(it+1), F0(it), smpx(it+1), ws(it+1)

            """
            if oo.ID_q2:   ####  mod. strength trends don't changes this part
                for m in xrange(ooTR):
                    #####################    sample q2
                    a = oo.a_q2 + 0.5*(ooN+1)  #  N + 1 - 1
                    rsd_stp = oo.smpx[m, 3:,0] - _N.dot(oo.smpx[m, 2:-1], oo.F0).T
                    BB = oo.B_q2 + 0.5 * _N.dot(rsd_stp, rsd_stp.T)
                    oo.q2[m] = _ss.invgamma.rvs(a, scale=BB)
                    oo.x00[m]      = oo.smpx[m, 2]*0.1
                    oo.smp_q2[m, it]= oo.q2[m]
            else:
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
            """

            ###  update modulation strength trend parameter

            t2 = _tm.time()
            print "gibbs iter %.3f" % (t2-t1)

        #pool.close()
