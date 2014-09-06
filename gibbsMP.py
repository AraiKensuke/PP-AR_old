import commdefs as _cd
import numpy as _N
import numpy.polynomial.polynomial as _Npp
from multiprocessing import Pool
import time as _tm
import LogitWrapper as lw
import kfARlibMP as _kfar
import scipy.stats as _ss
from ARcfSmplFuncs import ampAngRep, buildLims, FfromLims, dcmpcff, initF

from ARcfSmpl import ARcfSmpl, FilteredTimeseries

def gibbsSampH(burn, NMC, AR2lims, vF_alfa_rep, R, Cs, Cn, TR, rn, _d, u, B, aS, q2, uts, wts, kp, ws, smpx, Bsmpx, smp_u, smp_q2, allalfas, fs, amps, ranks, priors, ARo, lm2, prior=_cd.__COMP_REF__, aro=_cd.__NF__, ID_q2=True):  ##################################
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

    if u == None:
        psthOffset = _N.empty((TR, _d.N+1))
        Wims         = _N.empty((TR, _d.N+1, _d.N+1))
        Oms          = _N.empty((TR, _d.N+1))
        smWimOm      = _N.zeros(_d.N + 1)
        bConstPSTH = False
    else:
        bConstPSTH  = True
        psthOffset = _N.empty(TR)

    it    = 0

    lrn   = _N.empty((TR, _d.N+1))
    if lm2 == None:
        lrn[:] = 1
    else:
        for tr in xrange(_d.TR):
            lrn[tr] = build_lrnLambda2(_d.N+1, _d.dN[tr], lm2)

    pool = Pool()
    while (it < NMC + burn - 1):
        t1 = _tm.time()
        it += 1
        print it
        if (it % 10) == 0:
            print it
        #  generate latent AR state
        _d.f_x[:, 0, :, 0]     = x00
        if it == 1:
            for m in xrange(TR):
                _d.f_V[m, 0, :, :]     = s2_x00
        else:
            _d.f_V[:, 0, :, :]     = _d.f_V[:, 1, :, :]


        if bConstPSTH:
            psthOffset[:] = u[:]
        else:
            BaS = _N.dot(B.T, aS)
            for m in xrange(_d.TR):
                psthOffset[m] = BaS
        ###  PG latent variable sample

        for m in xrange(_d.TR):
            _N.log(lrn[m] / (1 + (1 - lrn[m])*_N.exp(smpx[m, 2:, 0] + psthOffset[m])), out=ARo[m])

            lw.rpg_devroye(rn, smpx[m, 2:, 0] + psthOffset[m] + ARo[m], out=ws[m, :])
        if TR == 1:
            ws   = ws.reshape(1, _d.N+1)

        #  Now that we have PG variables, construct Gaussian timeseries
        #  ws(it+1)    using u(it), F0(it), smpx(it)
        for m in xrange(TR):
            _d.y[m, :]             = kp[m, :]/ws[m, :] - psthOffset[m] - ARo[m]
        _d.copyParams(F0, q2)
        _d.Rv[:, :] =1 / ws[:, :]   #  time dependent noise

        if bConstPSTH:
            for m in xrange(TR):
                A    = 0.5*(1./s2_u + _N.sum(ws[m]))
                B    = u_u/s2_u + _N.sum(kp[m] - ws[m]*(smpx[m, 2:, 0] + ARo[m]))
                u[m] = B/(2*A) + _N.sqrt(1/(2*A))*_N.random.randn()
                smp_u[m, it] = u[m]
        else:
            smWimOm[:] = 0
            #  cov matrix, prior of aS 
            iD = _N.diag(_N.ones(B.shape[0]))
            for m in xrange(TR):
                Wims[m, :] = _N.diag(ws[m])
                Oms[m, :]  = kp[m] / ws[m] - smpx[m, 2:, 0] - ARo[m]
                smWimOm[:] += _N.dot(Wims[m], Oms[m])
            Bi = _N.sum(Wims[:, :], axis=0)
            A  = _N.dot(_N.linalg.inv(Bi), smWimOm)
            #  now sample 
            iVAR = _N.dot(B, _N.dot(Bi, B.T)) + iD
            VAR  = _N.linalg.inv(iVAR)
            Mn   = _N.dot(VAR, _N.dot(B, _N.dot(Bi, A.T)))# + 0)
            #  multivar_normal returns a row vector
            aS   = _N.random.multivariate_normal(Mn, VAR, size=1)[0, :]
            smp_aS[it, :] = aS

        #  _d.F, _d.N, _d.ks, 
        tpl_args = zip(_d.y, _d.Rv, _d.Fs, q2, _d.Ns, _d.ks, _d.f_x[:, 0, :], _d.f_V[:, 0, :])

        sxv = pool.map(_kfar.armdl_FFBS_1itrMP, tpl_args)

        for m in xrange(TR):
            smpx[m, 2:] = sxv[m][0]
            _d.f_x[m] = sxv[m][1]
            _d.f_V[m] = sxv[m][2]
            smpx[m, 1, 0:k-1]   = smpx[m, 2, 1:]
            smpx[m, 0, 0:k-2]   = smpx[m, 2, 2:]
            Bsmpx[m, it, 2:]    = smpx[m, 2:, 0]
            
        # sample F0
        # for mh in xrange(50):
        #  F0(it+1)    using ws(it+1), u(it+1), smpx(it+1), ws(it+1)
    
        #  wt.shape = (TR, C, _d.N+1+2, 1)
        #  wts.shape = (TR, burn+NMC, C, _d.N+1+2, 1)
        #  ut.shape = (TR, C, _d.N+1+1, 1)
        ARcfSmpl(N+1, k, AR2lims, smpx[:, 1:, 0:k], smpx[:, :, 0:k-1], q2, R, Cs, Cn, alpR, alpC, _d, prior=prior, accepts=10, aro=aro)
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

        print prt
        #  sample u     WE USED TO Do this after smpx
        #  u(it+1)    using ws(it+1), F0(it), smpx(it+1), ws(it+1)

        if ID_q2:
            for m in xrange(TR):
                #####################    sample q2
                a = a_q2 + 0.5*(N+1)  #  N + 1 - 1
                rsd_stp = smpx[m, 3:,0] - _N.dot(smpx[m, 2:-1], F0).T
                BB = B_q2 + 0.5 * _N.dot(rsd_stp, rsd_stp.T)
                q2[m] = _ss.invgamma.rvs(a, scale=BB)
                x00[m]      = smpx[m, 2]*0.1
                smp_q2[m, it]= q2[m]
        else:
            a2 = a_q2 + 0.5*(TR*N + 2)  #  N + 1 - 1
            BB2 = B_q2
            for m in xrange(TR):
                #   set x00 
                x00[m]      = smpx[m, 2]*0.1

                #####################    sample q2
                rsd_stp = smpx[m, 3:,0] - _N.dot(smpx[m, 2:-1], F0).T
                BB2 += 0.5 * _N.dot(rsd_stp, rsd_stp.T)
            q2[:] = _ss.invgamma.rvs(a2, scale=BB2)

        smp_u[:, it] = u
        smp_q2[:, it]= q2
        t2 = _tm.time()
        print "%.5f" % (t2-t1)


    pool.close()
    return _N.array(F_alfa_rep)

def build_lrnLambda2(N, y, lmbda2):
    #  lmbda2 is short snippet of after-spike depression behavior
    lrn = _N.ones(N)
    lh    = len(lmbda2)

    lt   = -int(50*_N.random.rand())
    hst  = []    #  spikes whose history is still felt

    for i in xrange(N):
        L  = len(hst)
        lmbd = 1

        for j in xrange(L - 1, -1, -1):
            lrn[i] *= lmbda2[i - lt - 1]

        if y[i] == 1:
            lt = i

    return lrn
"""

def build_lrnLambda2(N, y, lmbda2):
    #  lmbda2 is short snippet of after-spike depression behavior
    lrn = _N.ones(N)
    lh    = len(lmbda2)

    hst  = []    #  spikes whose history is still felt

    for i in xrange(N):
        L  = len(hst)
        lmbd = 1

        for j in xrange(L - 1, -1, -1):
            th = hst[j]
            #  if i == 10, th == 9, lh == 1
            #  10 - 9 -1 == 0  < 1.   Still efective
            #  11 - 9 -1 == 1         No longer effective
            if i - th - 1 < lh:
                lmbd *= lmbda2[i - th - 1]
            else:
                hst.pop(j)

        if y[i] == 1:
            hst.append(i)

        lrn[i] *= lmbd
    return lrn

"""
