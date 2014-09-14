import numpy          as _N
import kfardat        as _kfardat
import kfARlib        as _kfar
import LogitWrapper   as lw
import scipy.stats    as _ss

import warnings
warnings.filterwarnings("error")

# cdef extern from "math.h":
#     double exp(double)
#     double sqrt(double)
#     double log(double)
#     double abs(double)

mvnrml    = _N.random.multivariate_normal

#  allow for vector observations
def gibbsSamp(burn, NMC, y, nStates=2, nWins=1, r=40, n=400, model="binomial"):
    #####  MCMC start
    trms = _N.empty(nStates)
    thr  = _N.empty(nStates)

    for it in xrange(1, NMC+burn):
        if (it % 100) == 0:
            print it

        if nWins == 1:
            kw  = kp / ws
        else:
            kw_w1  = kp_w1 / ws_w1
            kw_w2  = kp_w2 / ws_w2

        rnds =_N.random.rand(N+1)

        #  generate latent zs.  Depends on Xs and PG latents
        for n in xrange(N+1):
            thr[:] = 0
            if nWins == 1:
                #  for nStates, there are nStates - 1 thresholds
                for i in xrange(nStates):
                    trms[i] = -0.5*ws[n]*((u[i] + smpx[n]) - kw[n]) * ((u[i] + smpx[n]) - kw[n])
            else:
                for i in xrange(nStates):
                    #  rsd_w1 is 2-component vector (if nStates == 2)
                    trms[i] = -0.5*ws_w1[n] * ((u_w1[i] + smpx[n] - kw_w1[n]) * (u_w1[i] + smpx[n] - kw_w1[n]) - (kw_w1[n]*kw_w1[n])) \
                              -0.5*ws_w2[n] * ((u_w2[i] + smpx[n] - kw_w2[n]) * (u_w2[i] + smpx[n] - kw_w2[n]) - (kw_w2[n]*kw_w2[n]))
                    # trm    = trm_w1 * trm_w2  #  trm is 2-component vector
            for tp in xrange(nStates):
                for bt in xrange(nStates):
                    thr[tp] += (m[bt]/m[tp])*_N.exp(trms[bt] - trms[tp])
            thr = 1 / thr

            z[it, n, :] = states[nStates - 1]   #
            thrC = 0
            s = 0
            while s < nStates - 1:
                thrC += thr[s]
                if rnds[n] < thrC:
                    z[it, n, :] = states[s]
                    break
                s += 1

        if nWins == 1:
            us   = _N.dot(z[it, :, :], u)
            ws = lw.rpg_devroye(rn, smpx + us, num=(N + 1))
        else:
            #  generate PG latents.  Depends on Xs and us, zs.  us1 us2 
            us_w1 = _N.dot(z[it, :, :], u_w1)   #  either low or high u
            us_w2 = _N.dot(z[it, :, :], u_w2)
            ws_w1 = lw.rpg_devroye(rn[0], smpx + us_w1, num=(N + 1))
            ws_w2 = lw.rpg_devroye(rn[1], smpx + us_w2, num=(N + 1))

        _d.copyParams(_N.array([F0]), q2, _N.array([1]), 1)
        #  generate latent AR state
        _d.f_x[0, 0, 0]     = x00
        _d.f_V[0, 0, 0]     = V00
        if nWins == 1:
            _d.y[:]             = kp/ws - us
            _d.Rv[:] =1 / ws   #  time dependent noise
        else:
            btm      = 1 / ws_w1 + 1 / ws_w2   #  size N
            top = (kp_w1/ws_w1 - us_w1) / ws_w2 + (kp_w2/ws_w2 - us_w2) / ws_w1
            _d.y[:] = top/btm
            _d.Rv[:] =1 / (ws_w1 + ws_w2)   #  time dependent noise
        smpx = _kfar.armdl_FFBS_1itr(_d, samples=1)

        #  p3 --  samp u here

        dirArgs = _N.empty(nStates)

        for i in xrange(nStates):
            dirArgs[i] = alp[i] + _N.sum(z[it, :, i])
        m[:] = _N.random.dirichlet(dirArgs)

        # # sample u
        if nWins == 1:
            for st in xrange(nStates):
                A = 0.5*(1/s2_u[st,st] + _N.dot(ws, z[it, :, st]))
                B = u_u[st]/s2_u[st,st] + _N.dot(kp - ws*smpx, z[it, :, st])
                u[st] = B/(2*A) + _N.sqrt(1/(2*A))*_N.random.randn()
        else:
            for st in xrange(nStates):
                #  win1 for this state
                iw = st + 0 * nStates

                A = 0.5*(1/s2_u[iw,iw] + _N.dot(ws_w1, z[it, :, st]))
                B = u_u[iw]/s2_u[iw,iw] + _N.dot(kp_w1-ws_w1*smpx, z[it, :, st])
                u_w1[st] = B/(2*A) + _N.sqrt(1/(2*A))*_N.random.randn()
                iw = st + 1 * nStates

                A = 0.5*(1/s2_u[iw,iw] + _N.dot(ws_w2, z[it, :, st]))
                B = u_u[iw]/s2_u[iw,iw] + _N.dot(kp_w2-ws_w2*smpx, z[it, :, st])
                u_w2[st] = B/(2*A) + _N.sqrt(1/(2*A))*_N.random.randn()

        # sample F0
        F0AA = _N.dot(smpx[0:-1], smpx[0:-1])
        F0BB = _N.dot(smpx[0:-1], smpx[1:])

        F0std= _N.sqrt(q2/F0AA)
        F0a, F0b  = (a_F0 - F0BB/F0AA) / F0std, (b_F0 - F0BB/F0AA) / F0std
        F0=F0BB/F0AA+F0std*_ss.truncnorm.rvs(F0a, F0b)

        #####################    sample q2
        a = a_q2 + 0.5*(N+1)  #  N + 1 - 1
        rsd_stp = smpx[1:] - F0*smpx[0:-1]
        BB = B_q2 + 0.5 * _N.dot(rsd_stp, rsd_stp)
        q2 = _ss.invgamma.rvs(a, scale=BB)
        # #####################    sample x00
        mn  = (u_x00*V00 + s2_x00*x00) / (V00 + s2_x00)
        vr = (V00*s2_x00) / (V00 + s2_x00)
        x00 = mn + _N.sqrt(vr)*_N.random.randn()
        #####################    sample V00
        aa = a_V00 + 0.5
        BB = B_V00 + 0.5*(smpx[0] - x00)*(smpx[0] - x00)
        V00 = _ss.invgamma.rvs(aa, scale=BB)

        smp_F[it]       = F0
        smp_q2[it]      = q2
        if nWins == 1:
            smp_u[it, :] = u
        else:
            smp_u[it, 0, :] = u_w1
            smp_u[it, 1, :] = u_w2
        smp_m[it, :]    = m

        if it >= burn:
            Bsmpx[it-burn, :] = smpx
#        print m

    return Bsmpx, smp_F, smp_q2, smp_u, smp_m, z
