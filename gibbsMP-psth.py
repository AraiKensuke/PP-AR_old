def gibbsSampH(burn, NMC, AR2lims, vF_alfa_rep, R, Cs, Cn, TR, rn, _d, B, aS, q2, uts, wts, kp, ws, smpx, Bsmpx, smp_aS, smp_q2, allalfas, fs, amps, ranks, priors, ARo, lm2, bMW=True, prior=_cd.__COMP_REF__, aro=_cd.__NF__, ID_q2=True, ARfixed=False):  ##################################
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

    Wims         = _N.empty((TR, _d.N+1, _d.N+1))
    Oms          = _N.empty((TR, _d.N+1))
    smWimOm      = _N.zeros(_d.N + 1)
        
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
        if (it % 10) == 0:
            print it
        #  generate latent AR state
        _d.f_x[:, 0, :, 0]     = x00
        if it == 1:
            for m in xrange(TR):
                _d.f_V[m, 0, :, :]     = s2_x00
        else:
            _d.f_V[:, 0, :, :]     = _d.f_V[:, 1, :, :]

        # print "--1"
        # print aS.shape
        # print _N.dot(B.T, aS).shape
        # print "--2"
        ###  PG latent variable sample
        for m in xrange(_d.TR):
            _N.log(lrn[m] / (1 + (1 - lrn[m])*_N.exp(smpx[m, 2:, 0] + _N.dot(B.T, aS))), out=ARo[m])

            lw.rpg_devroye(rn, smpx[m, 2:, 0] + _N.dot(B.T, aS) + ARo[m], num=(N + 1), out=ws[m, :])
        if TR == 1:
            ws   = ws.reshape(1, _d.N+1)

        #  Now that we have PG variables, construct Gaussian timeseries
        #  ws(it+1)    using u(it), F0(it), smpx(it)
        for m in xrange(TR):
            _d.y[m, :]             = kp[m, :]/ws[m, :] - _N.dot(B.T, aS) - ARo[m]
        _d.copyParams(F0, q2)
        _d.Rv[:, :] =1 / ws[:, :]   #  time dependent noise

        ###  resample offset
        ###  We just call it A, B

        smWimOm[:] = 0
        #  cov matrix, prior of aS 
        iD = _N.diag(_N.ones(B.shape[0]))
        for m in xrange(TR):
            Wims[m, :] = _N.diag(ws[m])
            # print "***"
            # print kp[m].shape
            # print ws[m].shape
            # print smpx[m, 2:, 0].shape            
            # print ARo[m].shape
            Oms[m, :]  = kp[m] / ws[m] - smpx[m, 2:, 0] - ARo[m]
            smWimOm[:] += _N.dot(Wims[m], Oms[m])
        Bi = _N.sum(Wims[:, :], axis=0)
        A  = _N.dot(_N.linalg.inv(Bi), smWimOm)
        #  now sample 
        iVAR = _N.dot(B, _N.dot(Bi, B.T)) + iD
        VAR  = _N.linalg.inv(iVAR)
        Mn   = _N.dot(VAR, _N.dot(B, _N.dot(Bi, A.T)))# + 0)
        # print "----^^^^^"
        # print VAR.shape
        # print Mn.shape
        #  multivar_normal returns a row vector
        aS   = _N.random.multivariate_normal(Mn, VAR, size=1)[0, :]
        # print "assss"
        # print aS.shape
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

        if not ARfixed:
            if bMW:
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
            else:   #  Simple
                F0 = ARcfSimple(N+1, k, smpx[2:], q2)

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
            smp_q2[:, it]= q2

        t2 = _tm.time()
        print "%.5f" % (t2-t1)


    pool.close()
    return _N.array(F_alfa_rep)
