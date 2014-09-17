def fig1(mxv):
    bNoPCS  = False
    try:
        pcs = []
        for mt in xrange(0, mxv.burn + mxv.NMC, 10):
            pc, pv = _ss.pearsonr(mxv.z[mt, :, 0], mxv.zT)
            pcs.append(pc)
    except RuntimeWarning:
        bNoPCS    = True
        print "no pcs"

    fig = _plt.figure(figsize=(5*3, 4*2))
    _plt.subplot2grid((2, 3), (0, 0))
    if not bNoPCS:
        _plt.plot(pcs)
    _plt.subplot2grid((2, 3), (0, 1))
    _plt.plot(mxv.xT)
    _plt.plot(_N.mean(mxv.Bsmpx[mxv.NMC-2000:mxv.NMC], axis=0))
    _plt.subplot2grid((2, 3), (0, 2))
    _plt.plot(mxv.smp_F)
    _plt.subplot2grid((2, 3), (1, 0), colspan=3)
    _plt.plot(_N.mean(mxv.z[mxv.burn+mxv.NMC-1000:mxv.burn+mxv.NMC, :], axis=0)[:, 0], ls="", ms=6, marker=".", color="black")
    _plt.ylim(-0.1, 1.1)
    _plt.savefig("mcmc_run")
