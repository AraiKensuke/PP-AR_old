#  compare glm CIF vs. AR latent state.  Does the CIF represent the oscillation
#  well?

#  save oscMn from smplLatent
#  save est.params, X, from glm, LHbin, 
import filter as _flt
import numpy as _N
import scipy.stats as _ss
import matplotlib.pyplot as _plt
import mcmcFigs as mF
import myColors as myC

def compare(mARp, est, X, spkHist, oscMn, dat, gkW=20, earlyHist=None):
    params = _N.array(est.params)

    stT = spkHist.LHbin * (spkHist.nLHBins + 1)    #  first stT spikes used for initial history
    ocifs  = _N.empty((spkHist.endTR - spkHist.startTR, spkHist.t1-spkHist.t0 - stT))
    dt     = 0.001

    params[spkHist.endTR:spkHist.endTR+spkHist.LHbin] = params[spkHist.endTR+spkHist.LHbin]
    print params[spkHist.endTR:spkHist.endTR+spkHist.LHbin]
    fig = _plt.figure()
    _plt.plot(params[spkHist.endTR+spkHist.LHbin:])

    for tr in xrange(spkHist.endTR - spkHist.startTR):
        ocifs[tr] = _N.exp(_N.dot(X[tr], params)) / dt

    gk = _flt.gauKer(gkW)
    gk /= _N.sum(gk)

    TR  = oscMn.shape[0]
    corrs = _N.empty((TR, 3))

    cglmAll = _N.zeros((TR, mARp.N+1))
    infrdAll = _N.zeros((TR, mARp.N+1))
    xt   = _N.arange(stT, mARp.N+1)

    for tr in xrange(spkHist.startTR, TR):
        gt = dat[stT:, tr*3]
        gt /= _N.std(gt)

        glm = (ocifs[tr] - _N.mean(ocifs[tr])) / _N.std(ocifs[tr])
        cglm = _N.convolve(glm, gk, mode="same")
        cglm /= _N.std(cglm)

        infrd = oscMn[tr, stT:] / _N.std(oscMn[tr, stT:])
        infrd /= _N.std(infrd)

        # if tr < 3:
        #     fig = _plt.figure()
        #     _plt.plot(glm + 4)
        #     _plt.plot(cglm + 2)
        #     _plt.plot(infrd)
        #     _plt.plot(gt - 2)
        #     _plt.suptitle(tr)


        pc1, pv1 = _ss.pearsonr(glm, gt)
        pc1c, pv1c = _ss.pearsonr(cglm, gt)
        pc2, pv2 = _ss.pearsonr(infrd, gt)

        cglmAll[tr, stT:] = cglm
        infrdAll[tr, stT:] = infrd
        

        # fig = _plt.figure(figsize=(12, 4))
        # ax = fig.add_subplot(1, 1, 1)
        # _plt.plot(xt, infrd, color=myC.infrdM, lw=1.5)
        # _plt.plot(xt, cglm, color=myC.infrdM, lw=2., ls="--")
        # #_plt.plot(xt, glm, color=myC.infrdM, lw=2., ls="-.")
        # _plt.plot(xt, gt, color=myC.grndTruth, lw=3)
        # #_plt.title("%(1).3f   %(2).3f" % {"1" : pc1c, "2" : pc2})
        # _plt.xlim(stT, mARp.N+1)
        # mF.arbitraryAxes(ax, axesVis=[False, False, False, False], xtpos="bottom", ytpos="none")
        # mF.setLabelTicks(_plt, yticks=[], yticksDsp=None, xlabel="time (ms)", ylabel=None, xtickFntSz=24, xlabFntSz=26)
        # fig.subplots_adjust(left=0.05, right=0.95, bottom=0.2, top=0.85)
        # _plt.savefig("cmpGLMAR%d.eps" % tr, transparent=True)
        # _plt.close()


        corrs[tr] = pc1, pc1c, pc2
    mF.histPhase0_phaseInfrd(mARp, cglmAll, t0=stT, t1=(mARp.N+1), bRealDat=False, normed=True, maxY=1.8, fn="smthdGLMPhaseGLM")
    mF.histPhase0_phaseInfrd(mARp, infrdAll, t0=stT, t1=(mARp.N+1), bRealDat=False, normed=True, maxY=1.8, fn="smthdGLMPhaseInfrd")

    print _N.mean(corrs[:, 0])
    print _N.mean(corrs[:, 1])
    print _N.mean(corrs[:, 2])

    fig = _plt.figure(figsize=(8, 3.5))
    ax  = fig.add_subplot(1, 2, 1)
    _plt.hist(corrs[:, 1], bins=_N.linspace(-0.5, max(corrs[:, 2])*1.05, 30), color=myC.hist1)
    mF.bottomLeftAxes(ax)
    ax  = fig.add_subplot(1, 2, 2)
    _plt.hist(corrs[:, 2], bins=_N.linspace(-0.5, max(corrs[:, 2])*1.05, 30), color=myC.hist1)
    mF.bottomLeftAxes(ax)
    fig.subplots_adjust(left=0.05, bottom=0.1, right=0.95, top=0.88, wspace=0.2, hspace=0.2)

    _plt.savefig("cmpGLMAR_hist")
    _plt.close()
