import matplotlib.pyplot as _plt
import myColors as mC
import numpy as _N
import commdefs as _cd

def lowhis(mARp, bfn="low_high"):
    zTr, zFt, pctCrct = mARp.getZs()
    lst = 0   # low state
    occ = _N.mean(mARp.smp_zs[mARp.burn:, :, lst], axis=0)
    ms  = _N.mean(mARp.smp_m[mARp.burn:], axis=0)
    li  = _N.where(occ > ms[lst])

    zFt = _N.zeros(mARp.N+1, dtype=_N.int)
    zFt[li] = 1

    zTr = _N.zeros(mARp.N+1, dtype=_N.int)
    li  = _N.where(mARp.st == lst)[0]
    zTr[li] = 1


    fig, ax = _plt.subplots(figsize=(13, 4.5))
    _plt.plot(zFt, lw=4, color=mC.grndTruth, marker="o", ms=5)
    _plt.plot(zTr, lw=2.5, color=mC.infrdM, marker="o", ms=5)
    _plt.ylim(-0.3, 1.3)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")
    _plt.yticks([0, 1], ["Low", "High"], fontsize=26)
    tls = _plt.gca().get_yticklabels()
    for tl in tls:
        if tl.get_text() == "Low":
            tl.set_color("green")
        elif tl.get_text() == "High":
            tl.set_color("black")

    _plt.ylabel("Firing state", fontsize=28)
    _plt.xticks(range(0, mARp.N + 2, 20), fontsize=26)
    _plt.xlabel("trials", fontsize=28)
    _plt.xlabel("trial #")
    fig.subplots_adjust(bottom=0.2, left=0.15, right=0.9, top=0.86)
    _plt.xlim(-1, mARp.N+2)


    missed = _N.where(zFt != zTr)
    gotit  = _N.where(zFt == zTr)
    _plt.plot(missed[0], _N.ones(len(missed[0]))*-0.2, ls="", ms=6, marker="v", color="black", markerfacecolor="black", markeredgecolor="black")
    _plt.plot(gotit[0], _N.ones(len(gotit[0]))*1.2, ls="", ms=6, marker="^", color="black", markerfacecolor="black", markeredgecolor="black")

    _plt.suptitle("correct  %.3f" % pctCrct, fontsize=28)
    _plt.savefig("%s.eps" % bfn, transparent=True)



def dists(mARp, bfn="low_high"):
    fig, ax = _plt.subplots(figsize=(13, 9))

    ###########
    _plt.subplot2grid((3, 5), (0, 0), colspan=4)
    _plt.plot(mARp.y, marker=".", ms=10, color="black")
    _plt.ylim(0, int(max(mARp.y)*1.05))

    _plt.xticks(fontsize=22)
    _plt.yticks(fontsize=22)
    _plt.xlabel("trials", fontsize=24)
    _plt.ylabel("counts", fontsize=24)
    _plt.subplot2grid((3, 5), (0, 4), colspan=1)
    _plt.hist(mARp.y, bins=range(int(max(mARp.y)*1.05)), orientation='horizontal', color="grey")
    _plt.ylim(0, int(max(mARp.y)*1.05))
    _plt.yticks([]);     _plt.xticks([])
    _plt.xlabel("# trials", fontsize=24)
    ###########
    _plt.subplot2grid((3, 5), (1, 0), colspan=4)
    x = _N.mean(mARp.Bsmpx[mARp.burn:], axis=0)
    _plt.plot(mARp.x, lw=2, color="blue")
    _plt.plot(x, lw=2, color="grey", ls="--")
    _plt.xlabel("trials", fontsize=24)
    _plt.ylabel("latent state", fontsize=24)
    _plt.xticks(fontsize=22)
    _plt.yticks([], fontsize=22)

    ###########
    _plt.subplot2grid((3, 5), (1, 4))
    _plt.hist(mARp.smp_m[mARp.burn:, 0], bins=_N.linspace(0, 1, 50), color="black")
    _plt.xlabel("low mix %", fontsize=24)
    _plt.xticks([0, 1], fontsize=22);  _plt.yticks([])
    fig.subplots_adjust(wspace=0.32, hspace=0.5, left=0.13, bottom=0.13)

    ###########
    _plt.subplot2grid((3, 5), (2, 0), colspan=5)
    cvs = _N.empty((mARp.burn+mARp.NMC, mARp.J))
    sp  = 1/(1 + _N.exp(-mARp.smp_u))

    for it in xrange(mARp.burn+mARp.NMC):
        for j in xrange(mARp.J):
            if mARp.smp_dty[it, j] == _cd.__NBML__:
                cvs[it, j] = 1 / (1 - sp[it, j])
            elif mARp.smp_dty[it, j] == _cd.__BNML__:
                cvs[it, j] = (1 - sp[it, j])
    _plt.plot(cvs[:, 0], color="black", lw=2)
    _plt.plot(cvs[:, 1], color="blue", lw=2)
    _plt.xlabel("Gibbs iters", fontsize=24)
    _plt.ylabel("intrinsic FF", fontsize=24)
    _plt.xticks(fontsize=22)

    _plt.savefig("%s.eps" % bfn, transparent=True)
