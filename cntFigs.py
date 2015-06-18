import matplotlib.pyplot as _plt
import myColors as mC
import numpy as _N

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


