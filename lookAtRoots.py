import pickle
import mcmcFigs as mF

def lAR(dir, fMult=1):
    dmp = open("../Results/%s/smpls.dump" % dir, "rb")
    lm = pickle.load(dmp)
    dmp.close()
    
    fs   = (lm["fs"]/fMult)*500  #  nyquist
    amps = lm["amps"]

    cmps = fs.shape[1]

    fig  = _plt.figure(figsize=(6, 4))
    ax   = fig.add_subplot(1, 1, 1)
    _plt.plot(fs[1:, 0], amps[1:, 0], color="black")

    clrs = ["grey", "blue", "cyan", "pink", "green", "orange"]
    
    for i in xrange(1, cmps):
        _plt.plot(fs[:, i], amps[:, i], color=clrs[i-1])
    for i in xrange(cmps):  #  do this last, no cover up
        if i == 0:
            _plt.plot(fs[-1, i], amps[-1, i], color="yellow", marker="*", ms=15)
        else:
            _plt.plot(fs[-1, i], amps[-1, i], color="red", marker="*", ms=15)

    mF.setTicksAndLims(xlabel="modulus", ylabel="frequency", xticks=range(0, 501, 100), yticks=[0, 0.25, 0.5, 0.75, 1], yticksD=["0", "0.25", "0.5", "0.75", "1"], tickFS=20, labelFS=22, xlim=[-3, 503], ylim=[-0.03, 1.03])
    mF.arbitaryAxes(ax, axesVis=[True, True, False, False], x_tick_positions="bottom", y_tick_positions="left")
    _plt.ylabel("modulus")
    _plt.xlabel("frequency (Hz)")
    fig.subplots_adjust(left=0.18, bottom=0.18, top=0.95, right=0.95)

    _plt.savefig("../Results/%s/RootsIter.eps" % dir)
    _plt.close()


fMult=1
"""
for i in xrange(1, 5):
   for j in [1, 3, 5]:
       dir="irreg_%(i)d/mcmc%(j)d" % {"i" : i, "j" : j}
       lAR(dir, fMult=fMult)
"""
dir="irreg_1/insf3"
lAR(dir)
