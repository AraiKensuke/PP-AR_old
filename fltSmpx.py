from kassdirs import resFN
import scipy.signal as _ssig
import re as _re
exf("filter.py")

#  filter the x.  do 
fx    = _N.empty((TR, N+1))
for tr in xrange(TR):
    #fx[tr] = lpFilt(20, 26, 500, x[tr])
    fx[tr] = bpFilt(20, 45, 10, 55, 500, x[tr])  #(fpL, fpH, fsL, fsH, nyqf, y):

zch  = 0
pcsf = []
T   = 1000
allWFs = _N.empty(TR*T)
allXs  = _N.empty(TR*T)
for tr in xrange(TR):
    print tr
    fwf   = _N.mean(zt[tr, 300:500, 1:, zch], axis=0)
    allWFs[tr*T:(tr+1)*T] = fwf
    allXs[tr*T:(tr+1)*T] = fx[tr]
    #fwf  = bpFilt(8, 20, 1, 30, 500, wf)
    pcf, pvf = _ss.pearsonr(fwf, fx[tr])
    fig = _plt.figure(figsize=(12, 4))
    _plt.plot(fwf, lw=2, color="red")
    _plt.plot((fx[tr] * _N.std(fwf) / _N.std(fx[tr])), lw=2, color="black")
    _plt.suptitle("zch=(zch)d  pcf=%(pcf).3f" % {"pcf" : pcf, "zch" : zch})
    _plt.savefig(resFN("zt%(zch)d,tr=%(tr)d" % {"zch" : zch, "tr" : tr}, dir=setdir)) 
    _plt.close()

    plotWFandSpks(N, y[tr], [fx[tr]*_N.std(fwf)/_N.std(fx[tr]), fwf], sFilename=resFN("fx_zt%(zch)d_spks,tr=%(tr)d" % {"tr" : tr, "zch" : zch}, dir=setdir), tMult=2)

    pcsf.append(pcf)

fig = _plt.figure(figsize=(4, 3))
_plt.hist(pcsf, bins=_N.linspace(-0.8, 0.8, 26), color="black")
_plt.xlim(-0.7, 0.7)
_plt.axvline(x=_N.mean(pcsf), color="red", lw=2, ls="--")
_plt.grid()
_plt.savefig(resFN("pc_hist", dir=setdir))
_plt.close()

