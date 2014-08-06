import scipy.stats as _ss
from kassdirs import resFN

zch0 = 0
zch1 = 1

pcZs  = _N.empty(TR)
pcFXs = _N.empty(TR)

for tr in xrange(TR):
    fwf0   = _N.mean(zt[tr, 1000:1200, 1:, zch0], axis=0)
    fwf1   = _N.mean(zt[tr, 1000:1200, 1:, zch1], axis=0)

    pcZ, pv = _ss.pearsonr(fwf0, fwf1)

    fx0 = lpFilt(20, 26, 500, x[tr])
    fx1 = bpFilt(25, 55, 15, 65, 500, x[tr])

    pcFX, pv = _ss.pearsonr(fx0, fx1)

    pcZs[tr]  = pcZ
    pcFXs[tr] = pcFX

fig = _plt.figure()
_plt.hist(pcFXs, bins=_N.linspace(-0.3, 0.3, 30), color="black")
_plt.hist(pcZs,  bins=_N.linspace(-0.3, 0.3, 30), color="red")
_plt.savefig(resFN("ccBTWNztComps%(1)d,%(2)d" % {"1" : zch0, "2" : zch1}, dir=setdir))

