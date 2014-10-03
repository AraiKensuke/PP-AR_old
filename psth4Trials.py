from kassdirs import resFN
from kflib import quickPSTH
setname="080402-0-121-cL"

_dat = _N.loadtxt(resFN("xprbsdN.dat", dir=setname))

tr0  = 0
tr1  = 87
trials = _N.array([tr0, tr1])

dat  = _dat[:, tr0*3:tr1*3]
quickPSTH(dat, tr1-tr0, 3, plot=True, fn=resFN("psth_%(1)d_%(2)d" % {"1" : tr0, "2" : tr1}, dir=setname))
