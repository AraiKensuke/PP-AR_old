import scipy.optimize as _sco
exf("kflib.py")
exf("mcmcARpFuncs.py")
from kassdirs import resFN, datFN
from mcmcARpPlot import plotWFandSpks
import utilities as _U

import pickle as _pk
import warnings
import numpy.polynomial.polynomial as _Npp


dt = 0.001
setname="spksFastOsc-Sin"   #  params_XXX.py   Results/XXX/params.py

#  now using the same data, I might want to run it with different EM settings

#  start with
stNz=5e-9
#  data length N + 1.  n=0...N
#  state vector dim k.  For AR(p).  k=p

# observations.  dim-l vector  (l=1 here)

k     = 1
beta  = _N.zeros(k)
beta[0] = 1.   #  FOR pp-0, setting beta0 to 0 causes nan.  investigate

N     = 50000
trim  = 0
u     = 3
th    = 0.08
f     = th*500
t     = _N.linspace(0, (N-1)*dt, N)
xSin  = 2*_N.sin(2*_N.pi*f*t)
x, dN, prbs, fs = createDataPP(N+trim, None, beta, u, stNz, trim=trim, x=xSin)


#x     = _N.zeros(N + trim)
#prbs  = _N.zeros(N + trim)

if len(x.shape) == 2:
    rows, cols = x.shape
else:
    rows       = x.shape[0]
N   = rows - 1    #  actual data length is N + 1

saveset(setname, noparam=True)

plotWFandSpks(N, dN, [x], sTitle="AR2 freq %(f).1fHz  num spks %(d).0f   spk Hz %(spkf).1fHz" % {"f" : (500*th), "d" : _N.sum(dN), "spkf" : (_N.sum(dN) / (N*0.001))}, sFilename=resFN("generative", dir=setname))

spkTs = _N.where(dN == 1)[0]
isis  = _U.toISI([spkTs])
cv    = _N.std(isis[0]) / _N.mean(isis[0])
fig   = _plt.figure(figsize=(7, 3.5))
_plt.hist(isis[0], bins=range(0, 100), color="black")
_plt.grid()
_plt.suptitle("ISI cv %.2f" % cv)
_plt.savefig(resFN("ISIhist", dir=setname))
_plt.close()
