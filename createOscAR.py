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
setname="mt"   #  params_XXX.py   Results/XXX/params.py

#  now using the same data, I might want to run it with different EM settings

#  start with
stNz=1e-4
#  data length N + 1.  n=0...N
#  state vector dim k.  For AR(p).  k=p

# observations.  dim-l vector  (l=1 here)

r    = 0.97
th   = 0.06
th1pi= _N.pi*th
absrefr=2

alfa  = _N.array([r*(_N.cos(th1pi) + 1j*_N.sin(th1pi)), 
                  r*(_N.cos(th1pi) - 1j*_N.sin(th1pi))])

ARcoeff          = (-1*_Npp.polyfromroots(alfa)[::-1][1:]).real
u     = 3.4

#  AR weights.  In Kitagawa, a[0] is weight for most recent state
#  when used like dot(a, x), a needs to be stored in reverse direction
k   = len(ARcoeff)      #  this is user set2
beta  = _N.zeros(k)
beta[0] = 1.   #  FOR pp-0, setting beta0 to 0 causes nan.  investigate

N     = 100000
trim  = 1000
x, dN, prbs, fs = createDataPP(N+trim, ARcoeff, beta, u, stNz, p=1, trim=trim, absrefr=absrefr)
#x     = _N.zeros(N + trim)
#prbs  = _N.zeros(N + trim)

if len(x.shape) == 2:
    rows, cols = x.shape
else:
    rows       = x.shape[0]
N   = rows - 1    #  actual data length is N + 1

saveset(setname)

plotWFandSpks(N, dN, [x], sTitle="AR2 freq %(f).1fHz  num spks %(d).0f   spk Hz %(spkf).1fHz" % {"f" : (500*th), "d" : _N.sum(dN), "spkf" : (_N.sum(dN) / (N*0.001))}, sFilename=resFN("generative", dir=setname))

plotWFandSpks(N, dN, [x], sTitle="AR2 freq %(f).1fHz  num spks %(d).0f   spk Hz %(spkf).1fHz" % {"f" : (500*th), "d" : _N.sum(dN), "spkf" : (_N.sum(dN) / (N*0.001))}, sFilename=resFN("generative-det", dir=setname), intv=[1000, 2000])

spkTs = _N.where(dN == 1)[0]
isis  = _U.toISI([spkTs])
cv    = _N.std(isis[0]) / _N.mean(isis[0])
fig   = _plt.figure(figsize=(7, 3.5))
_plt.hist(isis[0], bins=range(0, 100), color="black")
_plt.xticks(range(0, 100, 10))
_plt.grid()
_plt.suptitle("ISI cv %.2f" % cv)
_plt.savefig(resFN("ISIhist", dir=setname))
_plt.close()

