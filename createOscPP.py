import scipy.optimize as _sco
exf("kflib.py")
exf("mcmcARpFuncs.py")
from kassdirs import resFN, datFN
from mcmcARpPlot import plotWFandSpks
import utilities as _U

import pickle as _pk
import warnings
import numpy.polynomial.polynomial as _Npp

####   
#  _N.sin(t_orw)    #  t is an offseted random walk
dt = 0.001
setname="spksSloLmNtAR-1"   #  params_XXX.py   Results/XXX/params.py

#  now using the same data, I might want to run it with different EM settings

#  start with
stNz=5e-9    
#  data length N + 1.  n=0...N
#  state vector dim k.  For AR(p).  k=p

# observations.  dim-l vector  (l=1 here)

k     = 1
beta  = _N.zeros(k)
beta[0] = 1.   #  FOR pp-0, setting beta0 to 0 causes nan.  investigate

N     = 80000
tMon  = _N.linspace(0, N*dt, N)
trim  = 100

th    = 0.03
f     = th*500
absrefr=2


#  Compared to step size <dt> of linear increase, we control the fluctuation
#  in step size to make noisy sin
ar1_t = 0.97
ar1_a = 0.97
t_orw, y= createDataAR(N+trim, _N.array([ar1_t]), 0.000002, 0.01, trim=trim)
amps, y = createDataAR(N, _N.array([ar1_a]), 0.001, 0.01, trim=0)
amps  /= (3*_N.std(amps))    #  more variability in amplitude
amps  += 1    #  amplitude is [1 - A, 1 + A]
Nz_t    = 8   #  larger == more jiggle in monotonic time
t_orw   *= (dt / _N.std(t_orw)) * Nz_t

xSin  = amps*_N.sin(2*_N.pi*f*(tMon+t_orw))

u     = 3.4
m               = 0.5   #  final scaling of xSin

x, dN, prbs, fs = createDataPP(N, None, beta, u, stNz, trim=0, x=(xSin*m), absrefr=absrefr)

if len(x.shape) == 2:
    rows, cols = x.shape
else:
    rows       = x.shape[0]
N   = rows - 1    #  actual data length is N + 1

saveset(setname, noparam=True)

plotWFandSpks(N, dN, [x], sTitle="AR2 freq %(f).1fHz  num spks %(d).0f   spk Hz %(spkf).1fHz" % {"f" : (500*th), "d" : _N.sum(dN), "spkf" : (_N.sum(dN) / (N*0.001))}, sFilename=resFN("generative", dir=setname))

plotWFandSpks(N, dN, [x], sTitle="AR2 freq %(f).1fHz  num spks %(d).0f   spk Hz %(spkf).1fHz" % {"f" : (500*th), "d" : _N.sum(dN), "spkf" : (_N.sum(dN) / (N*0.001))}, sFilename=resFN("generative-det", dir=setname), intv=[1000, 2000])

spkTs = _N.where(dN == 1)[0]
isis  = _U.toISI([spkTs])
cv    = _N.std(isis[0]) / _N.mean(isis[0])
fig   = _plt.figure(figsize=(7, 3.5))
_plt.hist(isis[0], bins=range(0, 100), color="black")
_plt.grid()
_plt.suptitle("ISI cv %.2f" % cv)
_plt.savefig(resFN("ISIhist", dir=setname))
_plt.close()


#  save th
#  u
#  m
#  Nz_t
#  ar1_t
#  ar1_a
#  th

str =  "u=%.2f\n" % u
str += "m=%.2f\n" % m
str += "Nz_t=%.2f\n" % Nz_t
str += "ar1_t=%.3f\n" % ar1_t
str += "ar1_a=%.3f\n" % ar1_a
str += "th=%.3f\n" % th
fp = open(resFN("params.py", dir=setname), "w")
fp.write(str)
fp.close()

