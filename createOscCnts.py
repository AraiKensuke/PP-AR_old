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
setname="oscCntsNonAR-2"   #  params_XXX.py   Results/XXX/params.py

model = "binomial"
if model=="binomial":
    #  mean rate is n x p
    n  = 10
    p = 0.08
elif model == "negative binomial":
    #  mean rate is r x p / (1 - p)    .  p prob success
    r  = 80
    p = 1 - 0.16
#  mean is (1-pL)*r/pL for low spike counts
u = _N.log(p / (1 - p))

#  now using the same data, I might want to run it with different EM settings

#  start with
stNz=5e-9    
#  data length N + 1.  n=0...N
#  state vector dim k.  For AR(p).  k=p

# observations.  dim-l vector  (l=1 here)

k     = 1
beta  = _N.zeros(k)
beta[0] = 1.   #  FOR pp-0, setting beta0 to 0 causes nan.  investigate

N     = 20000
tMon  = _N.linspace(0, N*dt, N)
trim  = 100

th    = 0.016
f     = th*500

#  Compared to step size <dt> of linear increase, we control the fluctuation
#  in step size to make noisy sin
t_orw, y= createDataAR(N+trim, _N.array([0.98]), 0.000002, 0.01, trim=trim)
amps, y = createDataAR(N, _N.array([0.98]), 0.001, 0.01, trim=0)
amps  /= (3*_N.std(amps))    #  more variability in amplitude
amps  += 1    #  amplitude is [1 - A, 1 + A]
Nz_t    = 8   #  larger == more jiggle in monotonic time
t_orw   *= (dt / _N.std(t_orw)) * Nz_t

m               = 0.5   #  final scaling of xSin
x  = m*amps*_N.sin(2*_N.pi*f*(tMon+t_orw))

cts = _N.empty(N)
data= _N.empty((N, 2))   #  x, spkct x 

for tr in xrange(N):
    ex = _N.exp(u + x[tr])
    if model == "binomial":
        cts[tr] = _N.random.binomial(n, ex / (1 + ex))
    else:
        cts[tr] = _N.random.negative_binomial(r, ex / (1 + ex))
    data[tr, 0] = x[tr]
    data[tr, 1]  = cts[tr]

_N.savetxt(resFN("cnt_data.dat", dir=setname, create=True), data, fmt="%.5f")

fig =_plt.figure(figsize=(13, 3.5*2))
fig.add_subplot(2, 1, 1)
_plt.plot(cts)
fig.add_subplot(2, 1, 2)
_plt.plot(x)
_plt.savefig(resFN("cts_ts", dir=setname, create=True))
_plt.close()

fp = open(resFN("params.py", dir=setname, create=True), "w")
fp.write("#  count data.  --  generated with binomial\n")
fp.write("u   = %.2e\n" % u)
fp.write("p   = %.2e\n" % p)
fp.write("n   = %d\n" % n)
fp.close()
