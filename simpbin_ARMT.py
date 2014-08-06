import numpy.polynomial.polynomial as _Npp
import scipy.stats as _ss
exf("kflib.py")
import kstat as _ks
from kassdirs import resFN, datFN

def ccW12(y, N, blks):
    pcs = []
    for i in xrange(0, N - blks, blks):
        pc, pv = _ss.pearsonr(y[i:i + blks, 0], y[i:i + blks, 1])
        pcs.append(pc)

    return pcs

setname="oscCts-1MT"   #  params_XXX.py   Results/XXX/params.py

nWins = 1
model = "binomial"
if model=="binomial":
    #  mean rate is n x p
    n  = 30   #  keep n fixed.  Allow p0 to very from trial to trial
    p0 = 0.5
elif model == "negative binomial":
    #  mean rate is r x p / (1 - p)    .  p prob success
    r  = 80
    p0 = 1 - 0.16
#  mean is (1-pL)*r/pL for low spike counts

TR=3
N=800
trim= 100

cts = _N.empty((TR, N))

r1 = 0.95
th1= _N.pi*0.05    #  0.1 is about 50Hz

alfa  = _N.array([r1*(_N.cos(th1) + 1j*_N.sin(th1)), 
                  r1*(_N.cos(th1) - 1j*_N.sin(th1))])
ARcoeff          = (-1*_Npp.polyfromroots(alfa)[::-1][1:]).real

stNz0   = 2e-3
alldat= _N.empty((N, TR*2))   #  x, spkct x 
stNzs   = _N.empty(TR)
ps      = _N.empty(TR)
us      = _N.empty(TR)

stNzs[0] = stNz0
stNzs[1] = stNz0
stNzs[2] = stNz0/200.
for tr in xrange(TR):
    ps[tr]     = p0# * (1 + 0.3*_N.random.randn())
    us[tr]     = _N.log(ps[tr] / (1 - ps[tr]))
#    stNzs[tr]     = stNz0
    x,y   = createDataAR(N + trim, ARcoeff, stNzs[tr], stNzs[tr], trim=trim)
    ex = _N.exp(us[tr] + x)
    P  = ex / (1 + ex)
    if model == "binomial":
        cts = _N.random.binomial(n, P)
    else:
        cts = _N.random.negative_binomial(r, P)
    alldat[:, tr*2]      = x
    alldat[:, tr*2 + 1]  = cts

fig =_plt.figure(figsize=(13, 3.5*2*TR))
for tr in xrange(TR):
    fig.add_subplot(2*TR, 1, 1+2*tr)
    _plt.plot(alldat[:, tr*2+1])
    fig.add_subplot(2*TR, 1, 2+2*tr)
    _plt.plot(alldat[:, tr*2])
_plt.savefig(resFN("cts_ts", dir=setname, create=True))
_plt.close()

savesetMT(model, setname)
