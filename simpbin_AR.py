import numpy.polynomial.polynomial as _Npp
import scipy.stats as _ss
exf("kflib.py")
import kstat as _ks
from kassdirs import resFN, datFN
#warnings.filterwarnings("error")

def ccW12(y, N, blks):
    pcs = []
    for i in xrange(0, N - blks, blks):
        pc, pv = _ss.pearsonr(y[i:i + blks, 0], y[i:i + blks, 1])
        pcs.append(pc)

    return pcs

setname="nooscCts-1"   #  params_XXX.py   Results/XXX/params.py

model = "binomial"
if model=="binomial":
    #  mean rate is n x p
    n  = 10
    p = 0.15
elif model == "negative binomial":
    #  mean rate is r x p / (1 - p)    .  p prob success
    r  = 80
    p = 1 - 0.16
#  mean is (1-pL)*r/pL for low spike counts
u = _N.log(p / (1 - p))
TRIALS=3000

cts = _N.empty(TRIALS)

r1 = 0.95
th1= _N.pi*0.08    #  0.1 is about 50Hz
r2 = 0.95
th2= _N.pi*0.018    #  0.1 is about 50Hz

alfa  = _N.array([r1*(_N.cos(th1) + 1j*_N.sin(th1)), 
                  r1*(_N.cos(th1) - 1j*_N.sin(th1))])
#                  r2*(_N.cos(th2) + 1j*_N.sin(th2)), 
#                  r2*(_N.cos(th2) - 1j*_N.sin(th2))])

phi          = (-1*_Npp.polyfromroots(alfa)[::-1][1:]).real

e   = 4e-7
data= _N.empty((TRIALS, 2))   #  x, spkct x 

trim= 100
x,y   = createDataAR(TRIALS + trim, phi, e, e, trim=trim)

for tr in xrange(TRIALS):
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
fp.write("ARcoeff=_N.array(%s)\n" % str(phi))
fp.write("alfa=_N.array(%s)\n" % str(alfa))
fp.write("e    = %.2e\n" % e)
fp.write("u   = %.2e\n" % u)
fp.write("p   = %.2e\n" % p)
fp.write("n   = %d\n" % n)
fp.close()
