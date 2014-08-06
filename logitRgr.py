from scipy.optimize import minimize
import scipy.stats as _ss
from logitRgrFncs import LL, LLr, LLr2, jacb, build_lrn
from kassdirs import resFN

setname="rgr1oscAR"
xyp = _N.loadtxt(resFN("data.dat", dir=setname))   #  x, y, k are global variables

absRf = 8
Nall= xyp.shape[0]
k   = xyp.shape[1] - 2     #  k includes 1 offset component

N   = Nall    #  we can run with lesser amounts of data
x   = xyp[:, 0:k]
y   = xyp[:, k]
pr  = xyp[:, k+1]

TR  = 100                  #  how many times to try the minimization
k   = 3                    #  we are 
res = _N.empty((TR, k+1))  #  place to store result    + 1 for the LL
meth= "L-BFGS-B"

lrn = build_lrn(N, y, absRf)   #  lambda_R

######    make boundaries.  B's drawn from this range only
prbd = (-5, 5)
bds  = []
for ik in xrange(k):
    bds.append(prbd)

for tr in xrange(TR):
    A0  = 3*_N.random.randn(k)
    #Am  = minimize(LL, A0, args=(k, x, y), jac=jacb, method=meth, bounds=bds)

    Am  = minimize(LLr2, A0, args=(k, x, y, lrn), method=meth, bounds=bds)
    #Am  = minimize(LLr, A0, args=(k, x, y, lrn), method=meth, bounds=bds)
    res[tr, 0:k] = Am.x
    res[tr, k]   = Am.fun

######     FIGURE
fig = _plt.figure(figsize=(4.5*(k+1), 3.7))
for ik in xrange(k):
    fig.add_subplot(1, k+1, ik+1)
    _plt.scatter(res[:, ik], res[:, k])
    _plt.grid()

rks    = _N.array(_ss.rankdata(res[:, k], method="ordinal") - 1, dtype=_N.int).tolist()
indOrd = [rks.index(tr) for tr in range(TR)]

fig.add_subplot(1, k+1, ik+2)
_plt.plot(res[indOrd, k])

fig.subplots_adjust(left=0.05, right=0.95)
_plt.savefig("logitRgr-%s" % meth)

print res[indOrd[0]]

