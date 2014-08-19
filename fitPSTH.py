import patsy
import scipy.optimize as _sco
from kassdirs import resFN



##########################################################
setname = "simpPSTH2"

useIntvs = True

if useIntvs:
    exf("fitPSTHlib2.py")
else:
    exf("fitPSTHlib.py")
dCols = 3
dat   = _N.loadtxt(resFN("xprbsdN.dat", dir=setname))
N     = dat.shape[0]
M     = dat.shape[1] / dCols

sts   = []
itvs  = []
rpsth = []

if useIntvs:
    for tr in xrange(M):
        itvs.append([])
        lst = _N.where(dat[:, dCols*tr + 2] == 1)[0].tolist()
        lst.insert(0, -20)    #  one dummy spike
        sts.append(_N.array(lst))
        rpsth.extend(lst)
        Lm  = len(lst) - 1    #  number of spikes this trial
        itvs[tr].append([0, lst[1]+1])
        for i in xrange(1, Lm):
            itvs[tr].append([lst[i]+1, lst[i+1]+1])
        itvs[tr].append([lst[Lm]+1, N])
else:
    for tr in xrange(M):
        lst = _N.where(dat[:, dCols*tr + 2] == 1)[0].tolist()
        sts.append(lst)
        rpsth.extend(lst)

dt= 0.001
x = _N.linspace(0., (N-1)*dt, N)
nbs = 6
B = patsy.bs(x, df=nbs, include_intercept=True)    #  spline basis
B  = B.T

#####  Initialize
nbins = 20
h, bs = _N.histogram(rpsth, bins=_N.linspace(0, N, nbins+1))
bnsz   = (N/nbins)
fs     = h / (M * bnsz * dt)
apsth = _N.repeat(fs, bnsz)    #    piecewise boxy approximate PSTH
fsbnsz = _N.mean(fs) + _N.random.randn(1000)*0.5
aS     = _N.linalg.solve(_N.dot(B, B.T), _N.dot(B, _N.log(fsbnsz)))
print aS


#####
sol = _sco.root(dL, aS, jac=d2L, args=(nbs, M, B, sts, itvs))

psth = _N.exp(_N.dot(B.T, sol.x))

_plt.plot(psth)
_plt.plot(apsth)
_plt.plot(_N.exp(_N.dot(B.T, aS)))


