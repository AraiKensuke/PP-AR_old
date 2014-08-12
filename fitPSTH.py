import patsy
import scipy.optimize as _sco
from kassdirs import resFN

exf("fitPSTHlib.py")

##########################################################
setname = "simpPSTH1"

dCols = 3
dat   = _N.loadtxt(resFN("xprbsdN.dat", dir=setname))
N     = dat.shape[0]
M     = dat.shape[1] / dCols

sts   = []
rpsth = []

for tr in xrange(M):
    lst = _N.where(dat[:, dCols*tr + 2] == 1)[0].tolist()
    sts.append(lst)
    rpsth.extend(lst)

x = _N.linspace(0., 1, 1000)
dt= 0.001
nbs = 6
B = patsy.bs(x, df=nbs, include_intercept=True)    #  spline basis
B  = B.T

#####  Initialize
h, bs = _N.histogram(rpsth, bins=_N.linspace(0, 1000, 21))

bnsz   = 50
fs     = h / (M * bnsz * dt)
fsbnsz = _N.repeat(fs, bnsz)
aS     = _N.linalg.solve(_N.dot(B, B.T), _N.dot(B, fsbnsz))

#####
sol = _sco.root(dL, aS, jac=d2L, args=(nbs, M, B, sts))

psth = _N.empty(1000)
for t in xrange(1000):
    psth[t] = _N.dot(B.T[t, :], sol.x)
_plt.plot(psth)
