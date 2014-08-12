#  Fit PSTH with spike history
import patsy
import scipy.optimize as _sco
from kassdirs import resFN

exf("fitPSTHhistlib.py")

##########################################################
setname = "histPSTH"

dCols = 3
dat   = _N.loadtxt(resFN("xprbsdN.dat", dir=setname))
N     = dat.shape[0]
M     = dat.shape[1] / dCols

sts   = []   #  will include one dummy spike
itvs  = []
rpsth = []

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

    #  [-20, 16, 66]       spks
    #  [0, 17), [17, 67)   intvs.  first 

    #  [-20, 999]       spks
    #  [0, 1000), [1000, 1000)   intvs.  first    a[1000:1000] empty

x = _N.linspace(0., 1, 1000)
dt= 0.001
nbs1 = 4
B = patsy.bs(x, df=nbs1, include_intercept=True)    #  spline basis
B  = B.T
#Gm = patsy.bs(_N.linspace(0, 0.2, 200), knots=_N.array([0.002, 0.006, 0.012, 0.02, 0.08, 0.15]), include_intercept=True)
Gm = patsy.bs(_N.linspace(0, 0.2, 200), knots=_N.array([0.002, 0.006, 0.012, 0.02, 0.08, 0.15]), include_intercept=True)
#Gm = patsy.bs(_N.linspace(0, 0.2, 200), df=4, include_intercept=True)
Gm = Gm.T
nbs2 = Gm.shape[0]

#####  Initialize
h, bs = _N.histogram(rpsth, bins=_N.linspace(0, 1000, 21))
bnsz   = 50
fs     = h / (M * bnsz * dt)
fsbnsz = _N.repeat(fs, bnsz)    #    piecewise boxy approximate PSTH
aS     = _N.linalg.solve(_N.dot(B, B.T), _N.dot(B, _N.log(fsbnsz)))
#aS     = _N.linalg.solve(_N.dot(B, B.T), _N.dot(B, _N.log(_N.mean(fsbnsz)*_N.ones(N))))

r     = 30

M     = 20
Ls    = _N.empty(M)
p0s   = _N.linspace(0.5, 1., r)

l2    = 1 + _N.random.randn(200)*0.02
#l2[0:r] = p0s
phiS     = _N.linalg.solve(_N.dot(Gm, Gm.T), _N.dot(Gm, _N.log(l2)))
x     = _N.array(aS.tolist() + phiS.tolist())

print "----"
print x
print "----"
#  If we estimate the the Jacobian, then even if h_dL 
sol = _sco.root(h_dL, x, jac=h_d2L, args=(nbs1, nbs2, M, B, Gm, sts, itvs, True, True))
print sol.x[nbs1:]


fig = _plt.figure(figsize=(5, 2*4))
fig.add_subplot(2, 1, 1)
psth = _N.empty(1000)
for t in xrange(1000):
    psth[t] = _N.dot(B.T[t, :], sol.x[0:nbs1])
_plt.plot(_N.exp(psth))

fig.add_subplot(2, 1, 2)
hist = _N.dot(Gm.T, sol.x[nbs1:])
_plt.plot(_N.exp(hist))

