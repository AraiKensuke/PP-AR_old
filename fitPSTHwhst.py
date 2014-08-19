#  Fit PSTH with spike history
import patsy
import scipy.optimize as _sco
from kassdirs import resFN

exf("fitPSTHhistlib.py")
#exf("fitPSTHhistlibHist.py")

##########################################################
setname = "histPSTH7"

dCols = 3
dat   = _N.loadtxt(resFN("xprbsdN.dat", dir=setname))
l2gen   = _N.loadtxt(resFN("generate-l2.dat", dir=setname))   #  

N     = dat.shape[0]
M     = dat.shape[1] / dCols

sts   = []   #  will include one dummy spike
itvs  = []
rpsth = []

for tr in xrange(M):
    itvs.append([])
    lst = _N.where(dat[:, dCols*tr + 2] == 1)[0].tolist()
    lst.insert(0, int(-30*_N.random.rand()))    #  one dummy spike
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

dt= 0.001
x = _N.linspace(0., dt*(N-1), N)
nbs1 = 6
B = patsy.bs(x, df=nbs1, include_intercept=True)    #  spline basis
B  = B.T
#Gm = patsy.bs(_N.linspace(0, 0.2, 200), knots=_N.array([0.002, 0.006, 0.012, 0.02, 0.08, 0.15]), include_intercept=True)
#Gknts  = _N.array([0.005, 0.015, 0.08])
#Gknts  = _N.array([0.003, 0.01, 0.02, 0.04])
TM  = 60  # ms
#_Gm = patsy.bs(_N.linspace(0, dt*(TM-1), TM), knots=Gknts, include_intercept=True)
_Gm = patsy.bs(_N.linspace(0, dt*(TM-1), TM), df=6, include_intercept=True)
nbs2= _Gm.shape[1]
Gm = _N.zeros((N, nbs2))
Gm[0:TM] = _Gm

#Gm = patsy.bs(_N.linspace(0, 0.2, 200), df=4, include_intercept=True)
Gm = Gm.T
nbs2 = Gm.shape[0]


#####  Initialize
h, bs = _N.histogram(rpsth, bins=_N.linspace(0, 1000, 21))
bnsz   = 50
fs     = (h / (M * bnsz * dt))
apsth = _N.repeat(fs, bnsz)    #    piecewise boxy approximate PSTH
fsbnsz = _N.mean(fs) + _N.random.randn(1000)*0.5

aS     = _N.linalg.solve(_N.dot(B, B.T), _N.dot(B, _N.log(apsth)))
#aS     = _N.linalg.solve(_N.dot(B, B.T), _N.dot(B, _N.log(_N.mean(fsbnsz)*_N.ones(N))))

l2    = 0.5+_N.random.randn(TM)*0.001

phiS     = _N.linalg.solve(_N.dot(Gm[:, 0:TM], Gm[:, 0:TM].T), _N.dot(Gm[:, 0:TM], _N.log(l2)))
x     = _N.array(aS.tolist() + phiS.tolist())

print aS

#  If we estimate the the Jacobian, then even if h_dL 
sol = _sco.root(h_dL, x, jac=h_d2L, args=(nbs1, nbs2, M, B, Gm, sts, itvs, True, True, TM))
print sol.x[nbs1:]

L0 = h_L(aS, phiS, M, B, Gm, sts, itvs, TM)
L1 = h_L(sol.x[0:nbs1], sol.x[nbs1:nbs1+nbs2], M, B, Gm, sts, itvs, TM)

fig = _plt.figure(figsize=(5, 2*4))
#####
fig.add_subplot(2, 1, 1)
#  PSTH used to generate data
_plt.plot(dat[:, 1]/dt, lw=2, color="red")
#  Calculated fit
_plt.plot(_N.exp(_N.dot(B.T, sol.x[0:nbs1])), lw=2, color="black")
#  naive PSTH
_plt.plot(apsth, lw=2, ls="--", color="grey")
#  fit to initial naive PSTH guess
_plt.plot(_N.exp(_N.dot(B.T, aS)), lw=2, color="blue")
_plt.grid()
_plt.ylim(0, 1.1*max(_N.exp(_N.dot(B.T, sol.x[0:nbs1]))))
#####
fig.add_subplot(2, 1, 2)
# l2 obtained by fitting
_plt.plot(range(1, TM+1), _N.exp(_N.dot(Gm.T[0:TM], sol.x[nbs1:])), color="black", lw=2)
# l2 used to generate data
#_plt.title("%s" % str(Gknts))
_plt.plot(range(1, TM+1), l2gen[1:TM+1], lw="2", color="red")
_plt.grid()
_plt.xlim(0, TM)
_plt.ylim(0, 1.2)
fig.suptitle("init L %(i).1f   final L %(f).1f" % {"i" : L0, "f" : L1})
_plt.savefig(resFN("FIT.png", dir=setname), background="transparent")
_plt.close()


