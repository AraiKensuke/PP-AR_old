import patsy
from kassdirs import pracFN

setname="SquareShlder2"
N = 70
x = _N.linspace(0, N-1, N)

######  SMOOTH SHOULDER
a = 7.
b = 1.5
y = _N.exp((x-a)/b) / (1 + _N.exp((x-a)/b))
#y = 1 + 0.001*_N.random.randn(N)

######  ABRUPT SHOULDER
R   = 20
#  y[0] very small causes bad fitting
y[0:R] = _N.linspace(0.001, 1, R)**2

knotss = [_N.array([5, 10, 20, 30, 50]),
          _N.array([4, 8, 12, 25, 45]),
          _N.array([2, 7, 20, 45]),
          _N.array([4, 8, 12, 40]),
          _N.array([3, 7, 30]),
          _N.array([2, 7, ]) ]
          # _N.array([2, 5, 10, 20, 40]),
          # _N.array([2, 5, 10, 30]),
          # _N.array([2, 7, 12]),

kts    = 0
for knots in knotss:
    Gm    = patsy.bs(_N.linspace(0, N-1, N), knots=knots, include_intercept=True)
    Gm    = Gm.T
    phi   = _N.linalg.solve(_N.dot(Gm, Gm.T), _N.dot(Gm, _N.log(y)))
    
    fig   = _plt.figure(figsize=(5, 2*4))
    fig.add_subplot(2, 1, 1)
    _plt.title("%(k)s  %(d)d" % {"k" : str(knots), "d" : Gm.shape[0]})
    for i in xrange(Gm.shape[0]):
        _plt.plot(Gm[i, :], color="black", lw=1.5)
    fig.add_subplot(2, 1, 2)
    _plt.plot(x, y, lw=2)
    _plt.ylim(0, 1.2)
    _plt.plot(_N.exp(_N.dot(Gm.T, phi)), lw=1.5)
    _plt.savefig(pracFN("fit,kt=%d" % kts, dir=setname, create=True))
    _plt.close()
    kts += 1
