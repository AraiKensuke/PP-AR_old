import patsy

#  Use this script to test whether a defined set of knots likely to 
#  fit history function likely to be encountered

N = 1000
a = 7
b = 1.5
c = 14.
d = 4.
ms = _N.arange(0, N)
l2 = (_N.exp((ms-a)/b) / (1 + _N.exp((ms-a)/b))) + 0.2*_N.exp(-0.5*(ms-c)*(ms-c) / (2*d*d))

knts= _N.array([0.006, 0.012, 0.02, 0.05, 0.08, 0.1, 0.15, 0.3, 0.5, 0.9])

Gm   = patsy.bs(_N.linspace(0, 1, 1000, endpoint=False), knots=knts, include_intercept=True)
Gm   = Gm.T

phiSi = _N.linalg.solve(_N.dot(Gm, Gm.T), _N.dot(Gm, _N.log(l2)))


_plt.plot(l2, color="black", lw=3)
_plt.plot(_N.exp(_N.dot(Gm.T, phiSi)), lw=2, color="red")
_plt.xlim(0, 100)
