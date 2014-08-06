from kassdirs import resFN
import numpy.polynomial.polynomial as _Npp
exf("kflib.py")

setname="rgr1oscAR"
###  Generate data for logit regression analysis.
###  covariates are oscillatory

N     = 5000
r     = 0.95
th    = 0.08
th1pi = _N.pi*th

alfa  = _N.array([r*(_N.cos(th1pi) + 1j*_N.sin(th1pi)), 
                  r*(_N.cos(th1pi) - 1j*_N.sin(th1pi))])

F     = (-1*_Npp.polyfromroots(alfa)[::-1][1:]).real

B   = _N.array([-1.8, 1.8, -1.7])    #  last component the offset 
k   = len(B)
covk= k - 1

xyp = _N.empty((N, k+2))           #  explanatory data
xyp[:, k-1] = 1

for ch in xrange(covk):
    x, y  = createDataAR(N, F, 1, 0.001, trim=0)
    x     /= _N.std(x)
    xyp[:, ch] = x

Bxi = _N.dot(B, xyp[:, 0:k].T)
eBxi= _N.exp(Bxi)
rs  = _N.random.rand(N)
xyp[:, k+1]  = eBxi / (1 + eBxi)    #  prob. of spk

absRf = 8
lastSpk = -(absRf+1)

xyp[:, k] = 0
for n in xrange(N):
    if (rs[n] < xyp[n, k+1]) and n > lastSpk + absRf:
        xyp[n, k] = 1
        lastSpk = n

"""
xyp[:, k]   = _N.array(rs < xyp[:, k+1], dtype=_N.int)   #  turn into 0s and 1s

absRf = 7
if absRf > 0:    #  add refractory period to data
    lastSpk = -(absRf+1)
    for n in xrange(N):
        if (n - lastSpk <= absRf) and (xyp[n, k] == 1):
            xyp[n, k] = 0
        if xyp[n, k] == 1:
            lastSpk = n
"""

sfmt=""
for ik in xrange(k - 1):
    sfmt += "% .3e  "
sfmt += "%d  %d  %.3e"   #  offset component, response variable, prob

_N.savetxt(resFN("data.dat", dir=setname, create=True), xyp, fmt=sfmt)
