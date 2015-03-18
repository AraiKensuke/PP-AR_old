import time as _tm

import mcmcCntDist as _mCD
#import binHastingsLibSwUc as _mCD
from kflib import createDataAR
#  Given data generated by binomial distribution with parameter
#  (p, n), fit this, and try to find (p, n)

######
burn = 1000
NMC  = 1000
gen_cv = 0.7
gen_mn  = 10
N  = 400

#  create data
cts = _N.empty(N)

#xn, yn = createDataAR(N, _N.array([0.97, ]), 0.02, 0.1, trim=0)
xn = _N.sin(2*_N.pi*_N.linspace(0, 10, N))*0.2
#xn, yn = createDataAR(N, _N.array([0.995, ]), 0.01, 0.1, trim=0)
#xn = _N.ones(N)*0.3

if gen_cv > 1:
    gen_dist = _mCD.__NBML__
    pT       = 1 - 1. / gen_cv
    rT       = int(gen_mn * (1-pT)/pT)
else:
    gen_dist = _mCD.__BNML__
    pT       = 1 - gen_cv
    nT       = int(gen_mn / pT)
uT           = -_N.log(1./pT - 1)

for n in xrange(N):
    if gen_dist == _mCD.__BNML__:
        p      = 1 / (1 + _N.exp(-uT - xn[n]))  #  1 - px
        cts[n] = _N.random.binomial(nT, p)
    else:
        p      = 1 - 1 / (1 + _N.exp(-uT - xn[n]))  #  1 - px
        cts[n] = _N.random.negative_binomial(rT, p)

print "pT is %f" % pT
"""
Estimate starting parameters
"""

mnNaiv = _N.mean(cts)
cvNaiv = _N.std(cts)**2 / mnNaiv

print "cvNaiv   %f" % cvNaiv

if cvNaiv > 1:
    pNaiv = 1 - 1/cvNaiv;    dist = _mCD.__NBML__
else:
    pNaiv = (1 - cvNaiv);    dist = _mCD.__BNML__

mL      = 20000
pcdlog  = _N.empty(mL)        #precomputed logs
pcdlog[1:mL] = _N.log(_N.arange(1, mL))

t1 = _tm.time()
rns  = _N.empty(burn + NMC)
us   = _N.empty(burn + NMC)
dty  = _N.empty(burn + NMC)


"""
################   Ignore xn
_mCD.MCMC(burn, NMC, cts, rns, us, dty, pcdlog)

t2 = _tm.time()
print (t2-t1)

ps = 1 / (1 + _N.exp(-us))

fig=_plt.figure(figsize=(2*5, 4*4))
_plt.subplot2grid((4, 2), (0, 0))
_plt.plot(ps)
_plt.yscale("log")
_plt.subplot2grid((4, 2), (0, 1))
_plt.plot(rns)
_plt.subplot2grid((4, 2), (1, 0))
_plt.hist(ps, bins=_N.linspace(0, 1, 101))
_plt.subplot2grid((4, 2), (1, 1))
_plt.hist(rns, bins=_N.linspace(0, 300, 301))
_plt.subplot2grid((4, 2), (2, 0), colspan=2)
_plt.plot(dty)
_plt.ylim(-0.1, 1.1)

mns = _N.empty(burn + NMC)
cvs = _N.empty(burn + NMC)

for it in xrange(burn+NMC):
    if dty[it] == _mCD.__BNML__:
        mns[it] = rns[it]*ps[it]
        cvs[it] = (1-ps[it])
    else:
        mns[it] = rns[it]*ps[it] / (1 - ps[it])
        cvs[it] = 1/(1-ps[it])


_plt.subplot2grid((4, 2), (3, 0))
_plt.plot(mns)
_plt.subplot2grid((4, 2), (3, 1))
_plt.plot(cvs)
"""

################   Consider xn
_mCD.MCMC(burn, NMC, cts, rns, us, dty, pcdlog, xn=xn)

t2 = _tm.time()
print (t2-t1)

ps = 1 / (1 + _N.exp(-us))

fig=_plt.figure(figsize=(2*5, 4*4))
_plt.subplot2grid((4, 2), (0, 0))
_plt.plot(ps)
_plt.yscale("log")
_plt.subplot2grid((4, 2), (0, 1))
_plt.plot(rns)
_plt.subplot2grid((4, 2), (1, 0))
_plt.hist(ps, bins=_N.linspace(0, 1, 101))
_plt.subplot2grid((4, 2), (1, 1))
_plt.hist(rns, bins=_N.linspace(0, 300, 301))
_plt.subplot2grid((4, 2), (2, 0), colspan=2)
_plt.plot(dty)
_plt.ylim(-0.1, 1.1)

mns = _N.empty(burn + NMC)
cvs = _N.empty(burn + NMC)

for it in xrange(burn+NMC):
    if dty[it] == _mCD.__BNML__:
        mns[it] = rns[it]*ps[it]
        cvs[it] = (1-ps[it])
    else:
        mns[it] = rns[it]*ps[it] / (1 - ps[it])
        cvs[it] = 1/(1-ps[it])


_plt.subplot2grid((4, 2), (3, 0))
_plt.plot(mns)
_plt.subplot2grid((4, 2), (3, 1))
_plt.plot(cvs)

