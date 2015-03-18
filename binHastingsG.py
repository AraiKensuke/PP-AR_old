import time as _tm

import binHastingsLibSwU as _bHL
#import binHastingsLibSwUc as _bHL
#  Given data generated by binomial distribution with parameter
#  (p, n), fit this, and try to find (p, n)

######
burn = 3000
NMC  = 1000
#pT = 1-0.2
pT = 0.5
nT = 10
N  = 300

#  create data
#cts = _N.random.binomial(nT, pT, size=N)
cts = _N.random.negative_binomial(nT, pT, size=N)
#cts = _N.random.poisson(10, size=N)
mL      = 20000
pcdlog  = _N.empty(mL)        #precomputed logs
pcdlog[1:mL] = _N.log(_N.arange(1, mL))

t1 = _tm.time()
rns  = _N.empty(burn + NMC)
us   = _N.empty(burn + NMC)
dty  = _N.empty(burn + NMC)

#####    GUESS INITIAL PARAMS
mn_cts = _N.mean(cts)
cv_cts = _N.std(cts)**2 / mn_cts

if cv_cts > 1:
    p0 = 1 - 1/cv_cts;    dist = _bHL.__NBML__
else:
    p0 = (1 - cv_cts);    dist = _bHL.__BNML__

#####    RUN MCMC
_bHL.MCMC(burn, NMC, cts, rns, us, dty, pcdlog, p0=p0, dist=dist)

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
    if dty[it] == _bHL.__BNML__:
        mns[it] = rns[it]*ps[it]
        cvs[it] = (1-ps[it])
    else:
        mns[it] = rns[it]*ps[it] / (1 - ps[it])
        cvs[it] = 1/(1-ps[it])


_plt.subplot2grid((4, 2), (3, 0))
_plt.plot(mns)
_plt.subplot2grid((4, 2), (3, 1))
_plt.plot(cvs)


