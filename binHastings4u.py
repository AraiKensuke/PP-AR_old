import time as _tm

import binHastingsLib1uu as _bHL
#import binHastingsLib1u as _bHL
#import binHastingsLib2 as _bHL
#  Given data generated by binomial distribution with parameter
#  (p, n), fit this, and try to find (p, n)

######
burn = 2000
NMC  = 2000
pT = 0.1
nT = 100
N  = 400

#  create data
cts = _N.random.binomial(nT, pT, size=N)
mL      = 20000
pcdlog  = _N.empty(mL)        #precomputed logs
pcdlog[1:mL] = _N.log(_N.arange(1, mL))

t1 = _tm.time()
ns   = _N.empty(burn + NMC)
us   = _N.empty(burn + NMC)
_bHL.MCMC(burn, NMC, cts, ns, us, 1, pcdlog)
t2 = _tm.time()
print (t2-t1)

ps = 1 / (1 + _N.exp(-us))

fig=_plt.figure(figsize=(2*5, 2*4))
fig.add_subplot(2, 2, 1)
_plt.plot(ps)
fig.add_subplot(2, 2, 2)
_plt.plot(ns)
fig.add_subplot(2, 2, 3)
_plt.hist(ps, bins=_N.linspace(0, 1, 101))
fig.add_subplot(2, 2, 4)
_plt.hist(ns, bins=_N.linspace(0, 300, 301))
