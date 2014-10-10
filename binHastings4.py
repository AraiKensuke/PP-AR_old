import time as _tm

#import binHastingsLib1ffi as _bHL
import binHastingsLib1ff as _bHL

#  Given data generated by binomial distribution with parameter
#  (p, n), fit this, and try to find (p, n)

######
burn = 5000
NMC  = 500
pT = 0.2
nT = 100
N  = 400
order=1

#  create data
cts = _N.random.binomial(nT, pT, size=N)

t1 = _tm.time()
ns   = _N.empty(burn + NMC, dtype=_N.int)
ps   = _N.empty(burn + NMC)
_bHL.MCMC(burn, NMC, cts, ns, ps, order)
t2 = _tm.time()
print (t2-t1)

fig=_plt.figure(figsize=(2*5, 2*4))
fig.add_subplot(2, 2, 1)
_plt.plot(ps)
fig.add_subplot(2, 2, 2)
_plt.plot(ns)
fig.add_subplot(2, 2, 3)
_plt.hist(ps, bins=_N.linspace(0, 1, 101))
fig.add_subplot(2, 2, 4)
_plt.hist(ns, bins=_N.linspace(0, 300, 301))
