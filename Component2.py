import pickle as _pkl
import scipy.signal as _ssig
from kassdirs import resFN, datFN
exf("mcmcARpFuncs.py")
exf("tmpSmplF.py")

exf("filter.py")

# setname="PCdat-1"
# k=15
# prH00=0

# fp= open(resFN("AR%(k)d_%(p).1f_dat.pkl" % {"k" : k, "p" : prH00}, dir=setname), "rb")
# dat = _pkl.load(fp)
# fp.close()



# allalfas = dat["all_alfas"]   #  ordered by freq
# wts      = dat["wt"]   #  ordered by freq
# uts      = dat["ut"]   #  ordered by freq

# burn     = dat["burn"]
# NMC      = dat["NMC"]

# C        = dat["C"]
# R        = dat["R"]

# fs       = dat["fs"]
# amps     = dat["amps"]


# model="bernoulli"
#loadDat(setname, model)

use = []
#  Params for spksFastOsc-1
c  = -0.04
for it in xrange(1, burn + NMC):
    if (fs[it, 0] > 0.075) and (fs[it, 0] < 0.085) and \
       (amps[it, 0] > 0.72) and (amps[it, 0] < 0.8):
        use.append(it)
#  Params for spksFastOsc-1
c  = -0.05
for it in xrange(1, burn + NMC):
    if (fs[it, 0] > 0.075) and (fs[it, 0] < 0.085) and \
       (amps[it, 0] > 0.72) and (amps[it, 0] < 0.8):
        use.append(it)


avgReals = _N.mean(uts[use, 0, 2:], axis=0)
avgImgs  = _N.mean(wts[use, 0, 2:], axis=0)

#  QUICK HACK
aI = _N.empty(N+1)

for n in xrange(N+1):
    amp = _N.sqrt(_N.abs(avgImgs[n] - c)*20)
    pol = (avgImgs[n] - c) > 0
    mp = -1
    if pol > 0:
        mp = 1
    
    aI[n] = mp * amp

ht_a  = _ssig.hilbert(aI - _N.mean(aI))
ph_a  = _N.empty(N+1)
fx    = bpFilt(15, 70, 1, 100, 500, x)   #  we want 
ht_x  = _ssig.hilbert(fx)
ph_x  = _N.empty(N+1)
for n in xrange(N + 1):
    ph_a[n] = base_q4atan(ht_a[n].real, ht_a[n].imag) / (2*_N.pi)
    ph_x[n] = base_q4atan(ht_x[n].real, ht_x[n].imag) / (2*_N.pi)

plot_cmptSpksAndX(N, ph_a, ph_x, y)
for ni in xrange(0, N+1, 500):
    _plt.xlim(ni, ni+500)
    _plt.savefig(resFN("phases%(k)d_%(p).1f,%(t)d.png" % {"k" : k, "p" : prH00, "t" : ni}, dir=setname))


avgImgsA  = _N.mean(wts[use, :, 2:], axis=0)
plot_cmptSpksAndX(N, avgImgsA[0, :], x, y)
_plt.xlim(1500, 2000)
_plt.savefig(resFN("comp1_%(k)d_%(p).1f,%(t)d.png" % {"k" : k, "p" : prH00, "t" : 1500}, dir=setname))
_plt.close()
plot_cmptSpksAndX(N, avgImgsA[1, :], x, y)
_plt.xlim(1500, 2000)
_plt.savefig(resFN("comp2_%(k)d_%(p).1f,%(t)d.png" % {"k" : k, "p" : prH00, "t" : 1500}, dir=setname))
_plt.close()
plot_cmptSpksAndX(N, avgImgsA[2, :], x, y)
_plt.xlim(1500, 2000)
_plt.savefig(resFN("comp3_%(k)d_%(p).1f,%(t)d.png" % {"k" : k, "p" : prH00, "t" : 1500}, dir=setname))
_plt.close()
