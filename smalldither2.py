import os
#  For theta at 2x, 80ms is 1 osc.  So dither lone spikes by 5ms

expt = "irreg_SM43"
bRealDat = False
COLS = 4 if bRealDat else 3

if not os.access("../Results/%ss" % expt, os.F_OK):
    os.mkdir("../Results/%ss" % expt)
d = _N.loadtxt("../Results/%s/xprbsdN.dat" % expt)

TRLS    = d.shape[1]/COLS
isis0 = []
for tr in xrange(TRLS):
    st = _N.where(d[:, 2+COLS*tr] == 1)[0]
    isis0.extend(_N.diff(st))

hstgm = _N.histogram(isis0, bins=_N.linspace(0, max(isis0)+1, max(isis0)+2))

cdf = _N.zeros(hstgm[0].shape[0]+1)
for l in xrange(len(hstgm[0])):
    cdf[l+1] = cdf[l] + hstgm[0][l]

cdf /= _N.sum(hstgm[0])
    
N    = d.shape[0]

s    = _N.zeros(2*N, dtype=_N.int)
for tr in xrange(TRLS):
    d[:, 2+COLS*tr] = 0
    t = 0
    s[:] = 0
    while t < 2*N:
        dt = _N.where(_N.random.rand() < cdf)[0][0]
        if t + dt < 2*N:
            s[t + dt] = 1
        t += dt
    st = _N.where(s[N:] == 1)[0]
    d[st, 2+COLS*tr] = 1

s3 = "%.4f %.4f %d "
s4 = "%.4f %.4f %.4f %d "
fmt = s4 * TRLS if bRealDat else s3 * TRLS
_N.savetxt("../Results/%ss/xprbsdN.dat" % expt, d, fmt)

isisS = []
for tr in xrange(TRLS):
    st = _N.where(d[:, 2+COLS*tr] == 1)[0]
    isisS.extend(_N.diff(st))

fig = _plt.figure(figsize=(4, 7))
fig.add_subplot(2, 1, 1)
_plt.hist(isis0, bins=_N.linspace(0, 150, 76))
fig.add_subplot(2, 1, 2)
_plt.hist(isisS, bins=_N.linspace(0, 150, 76))
_plt.savefig("../Results/%ss/ISIcmp" % expt)
