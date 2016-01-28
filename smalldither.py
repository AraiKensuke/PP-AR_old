"""
import os
#  For theta at 2x, 80ms is 1 osc.  So dither lone spikes by 5ms

expt = "irreg_SM11"
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


absrefr  = 3
for tr in xrange(TRLS):
    print "tr %d" % tr
    st = _N.where(d[:, 2+COLS*tr] == 1)[0]
    d[:, 2+COLS*tr] = 0

    N  = len(st)
    for s in xrange(1, N - 1):
        lR = st[s] - st[s-1]
        rR = st[s+1] - st[s]
        
        lP = float(lR-absrefr) / (lR + rR)
        rP = float(rR-absrefr) / (lR + rR)
        lP = lP if (lP >= 0) else 0
        rP = rP if (rP >= 0) else 0
        tP = lP + rP
        
        if tP > 0:
            lP = lP / tP;        rP = rP / tP

            if (lP > 0) or (rP > 0):
                ot    = st[s]
                if (_N.random.rand() < lP) and (lP > 0):
                    dth = int(_N.abs(0.4*(lR-absrefr)*_N.random.randn()))
                    st[s] = st[s] - dth
                elif (rP > 0):
                    dth = int(_N.abs(0.4*(rR-absrefr)*_N.random.randn()))
                    st[s] = st[s] + dth
    d[st, 2+COLS*tr] = 1

s3 = "%.4f %.4f %d "
s4 = "%.4f %.4f %.4f %d "
fmt = s4 * TRLS if bRealDat else s3 * TRLS
_N.savetxt("../Results/%ss/xprbsdN.dat" % expt, d, fmt)


isisS = []
for tr in xrange(TRLS):
    st = _N.where(d[:, 2+COLS*tr] == 1)[0]
    isisS.extend(_N.diff(st))
"""
fig = _plt.figure(figsize=(4, 7))
fig.add_subplot(1, 2, 1)
_plt.hist(isis0, bins=_N.linspace(0, 150, 76))
fig.add_subplot(1, 2, 2)
_plt.hist(isisS, bins=_N.linspace(0, 150, 76))

