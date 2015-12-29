#  For theta at 2x, 80ms is 1 osc.  So dither lone spikes by 5ms

d = _N.loadtxt("080402-0-121-ltrthM2/xprbsdN.dat")

absrefr = 8
TRLS    = d.shape[1]/4
for tr in xrange(TRLS):
    st = _N.where(d[:, 2+4*tr] == 1)[0]
    d[:, 2+4*tr] = 0

    N  = len(st)
    for s in xrange(1, N - 1):
        lR = st[s] - st[s-1] - absrefr
        rR = st[s+1] - st[s] - absrefr
        lR = lR if lR > 0 else 0
        rR = rR if rR > 0 else 0

        nt = int((lR+rR) * _N.random.rand())
        st[s] = st[s-1] + nt
    
    d[st, 2+4*tr] = 1

fmt = "%.4f %.4f %d %.4f " * TRLS
_N.savetxt("080402-0-121-ltrthM2s/xprbsdN.dat", d, fmt)
