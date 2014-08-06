map_smpx = _N.empty(_d.N + 1)
for n in xrange(2, _d.N+1+2):
    cts, vals = _N.histogram(Bsmpx[0, :, n], bins=50)
    l = cts.tolist()
    i = l.index(cts.max())
    map_smpx[n-2] = 0.5*(vals[i] + vals[i+1])
