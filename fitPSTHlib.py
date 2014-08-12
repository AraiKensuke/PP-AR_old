#  We want to find the roots of a gradient to a scalar function
def L(a, *args):
    nbs = args[0]       #  number basis splines
    M   = args[1]       #  number trials
    B   = args[2]       #  basis splines
    sts = args[3]       #  spike time   -- list of a list

    L   = 0
    lam = _N.zeros(N)
    for t in xrange(N):
        lam[t] = _N.dot(B.T[t, :], a)

    L += -M*dt*_N.sum(lam)

    for m in xrange(M):
        for s in sts[m]:
            L += _N.log(lam[s])

    print L
    return -L

def dL(a, *args):
    nbs = args[0]       #  number basis splines
    M   = args[1]       #  number trials
    B   = args[2]       #  basis splines
    sts = args[3]       #  spike time   -- list of a list

    dL  = _N.empty(nbs)

    for j in xrange(nbs):
        dL[j] = -M*dt*_N.sum(B.T[:, j])
        for m in xrange(M):
            for s in sts[m]:
                dL[j] += B.T[s, j] / _N.dot(B.T[s, :], a)
    return dL

def d2L(a, *args):
    nbs = args[0]       #  number basis splines
    M   = args[1]       #  number trials
    B   = args[2]       #  basis splines
    sts = args[3]       #  spike time   -- list of a list

    d2L  = _N.zeros((nbs, nbs))

    for j in xrange(nbs):
        for k in xrange(nbs):
            for m in xrange(M):
                for s in sts[m]:
                    d2L[j, k] -= ((B.T[s, j]*B.T[s, k]) / (_N.dot(B.T[s, :], a)**2))
    return d2L

