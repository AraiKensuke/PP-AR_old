def dL(a, *args):
    nbs = args[0]       #  number basis splines
    M   = args[1]       #  number trials
    B   = args[2]       #  basis splines
    sts = args[3]       #  spike time   -- list of a list

    dL  = _N.empty(nbs)

    print a
    expV = _N.exp(_N.dot(B.T, a))
    for j in xrange(nbs):
        dL[j] = -M*dt*_N.dot(B.T[:, j], expV)
        for m in xrange(M):
            dL[j] += _N.sum(B.T[sts[m], j])
    print dL
    return dL

def d2L(a, *args):
    nbs = args[0]       #  number basis splines
    M   = args[1]       #  number trials
    B   = args[2]       #  basis splines
    sts = args[3]       #  spike time   -- list of a list

    d2L  = _N.zeros((nbs, nbs))

    expV = _N.exp(_N.dot(B.T, a))
    for j in xrange(nbs):
        for k in xrange(nbs):
            for m in xrange(M):
                d2L[j, k] -= M*dt*_N.dot(B.T[sts[m], j]*B.T[sts[m], k], expV[sts[m]])
    return d2L

