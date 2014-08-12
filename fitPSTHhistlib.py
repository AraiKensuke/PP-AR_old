
##################################
# Functions for the history term #
##################################

def h_dL(a_phi, *args):
    nbs1 = args[0]       #  number basis splines PSTH
    nbs2 = args[1]       #  number basis splines hist
    M    = args[2]       #  number trials
    B    = args[3]       #  basis splines PSTH
    Gm   = args[4]       #  basis splines hist
    sts  = args[5]       #  spike time   -- list of a list
    itvs = args[6]       #  intervals
    doAl = args[7]       #  fix either PSTH or history term
    doPh = args[8]

    aS   = a_phi[0:nbs1]
    phiS = a_phi[nbs1:]

    N    = B.shape[1]    #  
    dL  = _N.zeros(nbs1 + nbs2)

    ##  phi COMPONENTS
    for m in xrange(M):
        ITVS = len(itvs[m])
        nSpks= ITVS - 1

        for it in xrange(ITVS):    #  
            i0 = itvs[m][it][0]
            i1 = itvs[m][it][1]
            #  The time integral
            expV = _N.exp(_N.dot(B.T[i0:i1], aS) + _N.dot(Gm.T[0:i1-i0], phiS))

            if doAl:
                for j in xrange(nbs1):
                    dL[j] += -dt*_N.dot(B.T[i0:i1, j], expV)
            if doPh:
                for j in xrange(nbs1, nbs1+nbs2):
                    dL[j] += -dt*_N.dot(Gm.T[0:i1-i0, j-nbs1], expV)
        if doAl:
            for j in xrange(nbs1):
                dL[j] += _N.sum(B.T[sts[m][1:], j])
        if doPh:
            for j in xrange(nbs1, nbs1 + nbs2):
                dL[j] += _N.sum(Gm.T[sts[m][1:] - sts[m][0:-1], j-nbs1])

    #print dL
    return dL

def h_d2L(a_phi, *args):
    print "2**"
    nbs1 = args[0]       #  number basis splines PSTH
    nbs2 = args[1]       #  number basis splines hist
    M    = args[2]       #  number trials
    B    = args[3]       #  basis splines PSTH
    Gm   = args[4]       #  basis splines hist
    sts  = args[5]       #  spike time   -- list of a list
    itvs = args[6]       #  intervals
    doAl = args[7]
    doPh = args[8]

    aS   = a_phi[0:nbs1]
    phiS  = a_phi[nbs1:]

    d2L  = _N.zeros((nbs1 + nbs2, nbs1 + nbs2))

    ##  "DIAGONAL" elements   phiS
    for m in xrange(M):
        ITVS = len(itvs[m])
        nSpks= ITVS - 1

        for it in xrange(ITVS):    #  
            i0 = itvs[m][it][0]
            i1 = itvs[m][it][1]
            expV = _N.exp(_N.dot(B.T[i0:i1], aS) + _N.dot(Gm.T[0:i1-i0], phiS))

            if doAl:
                for j in xrange(nbs1):
                    for k in xrange(j, nbs1):
                        d2L[j, k] += -dt*_N.dot((B.T[i0:i1, j] * B.T[i0:i1, k]), expV)
            if doAl and doPh:
                for j in xrange(nbs1):
                    for k in xrange(nbs1, nbs1+nbs2):
                        d2L[j, k] += -dt*_N.dot((Gm.T[0:i1-i0, k-nbs1] * B.T[i0:i1, j]), expV)
            if doPh:
                for j in xrange(nbs1, nbs1 + nbs2):
                    for k in xrange(j, nbs1+nbs2):
                        d2L[j, k] += -dt*_N.dot((Gm.T[0:i1-i0, j-nbs1] * Gm.T[0:i1-i0, k-nbs1]), expV)

        for j in xrange(nbs1 + nbs2):
            for k in xrange(j, nbs1+nbs2):
                d2L[k, j] = d2L[j, k]

    return d2L
