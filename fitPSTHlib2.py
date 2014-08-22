
##################################
# Functions for the history term #
##################################

def dL(a, *args):
    nbs = args[0]       #  number basis splines PSTH
    M    = args[1]       #  number trials
    B    = args[2]       #  basis splines PSTH
    sts  = args[3]       #  spike time   -- list of a list
    itvs = args[4]       #  intervals
    
    dL  = _N.zeros(nbs)

    ##  phi COMPONENTS
    for m in xrange(M):
        ITVS = len(itvs[m])
        nSpks= ITVS - 1

        #print nSpks
        for it in xrange(ITVS):    #  
            i0 = itvs[m][it][0]
            i1 = itvs[m][it][1]
            #  The time integral
            expV = _N.exp(_N.dot(B.T[i0:i1], a))

            for j in xrange(nbs):
                dL[j] += -dt*_N.dot(B.T[i0:i1, j], expV)
        #print dL
        for j in xrange(nbs):
            dL[j] += _N.sum(B.T[sts[m][1:], j])

    print dL
    return dL

def d2L(a, *args):
    nbs = args[0]       #  number basis splines PSTH
    M    = args[1]       #  number trials
    B    = args[2]       #  basis splines PSTH
    sts  = args[3]       #  spike time   -- list of a list
    itvs = args[4]       #  intervals

    d2L  = _N.zeros((nbs, nbs))

    ##  "DIAGONAL" elements   phiS
    for m in xrange(M):
        ITVS = len(itvs[m])
        nSpks= ITVS - 1

        for it in xrange(ITVS):    #  
            i0 = itvs[m][it][0]
            i1 = itvs[m][it][1]
            expV = _N.exp(_N.dot(B.T[i0:i1], a))

            for j in xrange(nbs):
                for k in xrange(nbs):
                    d2L[j, k] += -dt*_N.dot((B.T[i0:i1, j] * B.T[i0:i1, k]), expV)

    return d2L
