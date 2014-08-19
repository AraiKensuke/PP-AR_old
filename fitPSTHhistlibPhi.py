
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
    #  history term expressed in terms of splines until time TM
    TM   = args[9]
    
    aS   = a_phi[0:nbs1]
    phiS = a_phi[nbs1:]
    #Gm[TM:, 0:nbs2-1] = 0    #  extended Gm so lambda(2) = 1 for t > TM
    #Gm[TM:, nbs2-1]   = 1./phiS[nbs2-1]

    N    = B.shape[1]    #  
    dL  = _N.zeros(nbs1 + nbs2)

    ##  phi COMPONENTS
    print M
    for m in xrange(M):
        ITVS = len(itvs[m])
        nSpks= ITVS - 1

        for it in xrange(ITVS):    #  
            i0 = itvs[m][it][0]
            i1 = itvs[m][it][1]
            #  The time integral
            expV = _N.exp(_N.dot(B.T[i0:i1], aS))

            if doAl:
                for j in xrange(nbs1):
                    dL[j] += -dt*_N.dot(B.T[i0:i1, j], expV)
        if doAl:
            for j in xrange(nbs1):
                dL[j] += _N.sum(B.T[sts[m][1:], j])

    print dL[0:nbs1]
    return dL

def h_d2L(a_phi, *args):
    nbs1 = args[0]       #  number basis splines PSTH
    nbs2 = args[1]       #  number basis splines hist
    M    = args[2]       #  number trials
    B    = args[3]       #  basis splines PSTH
    Gm   = args[4]       #  basis splines hist
    sts  = args[5]       #  spike time   -- list of a list
    itvs = args[6]       #  intervals
    doAl = args[7]
    doPh = args[8]
    TM   = args[9]

    aS   = a_phi[0:nbs1]
    phiS  = a_phi[nbs1:]

    #Gm[TM:, 0:nbs2-1] = 0
    #Gm[TM:, nbs2-1]   = 1./phiS[nbs2-1]

    d2L  = _N.zeros((nbs1 + nbs2, nbs1 + nbs2))

    ##  "DIAGONAL" elements   phiS
    for m in xrange(M):
        ITVS = len(itvs[m])
        nSpks= ITVS - 1

        for it in xrange(ITVS):    #  
            i0 = itvs[m][it][0]
            i1 = itvs[m][it][1]
            expV = _N.exp(_N.dot(B.T[i0:i1], aS))

            if doAl:
                for j in xrange(nbs1):
                    for k in xrange(nbs1):
                        d2L[j, k] += -dt*_N.dot((B.T[i0:i1, j] * B.T[i0:i1, k]), expV)

    return d2L
