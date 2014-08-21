import numpy as _N
##################################
# Functions for the history term #
##################################

def h_L(aS, phiS, M, B, Gm, sts, itvs, TM, dt, frstSpk):
    L    = 0
    for m in xrange(M):
        ITVS = len(itvs[m])

        for it in xrange(frstSpk, ITVS):    #  
            i0 = itvs[m][it][0]
            i1 = itvs[m][it][1]
            #  The time integral
            expV = _N.exp(_N.dot(B.T[i0:i1], aS) + _N.dot(Gm.T[0:i1-i0], phiS))
            L += -dt*_N.sum(expV)  # integration

        L += _N.sum(_N.dot(B.T[sts[m][1:]], aS))   #  First spk is fake
        allISIs = sts[m][frstSpk+1:] - sts[m][frstSpk:-1]
        shrtISIs= allISIs[_N.where(allISIs < TM)[0]]
        print len(shrtISIs)
        L += _N.sum(_N.dot(Gm.T[shrtISIs], phiS))

    return L

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
    TM   = args[9]       #  how much of history term expressed by spline
    dt   = args[10]
    frstSpk = args[11]   #  Ignore first spike or not.  (0 or 1)
    
    aS   = a_phi[0:nbs1]
    phiS = a_phi[nbs1:]
    #Gm[TM:, 0:nbs2-1] = 0    #  extended Gm so lambda(2) = 1 for t > TM
    #Gm[TM:, nbs2-1]   = 1./phiS[nbs2-1]

    N    = B.shape[1]    #  
    dL  = _N.zeros(nbs1 + nbs2)

    ##  phi COMPONENTS
    for m in xrange(M):
        ITVS = len(itvs[m])

        for it in xrange(frstSpk, ITVS):    #  
            i0 = itvs[m][it][0]
            i1 = itvs[m][it][1]
            Tunt = i1 - i0
            if i1 - i0 > TM:
                Tunt = TM

            gt0= 0      #  
            if i0 < 1:  #  reference to fake spike
                gt0 = 0-i0
            gt1= i1-i0
            if i0 < 1:  #  reference to fake spike
                i0 = 0  #  if there was a spike, i0==1
            #  The time integral
            expV = _N.exp(_N.dot(B.T[i0:i1], aS) + _N.dot(Gm.T[gt0:gt1], phiS))

            if doAl:
                for j in xrange(nbs1):
                    dL[j] += -dt*_N.dot(B.T[i0:i1, j], expV)
            if doPh:
                Tunt = i1 - i0
                if i1 - i0 > TM:
                    Tunt = TM
                for j in xrange(nbs1, nbs1+nbs2):
                    dL[j] += -dt*_N.dot(Gm.T[gt0:Tunt, j-nbs1], expV[gt0:Tunt])
        if doAl:
            for j in xrange(nbs1):
                dL[j] += _N.sum(B.T[sts[m][1:], j])   #  always use 1st real spk
        if doPh:
            allISIs = sts[m][frstSpk+1:] - sts[m][frstSpk:-1]
            shrtISIs= allISIs[_N.where(allISIs < TM)[0]]
            for j in xrange(nbs1, nbs1 + nbs2):
                dL[j] += _N.sum(Gm.T[shrtISIs, j-nbs1])

    print dL
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
    dt   = args[10]
    frstSpk = args[11]   #  Ignore first spike or not.  (0 or 1)

    aS   = a_phi[0:nbs1]
    phiS  = a_phi[nbs1:]

    #Gm[TM:, 0:nbs2-1] = 0
    #Gm[TM:, nbs2-1]   = 1./phiS[nbs2-1]

    d2L  = _N.zeros((nbs1 + nbs2, nbs1 + nbs2))

    ##  "DIAGONAL" elements   phiS
    for m in xrange(M):
        ITVS = len(itvs[m])

        for it in xrange(frstSpk, ITVS):    #  
            i0 = itvs[m][it][0]
            i1 = itvs[m][it][1]
            Tunt = i1 - i0
            if i1 - i0 > TM:
                Tunt = TM

            gt0= 0      #  
            if i0 < 1:  #  reference to fake spike
                gt0 = 0-i0
            gt1= i1-i0
            if i0 < 1:  #  reference to fake spike
                i0 = 0  #  if there was a spike, i0==1

            expV = _N.exp(_N.dot(B.T[i0:i1], aS) + _N.dot(Gm.T[gt0:gt1], phiS))

            if doAl:
                for j in xrange(nbs1):
                    for k in xrange(j, nbs1):
                        d2L[j, k] += -dt*_N.dot((B.T[i0:i1, j] * B.T[i0:i1, k]), expV)
            if doAl and doPh:
                for j in xrange(nbs1):
                    for k in xrange(nbs1, nbs1+nbs2):
                        d2L[j, k] += -dt*_N.dot((Gm.T[0:Tunt, k-nbs1] * B.T[i0:i0+Tunt, j]), expV[0:Tunt])
            if doPh:
                for j in xrange(nbs1, nbs1 + nbs2):
                    for k in xrange(j, nbs1+nbs2):
                        d2L[j, k] += -dt*_N.dot((Gm.T[0:Tunt, j-nbs1] * Gm.T[0:Tunt, k-nbs1]), expV[0:Tunt])

        for j in xrange(nbs1 + nbs2):
            for k in xrange(j, nbs1+nbs2):
                d2L[k, j] = d2L[j, k]

    return d2L
