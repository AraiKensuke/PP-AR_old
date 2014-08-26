import numpy as _N
##################################
# Functions for the history term #
##################################

def h_L_func(aS, phiS, M, B, Gm, sts, itvs, TM, dt, mL=1):
    L    = 0
    for m in xrange(M):
        ITVS = len(itvs[m])

        for it in xrange(ITVS):    #  
            i0 = itvs[m][it][0]
            i1 = itvs[m][it][1]

            gt0= 0      #  if first is a real spike
            if i0 < 1:  #  reference to fake spike
                gt0 = 0-i0
            gt1= i1-i0
            if i0 < 1:  #  reference to fake spike
                i0 = 0  #  if there was a spike, i0==1
            #  The time integral

            expV = _N.exp(_N.dot(B.T[i0:i1], aS) + _N.dot(Gm.T[gt0:gt1], phiS))
            L += -dt*_N.sum(expV)  # integration

        i0 = itvs[m][0][0]
        iFR= 0      #  index of first real spike
        if i0 < 1:  #  not real spike
            iFR= 1      #  index of first real spike

        L += _N.sum(_N.dot(B.T[sts[m][iFR:]], aS))   #  First spk is fake
        allISIs = sts[m][1:] - sts[m][0:-1]
        L += _N.sum(_N.dot(Gm.T[allISIs], phiS))

    return L*mL

def h_L(a_phi, *args):
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
    mL   = args[11]
    aS   = a_phi[0:nbs1]
    phiS = a_phi[nbs1:]
    print aS
    print phiS

    L    = 0
    for m in xrange(M):
        ITVS = len(itvs[m])

        for it in xrange(ITVS):    #  
            i0 = itvs[m][it][0]    # spktime + 1
            i1 = itvs[m][it][1]

            gt0= 0      #  if first is a real spike
            gt1= i1-i0
            if i0 < 1:  #  reference to fake spike
                gt0 = 0-i0  #  > 0
                i0 = 0  #  if there was a spike, i0==1

            #  The time integral

            expV = _N.exp(_N.dot(B.T[i0:i1], aS) + _N.dot(Gm.T[gt0:gt1], phiS))
            L += -dt*_N.sum(expV)  # integration

        i0 = itvs[m][0][0]
        iFR= 0      #  index of first real spike
        if i0 < 1:  #  not real spike
            iFR= 1      #  index of first real spike

        L += _N.sum(_N.dot(B.T[sts[m][iFR:]], aS)) #  spk tms form psth, only real ones
        allISIs = sts[m][1:] - sts[m][0:-1]
        L += _N.sum(_N.dot(Gm.T[allISIs], phiS))

    print L*mL
    return L*mL

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
    mL   = args[11]
    
    aS   = a_phi[0:nbs1]
    phiS = a_phi[nbs1:]

    N    = B.shape[1]    #  
    dL  = _N.zeros(nbs1 + nbs2)

    ##  phi COMPONENTS
    for m in xrange(M):
        ITVS = len(itvs[m])

        for it in xrange(ITVS):    #  
            i0 = itvs[m][it][0]
            i1 = itvs[m][it][1]

            gt0= 0      #  if first is a real spike
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
                for j in xrange(nbs1, nbs1+nbs2):
                    dL[j] += -dt*_N.dot(Gm.T[gt0:gt1, j-nbs1], expV)

        i0 = itvs[m][0][0]
        iFR= 0      #  index of first real spike
        if i0 < 1:  #  not real spike
            iFR= 1      #  index of first real spike
        #print "%(iFR)d   %(st)d" % {"iFR" : iFR, "st" : sts[m][iFR]}
        ####  outside of previous for loop
        if doAl:
            for j in xrange(nbs1):
                dL[j] += _N.sum(B.T[sts[m][iFR:], j])   #  use 1st real spk
        if doPh:
            allISIs = sts[m][iFR+1:] - sts[m][iFR:-1]
            shrtISIs= allISIs[_N.where(allISIs < TM)[0]]
            for j in xrange(nbs1, nbs1 + nbs2):
                dL[j] += _N.sum(Gm.T[shrtISIs, j-nbs1])

    return dL*m

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
    m    = args[11]      # if using _sco.minimize, set this to -1

    aS   = a_phi[0:nbs1]
    phiS  = a_phi[nbs1:]

    d2L  = _N.zeros((nbs1 + nbs2, nbs1 + nbs2))

    ##  "DIAGONAL" elements   phiS
    for m in xrange(M):
        ITVS = len(itvs[m])

        for it in xrange(ITVS):    #  
            iFR= 0      #  index of first real spike
            i0 = itvs[m][it][0]
            i1 = itvs[m][it][1]

            gt0= 0      #  if first is a real spike
            if i0 < 1:  #  reference to fake spike
                gt0 = 0-i0
                iFR= 1      #  index of first real spike
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
                        d2L[j, k] += -dt*_N.dot((Gm.T[gt0:gt1, k-nbs1] * B.T[i0:i1, j]), expV)
            if doPh:
                for j in xrange(nbs1, nbs1 + nbs2):
                    for k in xrange(j, nbs1+nbs2):
                        d2L[j, k] += -dt*_N.dot((Gm.T[gt0:gt1, j-nbs1] * Gm.T[gt0:gt1, k-nbs1]), expV)

        for j in xrange(nbs1 + nbs2):
            for k in xrange(j, nbs1+nbs2):
                d2L[k, j] = d2L[j, k]

    return d2L*mL

def mkBounds(x, nbs1, nbs2):
    bds = _N.empty((nbs1+nbs2, 2))

    bds[0:nbs1, 0] = x[0:nbs1] - 2
    bds[0:nbs1, 1] = x[0:nbs1] + 2
    bds[nbs1:nbs1+3, 0] = -5
    bds[nbs1:nbs1+3, 1] = 0
    bds[nbs1+3:nbs1+nbs2, 0] = -5
    bds[nbs1+3:nbs1+nbs2, 1] = 3
    return bds

    # bds = []
    # for n in xrange(nbs1 + nbs2):
    #     bds.append([x[n] - 5, x[n] + 5)
    # return bds
