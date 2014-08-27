import numpy as _N
##################################
# Functions for the history term #
##################################

def h_L_func(aS, phiS, M, B, Gm, sts, itvs, TM, dt, mL=1, offsetHaTM=0):
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

            offv = _N.zeros(gt1 - gt0)
            if gt1 > TM:
                offv[TM-gt0:] = offsetHaTM
            expV = _N.exp(_N.dot(B.T[i0:i1], aS) + _N.dot(Gm.T[gt0:gt1], phiS) + offv)

            L += -dt*_N.sum(expV)  # integration

        i0 = itvs[m][0][0]
        iFR= 0      #  index of first real spike
        if i0 < 1:  #  not real spike
            iFR= 1      #  index of first real spike

        L += _N.sum(_N.dot(B.T[sts[m][iFR:]], aS)) #  spk tms form psth, only real ones
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
    dt   = args[9]
    mL   = args[10]
    aS   = a_phi[0:nbs1]
    phiS = a_phi[nbs1:]

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
            gt1= i1-i0
            if i0 < 1:  #  reference to fake spike
                gt0 = 0-i0
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

def mkBounds(x, nbs1, nbs2, nbs2v):
    bds = _N.empty((nbs1+nbs2, 2))

    bds[0:nbs1, 0] = x[0:nbs1] - 2
    bds[0:nbs1, 1] = x[0:nbs1] + 2
    bds[nbs1:nbs1+nbs2v, 0] = -5
    bds[nbs1:nbs1+nbs2v, 1] = 1.
    bds[nbs1+nbs2v:, 0] = 0.     #  constant
    bds[nbs1+nbs2v:, 1] = 0.

    return bds

