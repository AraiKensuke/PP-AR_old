import numpy as _N
##################################
# Functions for the history term #
##################################

def h_L_func(aS, phiS, M, B, Gm, sts, itvs, allISIs, gt01, dt, mL=1):
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
        L += _N.sum(_N.dot(Gm.T[allISIs[m]], phiS))

    print L*mL
    return L*mL

def h_L(a_phi, *args):
    nbs1 = args[0]       #  number basis splines PSTH
    nbs2 = args[1]       #  number basis splines hist
    M    = args[2]       #  number trials
    B    = args[3]       #  basis splines PSTH
    Gm   = args[4]       #  basis splines hist
    sts  = args[5]       #  spike time   -- list of a list
    itvs = args[6]       #  intervals
    allISIs= args[7]
    gt01 = args[8]
    doAl = args[9]       #  fix either PSTH or history term
    doPh = args[10]
    #  history term expressed in terms of splines until time TM
    dt   = args[11]
    mL   = args[12]
    aS   = a_phi[0:nbs1]
    phiS = a_phi[nbs1:]

    L    = 0
    for m in xrange(M):
        ITVS = len(itvs[m])

        for it in xrange(ITVS):    #  
            i0, i1 = itvs[m][it]    # spktime + 1
            if i0 < 1:  #  reference to fake spike
                i0 = 0  #  if there was a spike, i0==1

            gt0, gt1 = gt01[m][it]

            #  The time integral

            expV = _N.exp(_N.dot(B.T[i0:i1], aS) + _N.dot(Gm.T[gt0:gt1], phiS))
            L += -dt*_N.sum(expV)  # integration

        i0 = itvs[m][0][0]
        iFR= 0      #  index of first real spike
        if i0 < 1:  #  not real spike
            iFR= 1      #  index of first real spike

        L += _N.sum(_N.dot(B.T[sts[m][iFR:]], aS)) #  spk tms form psth, only real ones
        L += _N.sum(_N.dot(Gm.T[allISIs[m]], phiS))

    print L
    return L*mL

"""
def h_dL(a_phi, *args):
    nbs1 = args[0]       #  number basis splines PSTH
    nbs2 = args[1]       #  number basis splines hist
    M    = args[2]       #  number trials
    B    = args[3]       #  basis splines PSTH
    Gm   = args[4]       #  basis splines hist
    sts  = args[5]       #  spike time   -- list of a list
    itvs = args[6]       #  intervals
    allISIs= args[7]
    gt01 = args[8]
    doAl = args[9]       #  fix either PSTH or history term
    doPh = args[10]
    #  history term expressed in terms of splines until time TM
    dt   = args[11]
    mL   = args[12]
    aS   = a_phi[0:nbs1]
    phiS = a_phi[nbs1:]

    dL  = _N.zeros(nbs1 + nbs2)

    for m in xrange(M):
        ITVS = len(itvs[m])

        for it in xrange(ITVS):    #  
            i0, i1 = itvs[m][it]    # spktime + 1
            if i0 < 1:  #  reference to fake spike
                i0 = 0  #  if there was a spike, i0==1

            gt0, gt1 = gt01[m][it]

            #  The time integral

            expV = _N.exp(_N.dot(B.T[i0:i1], aS) + _N.dot(Gm.T[gt0:gt1], phiS))
            dL += -dt*_N.dot(B.T[i0:i1], expV)  # integration

        i0 = itvs[m][0][0]
        iFR= 0      #  index of first real spike
        if i0 < 1:  #  not real spike
            iFR= 1      #  index of first real spike

        L += _N.sum(_N.dot(B.T[sts[m][iFR:]], aS)) #  spk tms form psth, only real ones

    print L
    return dL*mL
"""


def mkBounds(x, nbs1, nbs2, nbs2v):
    bds = _N.empty((nbs1+nbs2, 2))

    bds[0:nbs1, 0] = x[0:nbs1] - 3
    bds[0:nbs1, 1] = x[0:nbs1] + 3
    bds[nbs1:nbs1+nbs2v, 0] = -5
    bds[nbs1:nbs1+nbs2v, 1] = 1.
    bds[nbs1+nbs2v:, 0] = 0.     #  constant
    bds[nbs1+nbs2v:, 1] = 0.

    return bds

