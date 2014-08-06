import numpy.polynomial.polynomial as _Npp
import scipy.stats as _ss
import kdist as _kd
import ARlib as _arl
import warnings
import logerfc as _lfc
import commdefs as _cd
import numpy as _N
from ARcfSmplFuncs import ampAngRep, randomF, dcmpcff, betterProposal

cdef extern from "math.h":
    double exp(double)
    double sqrt(double)
    double log(double)
    double abs(double)

def ARcfSimple(N, k, smpx, q2):
    Y    = smpx[1:N, 0].reshape(N-1, 1)    #  a column vector
    X    = smpx[0:N-1, :]                  #  a column vector

    XTX  = _N.dot(X.T, X)
    iXTX = _N.linalg.inv(XTX)
    XTY  = _N.dot(X.T, Y)
    Fm   = _N.dot(iXTX, XTY)

    Sg   = q2 * iXTX
    bBdd = False
    while not bBdd:
        F0   = _N.random.multivariate_normal(Fm[:, 0], Sg, size=1)[0, :]
        bBdd, iBdd, mags, vals = _arl.ARevals(F0)

    print ampAngRep(vals)
    return F0

def ARcfSmpl(N, k, AR2lims, smpxU, smpxW, q2, R, Cs, Cn, alpR, alpC, _d, accepts=1, prior=_cd.__COMP_REF__):
    C = Cs + Cn
    #  I return F and Eigvals

    ujs     = []
    wjs     = []

    #  CONVENTIONS, DATA FORMAT
    #  x1...xN  (observations)   size N-1
    #  x{-p}...x{0}  (initial values, will be guessed)
    #  smpx
    #  ujt      depends on x_t ... x_{t-p-1}  (p-1 backstep operators)
    #  1 backstep operator operates on ujt
    #  wjt      depends on x_t ... x_{t-p-2}  (p-2 backstep operators)
    #  2 backstep operator operates on wjt
    #  
    #  smpx is N x p.  The first element has x_0...x_{-p} in it
    #  For real filter
    #  prod_{i neq j} (1 - alpi B) x_t    operates on x_t...x_{t-p+1}
    #  For imag filter
    #  prod_{i neq j,j+1} (1 - alpi B) x_t    operates on x_t...x_{t-p+2}

    ######  COMPLEX ROOTS.  Cannot directly sample the conditional posterior

    Ff  = _N.zeros((1, k-1))
    F0  = _N.zeros(2)
    F1  = _N.zeros(2)
    A   = _N.empty(2)

    Xs     = _N.empty((_d.TR, N-2, 2))
    Ys     = _N.empty((N-2, 1))
    H      = _N.empty((_d.TR, 2, 2))
    iH     = _N.empty((_d.TR, 2, 2))
    mu     = _N.empty((_d.TR, 2, 1))
    Mj     = _N.empty(_d.TR)
    Mji    = _N.empty(_d.TR)
    mj     = _N.empty(_d.TR)

    #for c in xrange(C-1, -1, -1):
    for c from C > c >= 0:
        ph0L = -1
        ph0H = 0
        if c >= Cs:     #  we don't want the noise components to be too strong
            ph0L = -0.75
        j = 2*c + 1
        p1a =  AR2lims[c, 0]
        p1b =  AR2lims[c, 1]

        # given all other roots except the jth.  This is PHI0
        jth_r1 = alpC.pop(j)
        jth_r2 = alpC.pop(j-1)

        #  print jth_r1
        #  exp(-(Y - FX)^2 / 2q^2)
        Frmj = _Npp.polyfromroots(alpR + alpC).real

        Ff[0, :]   = Frmj[::-1]
        #  Ff first element k-delay,  Prod{i neq j} (1 - alfa_i B)

        ##########  CREATE FILTERED TIMESERIES   ###########
        ##########  Frmj is k x k, smpxW is (N+2) x k ######

        #for m in xrange(_d.TR):
        for m from 0 <= m < _d.TR:
            wj  = _N.dot(Ff, smpxW[m].T).T
            wjs.append(wj)

            ####   Needed for proposal density calculation
            Ys[:, 0]    = wj[2:N, 0]
            #Ys          = Ys.reshape(N-2, 1)
            Xs[m, :, 0] = wj[1:N-1, 0]   # new old
            Xs[m, :, 1] = wj[0:N-2, 0]
            iH[m]       = _N.dot(Xs[m].T, Xs[m]) / q2[m]
            H[m]        = _N.linalg.inv(iH[m])   #  aka A
            mu[m]        = _N.dot(H[m], _N.dot(Xs[m].T, Ys))/q2[m]

        #  
        Ji  = _N.sum(iH, axis=0)
        J   = _N.linalg.inv(Ji)
        U   = _N.dot(J, _N.einsum("tij,tjk->ik", iH, mu))

        # bBdd, iBdd, mags, vals = _arl.ARevals(U)
        # print ampAngRep(vals)
        # print "::::::"

        ##########  Sample *PROPOSED* parameters 

        bSimpOK = False

        # #  If posterior valid Gaussian    q2 x H - oNZ * prH00
        if prior == _cd.__COMP_REF__:
            ph1j1, ph1j2 = _N.random.multivariate_normal(U[:, 0], J, size=1)[0, :]

            #  sampled points lie in stationary region
            if ((ph1j2 < 0) and (ph1j2 > -1)) and \
               ((ph1j1 > p1a*sqrt(-1*ph1j2)) and (ph1j1 < p1b*sqrt(-1*ph1j2))):
                A[0] = ph1j1; A[1] = ph1j2
                bSimpOK = True

        if not bSimpOK:
            iAcc = 0

            vPr1  = J[0, 0] - (J[0, 1]*J[0, 1])   / J[1, 1]   # known as Mj
            vPr2  = J[1, 1]
            svPr2 = sqrt(vPr2)     ####  VECTORIZE
            svPr1 = sqrt(vPr1)

            b2Real = (U[1, 0] + 0.25*U[0, 0]*U[0, 0] > 0)
            ######  Initialize F0
            if not b2Real:    #  complex roots
                mu1prp = U[1, 0]
                mu0prp = U[0, 0]
            else:
                mu0prp, mu1prp = betterProposal(J, Ji, U)
                
            ph0j2 = _kd.truncnorm(a=ph0L, b=ph0H, u=mu1prp, std=svPr2)
            r1    = sqrt(-1*ph0j2)
            mj0   = mu0prp + (J[0, 1] * (ph0j2 - mu1prp)) / J[1, 1]
            ph0j1 = _kd.truncnorm(a=p1a*r1, b=p1b*r1, u=mj0, std=svPr1)
            F0[0] = ph0j1; F0[1] = ph0j2

            sinceLast = 0
            while (iAcc < accepts) and (sinceLast < 50):
                #print "%(1).3f   %(2).3f    %(sl)d" % {"1" : p1a, "2" : p1b, "sl" : sinceLast}

                ##########  *CURRENT* parameters, calculate 
                ph0j1     = F0[0]
                ph0j2     = F0[1]

                ########## phj2
                #  truncated Gaussian to [-1, 1]
                ph1j2 = _kd.truncnorm(a=ph0L, b=0, u=mu1prp, std=svPr2)
                r1    = sqrt(-1*ph1j2)
                mj0= mu0prp + (J[0, 1] * (ph1j2 - mu1prp)) / J[1, 1]
                ph1j1 = _kd.truncnorm(a=p1a*r1, b=p1b*r1, u=mj0, std=svPr1)

                F1[0] = ph1j1; F1[1] = ph1j2

                lNC2 = _lfc.trncNrmNrmlz(ph0L, ph0H, mu1prp, svPr2)
                lNC1 = _lfc.trncNrmNrmlz(p1a*r1, p1b*r1, mj0, svPr1)
                Apd_ph1j2    = -(ph1j2 - mu1prp)**2 / (2*vPr2) - lNC2
                Apd_ph1j1    = -(ph1j1 - mj0)**2 / (2*vPr1) - lNC1

                #  proposal density g.  Want to sample from p
                #  accR = min(1, [p(x') g(x'->x)] / [p(x) g(x->x')])
                #####  Calculate ratio of proposal density
                #  p(ph1j1, ph1j2) = p(ph1j1 | ph1j2) p(ph1j2)

                ################
                r0    = sqrt(-1*ph0j2)
                mj1= mu0prp + (J[0, 1] * (ph0j2 - mu1prp)) / J[1, 1]

                lNC2 = _lfc.trncNrmNrmlz(-1, 0, mu1prp, svPr2)
                lNC1 = _lfc.trncNrmNrmlz(p1a*r0, p1b*r0, mj1, svPr1)
                Apd_ph0j2    = -(ph0j2 - mu1prp)**2 / (2*vPr2) - lNC2
                Apd_ph0j1    = -(ph0j1 - mj1)**2 / (2*vPr1) - lNC1
                PrpRArg  = Apd_ph0j2 + Apd_ph0j1 - (Apd_ph1j2 + Apd_ph1j1)

                #####  Ratio of likelihood
                rsd12 = -0.5*_N.dot(F1 - U[:,0], _N.dot(Ji, F1 - U[:,0]))
                rsd02 = -0.5*_N.dot(F0 - U[:,0], _N.dot(Ji, F0 - U[:,0]))
                LRArg = (rsd12 - rsd02)

                RatArg = LRArg + PrpRArg
                if RatArg > 0:
                    RatArg = 0
                accR = min(1, exp(RatArg))

                if _N.random.rand() < accR:
                    A[0] = ph1j1; A[1] = ph1j2
                    iAcc += 1
                    sinceLast = 0
                else:
                    sinceLast += 1
                    A[0] = ph0j1; A[1] = ph0j2

                F0[:]  = A[:]

        vals, vecs = _N.linalg.eig(_N.array([A, [1, 0]]))
            
        alpC.insert(j-1, vals[0])
        alpC.insert(j, vals[1])       #  vals[1] now at end

    Ff  = _N.zeros((1, k))
    ######     REAL ROOTS.  Directly sample the conditional posterior
    for j from R > j >= 0:
        # given all other roots except the jth
        jth_r = alpR.pop(j)

        Frmj = _Npp.polyfromroots(alpR + alpC).real #  Ff first element k-delay
        Ff[0, :] = Frmj[::-1]   #  Prod{i neq j} (1 - alfa_i B)

        ##########  CREATE FILTERED TIMESERIES   ###########
        ##########  Frmj is k x k, smpxU is (N+1) x k ######

        for m from 0 <= m < _d.TR:
            uj  = _N.dot(Ff, smpxU[m].T).T
            ujs.append(uj)

            ####   Needed for proposal density calculation

            Mji[m] = _N.dot(uj[0:N, 0], uj[0:N, 0]) / q2[m]
            Mj[m] = 1 / Mji[m]
            mj[m] = _N.dot(uj[1:, 0], uj[0:N, 0]) / (q2[m]*Mji[m])

        #  truncated Gaussian to [-1, 1]
        Ji = _N.sum(Mji)
        J  = 1 / Ji
        U  = J * _N.dot(Mji, mj)

        #  we only want 
        rj = _kd.truncnorm(a=-1, b=0.2, u=U, std=sqrt(J))

        alpR.append(rj)

    return _N.array(ujs), _N.array(wjs)

