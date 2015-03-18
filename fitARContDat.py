import numpy.polynomial.polynomial as _Npp
import scipy.stats as _ss
from ARcfSmplFuncs import ampAngRep, buildLims, FfromLims, dcmpcff, initF
import logerfc as _lfc
import commdefs as _cd
from ARcfSmpl import ARcfSmpl, FilteredTimeseries
import numpy as _N

class fARCD:
    use_prior     = _cd.__COMP_REF__
    ARord         = None
    fSigMax       = 500.    #  fixed parameters
    freq_lims     = None  #[[1 / .85, fSigMax]]
    sig_ph0L      = None;    sig_ph0H      = None
    Cn            = None;    Cs = None
    lfs           = None
    #  q2  --  Inverse Gamma prior
    a_q2          = 1e-1;          B_q2         = 1e-6
    rt            = None;     zt   = None
    N             = None
    F0            = None;     smpx = None

    def __init__(self, Cs, Cn, R, ard):
        oo         = self
        #  guessing AR coefficients of this form
        oo.Cn      = Cn   # noise components
        oo.Cs      = Cs
        oo.R       = R

        oo.k       = 2*(Cn + Cs) + R
        oo.TR      = 1
        oo.lfc         = _lfc.logerfc()
        oo.N           = ard.N - oo.k
        oo.smpx        = _N.zeros((oo.TR, oo.N+2, oo.k))

        oo.freq_lims     = []
        for n in xrange(oo.Cs):
            oo.freq_lims.append([1 / .85, oo.fSigMax])

        for n in xrange(oo.N):
            #smpx[0, n+2] = ard.obsvd[0, n:n+oo.k]
            oo.smpx[0, n+2] = ard.obsvd[0, n+oo.k:n:-1]
        for m in xrange(oo.TR):
            oo.smpx[0, 1, 0:oo.k-1]   = oo.smpx[0, 2, 1:]   # 0:oo.k   now to k steps back in time
            oo.smpx[0, 0, 0:oo.k-2]   = oo.smpx[0, 2, 2:]


    def ARsmpl(self, ITER, sig_ph0L=-1, sig_ph0H=-0.94, arord=_cd.__NF__):
        oo           = self
        oo.ARord     = arord
        radians      = buildLims(oo.Cn, oo.freq_lims, nzLimL=1.)
        AR2lims      = 2*_N.cos(radians)

        oo.F_alfa_rep  = initF(oo.R, oo.Cs, oo.Cn).tolist()   #  init F_alfa_rep
        alpR        = oo.F_alfa_rep[0:oo.R]
        alpC        = oo.F_alfa_rep[oo.R:]
        q2          = _N.array([0.01])

        oo.fs           = _N.empty((ITER, oo.Cn + oo.Cs))
        oo.amps         = _N.empty((ITER, oo.Cn + oo.Cs))
        oo.q2s          = _N.empty((ITER, 1))

        for it in xrange(ITER):
            print "-------  %d" % it
            ARcfSmpl(oo.lfc, oo.N, oo.k, AR2lims, oo.smpx[:, 1:, 0:oo.k], oo.smpx[:, :, 0:oo.k-1], q2, oo.R, oo.Cs, oo.Cn, alpR, alpC, oo.TR, prior=oo.use_prior, accepts=30, aro=oo.ARord, sig_ph0L=sig_ph0L, sig_ph0H=sig_ph0H)  
            oo.F_alfa_rep = alpR + alpC   #  new constructed
            prt, rank, f, amp = ampAngRep(oo.F_alfa_rep, f_order=True)
            print prt

            for m in xrange(oo.TR):
                oo.amps[it, :]  = amp
                oo.fs[it, :]    = f

            oo.F0   = (-1*_Npp.polyfromroots(oo.F_alfa_rep)[::-1].real)[1:]


            a2 = oo.a_q2 + 0.5*(oo.TR*oo.N + 2)  #  N + 1 - 1
            BB2 = oo.B_q2
            for m in xrange(oo.TR):
                #   set x00 
                rsd_stp = oo.smpx[m, 3:,0] - _N.dot(oo.smpx[m, 2:-1], oo.F0).T
                BB2 += 0.5 * _N.dot(rsd_stp, rsd_stp.T)
            q2[0] = _ss.invgamma.rvs(a2, scale=BB2)
            oo.q2s[it] = q2


        oo.ut, oo.wt = FilteredTimeseries(oo.N, oo.k, oo.smpx[:, 1:, 0:oo.k], oo.smpx[:, :, 0:oo.k-1], q2, oo.R, oo.Cs, oo.Cn, alpR, alpC, oo.TR)
        oo.getComponents()

    def getComponents(self):
        oo    = self
        TR    = oo.TR
        R     = oo.R
        C     = oo.Cn + oo.Cs
        ddN   = oo.N-1

        oo.rt = _N.empty((TR, ddN+2, R))    #  real components   N = ddN
        oo.zt = _N.empty((TR, ddN+2, C))    #  imag components 

        for tr in xrange(TR):
            b, c = dcmpcff(alfa=_N.array(oo.F_alfa_rep))

            for r in xrange(R):
                oo.rt[tr, :, r] = b[r] * oo.ut[tr, r, :, 0]

            for z in xrange(C):
                #print "z   %d" % z
                cf1 = 2*c[2*z].real
                gam = oo.F_alfa_rep[R+2*z]
                cf2 = 2*(c[2*z].real*gam.real + c[2*z].imag*gam.imag)
                oo.zt[tr, 0:ddN+2, z] = cf1*oo.wt[tr, z, 1:ddN+3, 0] - cf2*oo.wt[tr, z, 0:ddN+2, 0]
