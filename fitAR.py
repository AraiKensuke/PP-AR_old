import ARlib as _arl
import numpy.polynomial.polynomial as _Npp
from ARcfSmpl import ARcfSmpl
from ARcfSmplFuncs import ampAngRep, buildLims, FfromLims, dcmpcff
import logerfc as _lfc
import kfardat as _kfardat
import scipy.stats as _ss
import commdefs as _cd

exf("kflib.py")
_lfc.init()

TR    = 1
N    = 8002   #  +1 for observation noise accounted for in AR model
burn = 1000
NMC  = 1000

q2   = 0.01

#####  Start from roots


r1 = 0.95
th1= _N.pi*0.09
r2 = 0.95
th2= _N.pi*0.33

alfaGN  = _N.array([r1*(_N.cos(th1) + 1j*_N.sin(th1)), 
                    r1*(_N.cos(th1) - 1j*_N.sin(th1)),
                    r2*(_N.cos(th2) + 1j*_N.sin(th2)), 
                    r2*(_N.cos(th2) - 1j*_N.sin(th2))])

Ftr          = (-1*_Npp.polyfromroots(alfaGN)[::-1][1:]).real

ampAngRep(alfaGN)

#####  From AR coeff
#Ftr  = _N.array([0.65, 0.25, 0.1, 0.03, 0.01, -0.1])   #  oscillatory
#Ftr   = _N.array([0.7, 0.2])
#bBdd, iBdd, mags, roots = _arl.ARroots(Ftr)
#alfa = 1 / roots
#print alfa

#  Treat x[0:k-1]   as unobserved initial values
#        x[k:k+N-1] as observed data values
#  data size N
#  exXU  data size N+1
#  exXW  data size N+2

fSigMax=100
freq_lims   = [[0.1, fSigMax], [0.1, fSigMax]]
#freq_lims   = [[0.1, fSigMax]]
Cn          = 2    #  # of noize components
Cs          = len(freq_lims)
C     = Cn + Cs
radians     = buildLims(Cn, freq_lims, nzLimL=1.)
AR2lims     = 2*_N.cos(radians)


R  = 3
k  = 2*C + R     #  k 
#x, y = createDataAR(N+k, Ftr, q2, 10)
x = _N.random.randn(N+k)

#  Initialize alfa to some other value
F_alfa_rep  = randomF(R, Cs+Cn, equalF=True)   #  init F_alfa_rep
#F_alfa_rep = alfaGN
alpR = F_alfa_rep[0:R].tolist()
alpC = F_alfa_rep[R:].tolist()

exX   = _N.zeros((1, N+2, k))
Y     = x[k:N+k]

exX[0, 0, 0:k-2] = x[0:k-2][::-1]
exX[0, 1, 0:k-1] = x[0:k-1][::-1]

for t in xrange(2, N+2):
    #  exX[t, 0] exX[t, 1] exX[t, 2]    (newest to oldest)
    exX[0, t, :] =    x[t-2:t+k-2][::-1]   #  exX and x seem displace by 1 (why?)

Fs         = _N.empty((burn + NMC, k))
rmjs       = _N.empty((burn + NMC, k, N))

#  q2  --  Inverse Gamma prior
a_q2         = 1e-1
B_q2         = 1e-6

#  Now initialize values
q20 = 0.01
q2s= _N.empty(burn + NMC)
q2          = _N.ones(TR)*q20
allalfas= _N.empty(((burn + NMC), k), dtype=_N.complex)


_d = _kfardat.KFARGauObsDat(TR, N, k)
for tr in xrange(burn + NMC):
    print "------TR   %d" % tr
    ARcfSmpl(N, k, AR2lims, exX[:, 1:, :], exX[:, :, 0:k-1], q2, R, Cs, Cn, alpR, alpC, _d, accepts=100, prior=_cd.__COMP_REF__)

    new_alfa = alpR + alpC
    print ampAngRep(new_alfa)
    allalfas[tr, :] = new_alfa
    Frep          = (-1*_Npp.polyfromroots(new_alfa)[::-1][1:]).real
    Fs[tr, :]     = Frep
    a = a_q2 + 0.5*(N+1)  #  N + 1 - 1
    rsd = _N.dot((Y - _N.dot(exX[0, 2:], Frep)).T, (Y - _N.dot(exX[0, 2:], Frep)))
    BB = B_q2 + 0.5 * rsd
    q2[0] = _ss.invgamma.rvs(a, scale=BB)
    q2s[tr] = q2[0]

