from kassdirs import prcmpFN
import pickle as _pkl

# The integrated area of the proposal (with unconstrained normalization)
# that is within the constrained (imaginary) root area

N = 200    #  calculate on N x N grid
M = 200
p1_0 = -2.
p1_1 = 2.
p2_0 = -1.
p2_1 = 0
sg   = 0.04  #

rn   = _N.empty((N, N))
p1p2 = _N.empty((N, N))

dp_1 = 10*sg / M
dp_2 = 10*sg / M

u1s  = _N.linspace(p1_0, p1_1, N)
u2s  = _N.linspace(p2_0, p2_1, N)

for ip1 in xrange(N):
    u1 = u1s[ip1]
    for ip2 in xrange(N):
        u2 = u2s[ip2]

        #  (p1, p2)  transition from here.  The normalization is determined
        #  by starting point of transition
        p1r = _N.linspace(u1 - 5*sg, u1 + 5*sg, M)    #  range
        p2r = _N.linspace(u2 - 5*sg, u2 + 5*sg, M)    #  range

        p1v, p2v = _N.meshgrid(p1r, p2r)
            
        msk1  = ((p1v**2 + 4*p2v) < 0)
        msk2  = (p2v >= -1)
        mskA  = _N.asfarray(msk1) * _N.asfarray(msk2)

        fxy   = _N.exp(-0.5*(p1v - u1)**2/(sg*sg) - 0.5*(p2v - u2)**2/(sg*sg))
        mfxy  = mskA * fxy
        rn[ip2, ip1] = _N.sum(mfxy)
        #xy[ip2, ip1] = _N.abs(u1)

rn *= dp_1*dp_2 / (2*_N.pi*sg*sg)

_plt.imshow(rn, origin="lower")
_plt.xticks(_N.linspace(0, N-1, 5), _N.linspace(-2, 2, 5))
_plt.yticks(_N.linspace(0, N-1, 5), _N.linspace(-1, 0, 5))

fpk = open(prcmpFN("ar2IntgArea,%.3f.dat" % sg), "w")
_pkl.dump(rn, fpk)
fpk.close()
