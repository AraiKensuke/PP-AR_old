import logerfc as _lfc
import pickle as _pkl

# The integrated area of the proposal that is within the constrained 
# (imaginary) root area

_lfc.init()
N = 200    #  calculate on N x N grid
p1_0 = -2.
p1_1 = 2.
p2_0 = -1.
p2_1 = 0
sg   = 0.04  #

A    = _N.empty((N, N))     #  area underneath proposal

u1s  = _N.linspace(p1_0, p1_1, N)
u2s  = _N.linspace(p2_0, p2_1, N)

for ip2 in xrange(N):
    u2 = u2s[ip2]
    lNC2 = _lfc.trncNrmNrmlz(-1, 0, u2, sg)
    r1    = _N.sqrt(-1*u2)

    for ip1 in xrange(N):
        u1 = u1s[ip1]

        lNC1 = _lfc.trncNrmNrmlz(-2*r1, 2*r1, u1, sg)
        A[ip2, ip1] = _N.exp(lNC1 + lNC2)
A /= 2*_N.pi*sg*sg

_plt.imshow(A, origin="lower")
_plt.xticks(_N.linspace(0, N-1, 5), _N.linspace(-2, 2, 5))
_plt.yticks(_N.linspace(0, N-1, 5), _N.linspace(-1, 0, 5))
