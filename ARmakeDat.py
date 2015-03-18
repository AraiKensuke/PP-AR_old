import numpy.polynomial.polynomial as _Npp
from kflib import createDataAR
import numpy as _N
import pickle

class ARdat:
    rs   = None;    ths  = None
    
    TR   = None;   N  = None
    obsvd = None;   #  TR x N
    sgnls = None;    nzs   = None;
    gCn   = None; gR = None; gCs = None
    bsnm  = None;
    nzRs  = None;

    def __init__(self, bsnm, N, rs, ths, gCn, gR):
        oo   = self
        oo.bsnm  = bsnm
        oo.gCs  = len(rs)
        oo.gCn  = gCn;     oo.gR  = gR
        oo.TR  = 1;        oo.N   = N
        
        oo.sgnls    = _N.empty((oo.gCs, oo.N))
        oo.nzs     = _N.empty((oo.gR + oo.gCn, oo.N))
        oo.rs      = _N.empty(oo.gCs);        oo.ths      = _N.empty(oo.gCs);
        
        for c in xrange(oo.gCs):   #  build signal components
            oo.rs[c]    = rs[c];     oo.ths[c]    = ths[c]
            th1pi= _N.pi*ths[c]

            alfa  = _N.array([rs[c]*(_N.cos(th1pi) + 1j*_N.sin(th1pi)), 
                              rs[c]*(_N.cos(th1pi) - 1j*_N.sin(th1pi))])

            ARcoeff          = (-1*_Npp.polyfromroots(alfa)[::-1][1:]).real
            oo.sgnls[c], y = createDataAR(N, ARcoeff, 0.1, 0.1)

        for n in xrange(oo.gR):
            #AR1 = _N.array([(_N.random.rand() - 0.5)*2])
            #AR1 = _N.array([_N.random.rand()*0.])
            AR1 = _N.array([0.])
            oo.nzs[n], y = createDataAR(oo.N, AR1, 0.1, 0.1)

    def ass(self, sigAmps, nzAmps):
        oo = self
        oo.obsvd   = _N.zeros((oo.TR, oo.N))
        Is = _N.diag(sigAmps)

        oo.obsvd[0]  = _N.sum(_N.dot(Is, oo.sgnls), axis=0)
        if (oo.nzs is not None) and (nzAmps is not None):
            In = _N.diag(nzAmps)
            oo.obsvd[0] += _N.sum(_N.dot(In, oo.nzs), axis=0)

    def save(self):
        print "saving"
        oo = self
        pcklme = [oo]

        with open("%s.dump" % oo.bsnm, "wb") as f:
            pickle.dump(pcklme, f)
