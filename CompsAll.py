import pickle as _pkl
from kassdirs import resFN
from ARcfSmplFuncs import dcmpcff
from mcmcARpPlot import plotWFandSpks

TR = 5
ddN= _d.N
rt = _N.empty((TR, burn+NMC, ddN+2, R))    #  real components   N = ddN
zt = _N.empty((TR, burn+NMC, ddN+2, C))    #  imag components 

#  I can't use allalfas that are done during Gibbs sampling.
i  = 0

rsds = _N.empty(burn+NMC)
for it in xrange(1, burn + NMC):
    b, c = dcmpcff(alfa=allalfas[it])

    for tr in xrange(TR):
        for r in xrange(R):
            rt[tr, it, :, r] = b[r] * uts[tr, it, r, :]

        for z in xrange(C):
            #print "z   %d" % z
            cf1 = 2*c[2*z].real
            gam = allalfas[it, R+2*z]
            #cf2 = 2*(c[2*z].real*gam.real + c[2*z].imag*gam.imag)
            cf2 = 2*(c[2*z].real*gam.real + c[2*z].imag*gam.imag)
            #print "%(1).3f    %(2).3f" % {"1": cf1, "2" : cf2}
            for n in xrange(1, ddN+3):
                zt[tr, it, n-1, z] = cf1*wts[tr, it, z, n] - cf2*wts[tr, it, z, n-1]

        sm = _N.sum(rt[0,it, 1:, :], axis=1) + _N.sum(zt[0, it, 1:, :], axis=1)
        rsds[it-burn]= _N.std(Bsmpx[tr, it, 2:] - sm)


    i += 1
