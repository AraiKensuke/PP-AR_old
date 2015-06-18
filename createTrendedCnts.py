from kassdirs import resFN, datFN
from shutil import copyfile
from utildirs import setFN
import utilities as _U
import commdefs as _cd
import numpy as _N
import kflib as _kfl
import matplotlib.pyplot as _plt

####   
#  _N.sin(t_orw)    #  t is an offseted random walk
setname="oscCnts-"   #  params_XXX.py   Results/XXX/params.py
model = _cd.__BNML__
rn    = None
rnM   = None
J     = 1    # number of states
m     = None
pM    = None
TR    = 1

###  
dt    = 0.001     #  need dt for timescale of modulation
useSines = False
f0       = None
Bf       = None;     Ba       = None;
dSA      = None;    dSF     = None

###
useAR    = False
ARcf     = None

N        = None
amp      = None

def create(setname, env_dirname=None, basefn="cnt_data", trend=None, dontcopy=False):
    global m, rn
    if not dontcopy:
        copyfile("%s.py" % setname, "%(s)s/%(s)s.py" % {"s" : setname, "to" : setFN("%s.py" % setname, dir=setname, create=True)})

    #global model, rn, p, dt, useSines, useAR, f0, Bf, Ba, dSA, dSF, ARcf, N
    #  mean is (1-pL)*r/pL for low spike counts

    #  Compared to step size <dt> of linear increase, we control the fluctuation
    #  in step size to make noisy sin

    COLS= 3
    cts = _N.empty(N)
    data= _N.empty((N, COLS*TR))   #  x, state, spkct x

    #  mixture distribution
    bMix = False
    if ((rnM is not None) and (m is None)) or \
       ((rnM is None) and (m is not None)):
        print "must set both rnM and m for mixture"
        exit()
    if (rnM is not None) and (m is not None):
        if len(rnM) != len(m):
            print "rnM and m length must be same for mixture"
            exit()
        m /= _N.sum(m)
        J = len(m)
        bMix = True
        uM = _N.log(pM / (1 - pM))
    if not bMix:
        u  = _N.log(p / (1 - p))

    for tr in xrange(TR):
        if trend is None:    #  generate x
            if useSines:
                _x = _kfl.createFlucOsc(f0, _N.zeros(1), N, dt, 1, Bf=Bf, Ba=Ba, smoothKer=5, dSF=dSF, dSA=dSA)
                x  = amp*(_x[0] / _N.std(_x[0]))
            else:
                trim = 100
                _x = _kfl.createDataAR(N+trim, ARcf, 0.1, 0.1, trim=trim)
                x  = amp*(_x[0] / _N.std(_x[0]))
                #x  = amp*(_x / _N.std(_x))
        else:
            if len(trend.shape) == 2:
                x = trend[tr]
            else:
                x = trend

        for t in xrange(N):
            st = 0
            if bMix:
                exM = _N.exp(uM + x[t])
                tot = 0
                rnd =_N.random.rand() 
                for i in xrange(J):
                    tot += m[i]
                    if (rnd >= tot - m[i]) and (rnd <= tot):
                        rn = rnM[i]
                        ex = exM[i]
                        st = i
            else:
                ex = _N.exp(u + x[t])

            if model == _cd.__BNML__:
                cts[t] = _N.random.binomial(rn, ex / (1 + ex))
            else:
                cts[t] = _N.random.negative_binomial(rn, 1-(ex / (1 + ex)))
            data[t, tr*COLS] = x[t]
            data[t, tr*COLS+1] = st
            data[t, tr*COLS+2]  = cts[t]

    fmtstr = "% .5f %d %d " * TR
    _N.savetxt(resFN("%s.dat" % basefn, dir=setname, create=True, env_dirname=env_dirname), data, fmt=fmtstr)

    fig =_plt.figure(figsize=(13, 3.5*2))
    _plt.subplot2grid((2, 8), (0, 0), colspan=4)
    _plt.plot(cts, color="black")
    _plt.subplot2grid((2, 8), (1, 0), colspan=4)
    _plt.plot(x, color="black")
    _plt.subplot2grid((2, 8), (0, 5), rowspan=3, colspan=4)
    _plt.hist(cts, bins=_N.linspace(0, 1.05*max(cts), 40), color="black")
    _plt.savefig(resFN("cts_%s.png" % basefn, dir=setname, env_dirname=env_dirname, create=True))
    _plt.close()
