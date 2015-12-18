from kassdirs import resFN, datFN
from shutil import copyfile
from utildirs import setFN
import utilities as _U
import commdefs as _cd
import numpy as _N
import kflib as _kfl
import matplotlib.pyplot as _plt
import mcmcFigs as mF

####   
#  _N.sin(t_orw)    #  t is an offseted random walk
setname="oscCnts-"   #  params_XXX.py   Results/XXX/params.py
model = _cd.__BNML__
rn    = None
J     = 1    # number of states
W     = 1    # number of windows
m     = None
p    = None
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

def create(setname, env_dirname=None, basefn="cnt_data", trend=None, dontcopy=True, epspng="png"):
    global m, rn, p, W, J, u, N, amp, model
    if not dontcopy:
        copyfile("%s.py" % setname, "%(s)s/%(s)s.py" % {"s" : setname, "to" : setFN("%s.py" % setname, dir=setname, create=True)})

    #global model, rn, p, dt, useSines, useAR, f0, Bf, Ba, dSA, dSF, ARcf, N
    #  mean is (1-pL)*r/pL for low spike counts

    #  Compared to step size <dt> of linear increase, we control the fluctuation
    #  in step size to make noisy sin

    #  mixture distribution
    bMix = False
    #  rnM, pM, model need to have same shape

    if (type(model) == _N.ndarray) and (type(rn) == _N.ndarray) and (type(p) == _N.ndarray) and (m is not None):
        if (model.shape == rn.shape) and (rn.shape == p.shape):
            bMix = True
        else:
            print "rn and m length must be same for mixture"
            exit()
        m /= _N.sum(m)
        print model.shape
        if len(model.shape) == 1:
            W = 1
            J = model.shape[0]
        else:   
            W, J = model.shape
        model = model.reshape(W, J)
        rn    = rn.reshape(W, J)
        p     = p.reshape(W, J)

        print p
        u = _N.log(p / (1 - p))
    if not bMix:   # windows not supported in non-mix data
        u  = _N.log(p / (1 - p))
        W  = 1

    COLS= 2+W
    cts = _N.empty((N, W))
    data= _N.empty((N, COLS*TR))   #  x, state, spkct x

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
            x *= amp

        for t in xrange(N):
            st = 0
            if bMix:
                ex = _N.exp(u + x[t])


                tot = 0
                rnd =_N.random.rand()
                for j in xrange(J):
                    tot += m[j]

                    for w in xrange(W):
                        if (rnd >= tot - m[j]) and (rnd <= tot):
                            rnwj = rn[w, j]
                            exwj = ex[w, j]
                            st = j
                            mdlwj = model[w, j]

                            if mdlwj == _cd.__BNML__:
                                cts[t, w] = _N.random.binomial(rnwj, exwj / (1 + exwj))
                            else:
                                cts[t, w] = _N.random.negative_binomial(rnwj, 1-(exwj / (1 + exwj)))
            else:
                ex = _N.exp(u + x[t])
                mdl = model

                if mdl == _cd.__BNML__:
                    cts[t] = _N.random.binomial(rn, ex / (1 + ex))
                else:
                    cts[t] = _N.random.negative_binomial(rn, 1-(ex / (1 + ex)))
            data[t, tr*COLS] = x[t]
            data[t, tr*COLS+1] = st
            for w in xrange(W):
                data[t, tr*COLS+2+w]  = cts[t, w]

    print cts
    ctstr = ""
    for w in xrange(W):
        ctstr += "%d "

    fmtstr = ("% .5f %d " + ctstr) * TR

    #return fmtstr, data, cts

    _N.savetxt(resFN("%s.dat" % basefn, dir=setname, create=True, env_dirname=env_dirname), data, fmt=fmtstr)

    for w in xrange(W):
        fig =_plt.figure(figsize=(13, 3.5*2))
        ax = _plt.subplot2grid((2, 8), (0, 0), colspan=4)
        _plt.plot(cts[:, w], color="black")
        mF.arbitaryAxes(ax, axesVis=[True, True, False, False], x_tick_positions="bottom", y_tick_positions="left")
        mF.setTicksAndLims(xlabel="trials", ylabel="spk counts", tickFS=18, labelFS=20)

        ax = _plt.subplot2grid((2, 8), (1, 0), colspan=4)
        _plt.plot(x, color="black")
        mF.arbitaryAxes(ax, axesVis=[True, True, False, False], x_tick_positions="bottom", y_tick_positions="left")
        mF.setTicksAndLims(xlabel="trials", ylabel="trend", tickFS=18, labelFS=20, yticks=[])
        ax = _plt.subplot2grid((2, 8), (0, 5), rowspan=3, colspan=4)
        _plt.hist(cts[:, w], bins=_N.linspace(0, 1.05*max(cts[:, w]), int(1.1*max(cts[:, w]))), color="black")
        mF.arbitaryAxes(ax, axesVis=[True, True, False, False], x_tick_positions="bottom", y_tick_positions="left")
        mF.setTicksAndLims(xlabel="counts", ylabel="freq.", tickFS=18, labelFS=20)
        fnSF = ("_%d" % w) if (W > 1) else ""
        fig.subplots_adjust(left=0.11, bottom=0.15, top=0.93, right=0.93, wspace=0.2, hspace=0.2)
        _plt.savefig(resFN("cts_%(fn)s%(sf)s.%(ep)s" % {"fn" : basefn, "sf" : fnSF, "ep" : epspng}, dir=setname, env_dirname=env_dirname, create=True), transparent=True)

        _plt.close()

