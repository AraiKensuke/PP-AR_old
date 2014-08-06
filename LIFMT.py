import numpy.polynomial.polynomial as _Npp
from mcmcARpPlot import plotWFandSpks
from kassdirs import resFN
import utilities as _U
exf("kflib.py")

setname="LIF-080402-0-121-LF-LM-1"
#  dV = -V/(RC) + I/C
#  dV = -V/tau + I/C  = -V/tau + eps I

N   = 1000
TR  = 60


tau = .2      #  ms.  time constant = 1/RC
dt  = 0.001     #  1ms
bcksig =210   #  background firing 
I   = 14

r    = 0.997
th   = 0.013
th1pi= _N.pi*th
errH  = 0.004
errL  = 0.00001
obsnz= 1e-1

alfa  = _N.array([r*(_N.cos(th1pi) + 1j*_N.sin(th1pi)), 
                  r*(_N.cos(th1pi) - 1j*_N.sin(th1pi))])
ARcoeff          = (-1*_Npp.polyfromroots(alfa)[::-1][1:]).real

rst   = 0
thr   = 1

V     = _N.empty(N)
V[0]  = _N.random.randn()
dV    = _N.empty(N)
dN    = _N.zeros(N)

xprbsdN= _N.empty((N, 3*TR))
isis  = []

lowQpc= 0.1
lowQs = []

spksPT= _N.empty(TR)
for tr in xrange(TR):
    eps = bcksig*_N.random.randn(N)   # time series
    err = errH
    if _N.random.rand() < lowQpc:
        err = errL
        lowQs.append(tr)
    sTs   = []
    x, y= createDataAR(N + 500, ARcoeff, err, obsnz, trim=0)
    dN[:] = 0
    for n in xrange(N-1):
        dV[n] = -V[n] / tau + eps[n] + I + x[n + 500]
        V[n+1] = V[n] + dV[n]*dt
            
        if V[n+1] > thr:
            V[n + 1] = rst
            dN[n] = 1
            sTs.append(n)
    spksPT[tr] = len(sTs)
        
    xprbsdN[:, tr*3]     = x[500:]
    xprbsdN[:, tr*3 + 1] = V
    xprbsdN[:, tr*3 + 2] = dN
    isis.extend(_U.toISI([sTs])[0])

fmt = "% .2e %.3f %d " * TR
_N.savetxt(resFN("xprbsdN.dat", dir=setname, create=True), xprbsdN, fmt=fmt)

if TR == 1:
    plotWFandSpks(N-1, dN, [x[500:]], sTitle="AR2 freq %(f).1fHz  num spks %(d).0f   spk Hz %(spkf).1fHz" % {"f" : (500*th), "d" : _N.sum(dN), "spkf" : (_N.sum(dN) / (N*0.001))}, sFilename=resFN("generative-det", dir=setname), intv=[2000, 4000])
    plotWFandSpks(N-1, dN, [x[500:]], sTitle="AR2 freq %(f).1fHz  num spks %(d).0f   spk Hz %(spkf).1fHz" % {"f" : (500*th), "d" : _N.sum(dN), "spkf" : (_N.sum(dN) / (N*0.001))}, sFilename=resFN("generative", dir=setname))
else:
    plotWFandSpks(N-1, dN, [x[500:]], sTitle="AR2 freq %(f).1fHz  num spks %(d).0f   spk Hz %(spkf).1fHz" % {"f" : (500*th), "d" : _N.sum(dN), "spkf" : (_N.sum(dN) / (N*0.001))}, sFilename=resFN("generative", dir=setname))


cv   = _N.std(isis) / _N.mean(isis)
fig  = _plt.figure(figsize=(7, 3.5))
_plt.hist(isis, bins=range(0, 50), color="black")
_plt.grid()
_plt.suptitle("ISI cv %.2f" % cv)
_plt.savefig(resFN("ISIhist", dir=setname))
_plt.close()

fp = open(resFN("params.py", dir=setname), "w")
fp.write("dt=%.3f\n"  % dt)
fp.write("#  AR params\n")
fp.write("r=%.3f\n"  % r)
fp.write("th=%.3f\n"  % th)
fp.write("err=%.3f\n"  % err)
fp.write("#  LIF params\n")
fp.write("tau=%.3f\n"  % tau)
fp.write("bcksig=%.3f\n"  % bcksig)
fp.write("I=%.3f\n"  % I)
fp.write("rst=%.3f\n"  % rst)
fp.write("lowQs=%s\n" % str(lowQs))
fp.close()
