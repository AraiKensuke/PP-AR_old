from kassdirs import resFN
import scipy.signal as _ssig
import re as _re
exf("filter.py")

#  modulation histogram.  phase @ spike
setname="frft-osc-3"

dat = _N.loadtxt(resFN("xprbsdN.dat", dir=setname))
N, cols = dat.shape

p = _re.compile("^\d{6}")   # starts like "exptDate-....."
m = p.match(setname)

bRealDat = True
COLS = 4

if m == None:
    bRealDat = False
    COLS = 3

TR   = cols / COLS

#fx = lpFilt(50, 60, 500, x)
#fx = lpFilt(20, 30, 500, x)

#fx = lpFilt(10, 15, 500, x)
#fx = bpFilt(15, 40, 5, 50, 500, x)  #(fpL, fpH, fsL, fsH, nyqf, y):
#fx = bpFilt(25, 55, 15, 65, 500, x)  #(fpL, fpH, fsL, fsH, nyqf, y):

wfs  = []
phs  = []

for tr in xrange(TR):
    phst  = []
    x   = dat[:, tr*COLS]
    fx = lpFilt(20, 26, 500, x)
    #fx = bpFilt(20, 40, 10, 55, 500, x)  #(fpL, fpH, fsL, fsH, nyqf, y):
    ht_x  = _ssig.hilbert(fx)
    ph_x  = _N.empty(N)
    for n in xrange(N):
        ph_x[n] = base_q4atan(ht_x[n].real, ht_x[n].imag) / (2*_N.pi)

    for i in xrange(50, N - 50):
        if dat[i, tr*COLS+(COLS-1)] == 1:
            wfs.append(ph_x[i-50:i+50])
            phs.append(ph_x[i])
            phst.append(ph_x[i])

wfsv= _N.array(wfs)
sta = _N.mean(wfsv, axis=0)

_plt.figure()
_plt.hist(phs, bins=_N.linspace(0, 1, 50), color="black")
_plt.grid()
_plt.savefig(resFN("modulationHistogram.png", dir=setname))
_plt.close()
