from kassdirs import resFN
import scipy.signal as _ssig
import re as _re
exf("filter.py")


#  modulation histogram.  phase @ spike
setname="080402-0-121-c3"
p = _re.compile("^\d{6}")   # starts like "exptDate-....."
m = p.match(setname)

bRealDat = True
COLS = 4

if m == None:
    bRealDat = False
    COLS = 3

dat  = _N.loadtxt(resFN("xprbsdN.dat", dir=setname))
N, cols = dat.shape

TR   = cols / COLS

dat2 = _N.zeros((N/2, cols))

for tr in xrange(TR):
    for n in xrange(N/2):
        if (dat[2*n, COLS*tr + 2] == 1) or (dat[2*n + 1, COLS*tr + 2] == 1):
            dat2[n, COLS*tr + 2]   = 1   #  we lose 1 spk.  not a big deal

    dat2[:, COLS*tr:COLS*tr+2] = dat[::2, COLS*tr:COLS*tr+2]

setnameM="%sL" % setname

if bRealDat:
    fmt = "% .3e % .3e %.3e %d " * TR
    _N.savetxt(resFN("xprbsdN.dat", dir=setnameM, create=True), dat2, fmt=fmt)
else:
    fmt = "% .3e %.3e %d " * TR
    _N.savetxt(resFN("xprbsdN.dat", dir=setnameM, create=True), dat2, fmt=fmt)
