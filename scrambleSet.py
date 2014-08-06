exf("kflib.py")
import utilities as _U
from kassdirs import resFN

setname ="spksMedOsc-7"   #  params_XXX.py   Results/XXX/params.py
ssetname="%s_Shf" % setname   #  params_XXX.py   Results/XXX/params.py

dat = _N.loadtxt(resFN("xprbsdN.dat", dir=setname))
sdat= _U.shuffle(dat[:, 2].tolist())
dat[:, 2] = sdat

_N.savetxt(resFN("xprbsdN.dat", dir=ssetname, create=True), dat, fmt="% .4e %.4e %d")
