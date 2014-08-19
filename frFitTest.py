from kassdirs import resFN
from tmrsclTest import timeRescaleTest, zoom
#  firing rate fit test.

#  Give me a firing rate and spikes

setname  = "frft-osc-3"
dat      = _N.loadtxt(resFN("xprbsdN.dat", dir=setname))
N        = dat.shape[0]
m        = 1
dt       = 0.001
spkts    = _N.where(dat[:, 2] == 1)[0]
fr       = dat[:, 1]/dt
#fr       = _N.ones(N) * _N.mean(fr)

if m > 1:
    dtm      = dt/m
    frm, mspkts = zoom(fr, spkts, m)
else:
    frm      = fr
    mspkts   = spkts
    dtm      = dt

x, zss, bs, bsp, bsn = timeRescaleTest(frm, mspkts, dtm)

fig = _plt.figure(figsize=(2*5, 2*4))
#fig.add_subplot(2, 2, 1)
_plt.subplot2grid((2, 2), (0, 0))
_plt.hist(zss, bins=_N.linspace(0, 1, 20))
_plt.subplot2grid((2, 2), (0, 1))
_plt.plot(x, bs,  lw=1.5, color="black")
_plt.plot(x, bsp, lw=3, color="black", ls=":")
_plt.plot(x, bsn, lw=3, color="black", ls=":")
_plt.plot(x, zss, lw=2, color="red")
_plt.subplot2grid((2, 2), (1, 0), colspan=2)
_plt.plot(dat[2000:5000, 1]/dt)
_plt.suptitle("Hz=%.1fHz" % (len(spkts)/(N*dt)))
_plt.savefig(resFN("frFit,m=%d" % m, dir=setname))
_plt.close()
