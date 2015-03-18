from kassdirs import resFN
from tmrsclTest import timeRescaleTest, zoom
#  firing rate fit test.

#  Give me a firing rate and spikes

setname  = "fig1"
dat      = _N.loadtxt(resFN("xprbsdN.dat", dir=setname))
N        = dat.shape[0]
M        = dat.shape[1]/3
m        = 100            # multiply time, alleviate bin size issue
dt       = 0.001
Lspkts   = []
for tr in xrange(M):
    Lspkts.append(_N.where(dat[:, 2+3*tr] == 1)[0])
fr       = dat[:, 1::3]/dt
fr       = fr.T      #  In general, 1st index is trial number

if m > 1:
    dtm      = dt/m
    frm, mspkts = zoom(fr, Lspkts, m)
else:
    frm      = fr
    mspkts   = Lspkts
    dtm      = dt

x, zss, bs, bsp, bsn = timeRescaleTest(frm, mspkts, dtm)

fig = _plt.figure(figsize=(2*9, 8))
_plt.subplot2grid((1, 2), (0, 0))
_plt.hist(zss, bins=_N.linspace(0, 1, 20))
_plt.subplot2grid((1, 2), (0, 1))
_plt.plot(x, bs,  lw=1.5, color="black")
_plt.plot(x, bsp, lw=3, color="black", ls=":")
_plt.plot(x, bsn, lw=3, color="black", ls=":")
_plt.plot(x, zss, lw=2, color="red")
_plt.ylim(-0.05, 1.05)
#_plt.suptitle("Hz=%.1fHz" % (len(spkts)/(N*dt)))
_plt.savefig(resFN("frFit,m=%d" % m, dir=setname))
_plt.close()
