from kassdirs import resFN, datFN

setname="071221-0-187-theta"

dat = _N.loadtxt(resFN("xprbsdN.dat", dir=setname))

tr0 = 0
tr1 = None
#tr1 = 158


COLS= 4

if tr1 is None:
    tr1 = dat.shape[1]/4


T   = 1500
trPpg = 25
pg   = 0
trOnPg = 0
fig = _plt.figure(figsize=(8, 11))   #  1 columns, 20 trials per column
_plt.ylim(-0.5, trPpg)
_plt.xlim(-100, T+100)

for tr in xrange(tr0, tr1):
    svd = False
    if ((tr - tr0) % trPpg == 0) and (tr != tr0):
        pg += 1
        _plt.savefig(resFN("rasterPerTrial,pg=%d" % pg, dir=setname))
        _plt.close()
        svd = True
        trOnPg = 0
        fig = _plt.figure(figsize=(8, 12))   #  1 columns, 20 trials per column
        _plt.ylim(-0.5, trPpg)
        _plt.xlim(-100, T+5)

    ts = _N.where(dat[0:T, 2+tr*COLS] == 1)[0]

    print ((trPpg - 1) - trOnPg)
    _plt.text(-90, (trPpg - 1) - trOnPg, "%d" % tr)
    _plt.text(T+30, (trPpg - 1) - trOnPg, "%d" % tr)
    for t in ts:
        _plt.plot([t, t], [0 + (trPpg - 1) - trOnPg, 0.3 + (trPpg - 1) - trOnPg], color="black", lw=2)
    _plt.plot([0, T], [0 + (trPpg - 1) - trOnPg - 0.05, 0 + (trPpg - 1) - trOnPg - 0.05], color="red", lw=1)

    trOnPg += 1

if svd == False:
    pg += 1
    fig.subplots_adjust(bottom=0.08, top=0.9, left=0.15, right=0.85)
    _plt.savefig(resFN("rasterPerTrial,pg=%d" % pg, dir=setname))
    _plt.close()
