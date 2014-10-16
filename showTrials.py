from kassdirs import resFN, datFN

setname="slOscMT-3nb"

dat = _N.loadtxt(resFN("xprbsdN.dat", dir=setname))

tr0 = 0
tr1 = 35

COLS= 3

T   = 1000
trPpg = 10
pg   = 0
trOnPg = 0
fig = _plt.figure(figsize=(8, 10))   #  1 columns, 20 trials per column
_plt.ylim(-0.5, trPpg)
_plt.xlim(-100, T+5)

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

    _plt.text(-90, (trPpg - 1) - trOnPg, "%d" % tr)
    for t in ts:
        _plt.plot([t, t], [0 + (trPpg - 1) - trOnPg, 0.3 + (trPpg - 1) - trOnPg], color="black", lw=2)
    
    trOnPg += 1

if svd == False:
    pg += 1
    _plt.savefig(resFN("rasterPerTrial,pg=%d" % pg, dir=setname))
    _plt.close()
