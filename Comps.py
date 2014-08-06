import pickle as _pkl
from kassdirs import resFN
from ARcfSmplFuncs import dcmpcff
from mcmcARpPlot import plotWFandSpks

lowF = _N.empty((NMC, C))
lowA = _N.empty((NMC, C))

for it in xrange(burn, burn + NMC):
    lowF[it-burn] = fs[it, ranks[it, :]]
    lowA[it-burn] = amps[it, ranks[it, :]]

nComps = 3

fig = _plt.figure(figsize=(3*5, 4*nComps))
for cmp in xrange(nComps):
    #fig.add_subplot(nComps, 3, 3*cmp + 1)
    _plt.subplot2grid((nComps+1, 3), (cmp, 0))
    _plt.hist(lowF[:, cmp], bins=_N.linspace(0, 0.3, 151), color="black")
    _plt.subplot2grid((nComps+1, 3), (cmp, 1))
    _plt.hist(lowA[:, cmp], bins=_N.linspace(0, 1, 101), color="black")
    _plt.subplot2grid((nComps+1, 3), (cmp, 2))
    _plt.scatter(lowF[:, cmp], lowA[:, cmp], color="black", s=5)
    _plt.xlim(0, 0.3)
    _plt.ylim(0, 1)
_plt.subplot2grid((nComps+1, 3), (nComps, 1))
_plt.scatter(lowF.flatten(), lowA.flatten(), color="black", s=5)
_plt.xlim(0, 1)
_plt.ylim(0, 1)

_plt.savefig(resFN("%s,lowestRoot" % baseFN, dir=setdir))
_plt.close()

medF = _N.median(lowF[:, 0])

use = []
for it in xrange(burn, burn + NMC):
    f = fs[it, ranks[it]][0] 
    a = amps[it, ranks[it]][0] 
    if ((f > 0.018) and (f < 0.02)) and (a > 0.65):
        use.append(it)

tr = 0

TR = 1
ddN= _d.N
luse = len(use)
rt = _N.empty((TR, luse, ddN+2, R))    #  real components   N = ddN
zt = _N.empty((TR, luse, ddN+2, C))    #  imag components 

#  I can't use allalfas that are done during Gibbs sampling.
i  = 0
lowest = _N.empty((TR, luse, ddN+2))
for it in use:
    b, c = dcmpcff(alfa=allalfas[it])

    for r in xrange(R):
        rt[tr, i, :, r] = b[r] * uts[tr, it, r, :]

    for z in xrange(C):
        #print "z   %d" % z
        cf1 = 2*c[2*z].real
        gam = allalfas[it, R+2*z]
        #cf2 = 2*(c[2*z].real*gam.real + c[2*z].imag*gam.imag)
        cf2 = 2*(c[2*z].real*gam.real + c[2*z].imag*gam.imag)
        #print "%(1).3f    %(2).3f" % {"1": cf1, "2" : cf2}
        for n in xrange(1, ddN+3):
            zt[tr, i, n-1, z] = cf1*wts[tr, it, z, n] - cf2*wts[tr, it, z, n-1]
    lowest[tr, i] = zt[tr, i, :, ranks[it, 0]]
    i += 1

ztm = _N.mean(lowest[0], axis=0)

Fall = _N.empty((NMC, k))
for it in xrange(NMC):
    Fall[it, :] = -_Npp.polyfromroots(allalfas[1]).real[::-1][1:]
avgF = _N.mean(Fall, axis=0)
bBdd, iBdd, mags, vals = _arl.ARevals(avgF)
aAR = ampAngRep(vals, sp=" | ")
pc, pv = _ss.pearsonr(ztm[1:], x[0, :])
sTitle = "%(aAR)s\n%(pc).3f" % {"pc" : pc, "aAR" : aAR}


A = _N.std(ztm[1:])
B = _N.std(x[0, :])
mag = B / A

plotWFandSpks(N, y[0], [x[0, :], mag*ztm[1:]], sTitle=sTitle, sFilename=resFN("%s,spks1" % baseFN, dir=setdir), intv=[0, 1000])
plotWFandSpks(N, y[0], [x[0, :], mag*ztm[1:]], sTitle=sTitle, sFilename=resFN("%s,spks2" % baseFN, dir=setdir), intv=[1000, 2000])

fig = _plt.figure(figsize=(3*5, 4))
fig.add_subplot(1, 3, 1)
_plt.hist(fs[1:].flatten(), bins=_N.linspace(0, 1, 101), color="black")
_plt.xticks(_N.linspace(0, 1, 11))
_plt.grid()
fig.add_subplot(1, 3, 2)
_plt.hist(fs[1:].flatten(), bins=_N.linspace(0, 1, 101), color="black")
_plt.xlim(0, 0.2)
_plt.xticks(_N.linspace(0, 0.2, 3))
_plt.grid()
fig.add_subplot(1, 3, 3)
_plt.scatter(fs[1:].flatten(), amps[1:].flatten(), color="black", s=6)
_plt.xticks(_N.linspace(0, 1, 11))
_plt.yticks(_N.linspace(0, 1, 11))
_plt.grid()
_plt.savefig(resFN("%s_fs" % baseFN, dir=setdir), intv=[1000, 2000])
_plt.close()

####  The real roots
clrs=["black", "blue", "green", "red", "orange", "gray", "brown", "purple", "pink", "cyan"]
fig = _plt.figure(figsize=(2*8, 3.5))
for r in xrange(R):
    fig.add_subplot(1, R, r+1)
    _plt.plot(allalfas[1:, r].real, color=clrs[r], lw=1.5)
    _plt.ylim(-1, 1)
    _plt.grid()

fig.subplots_adjust(left=0.06, right=0.94, top=0.85, bottom=0.15)
_plt.savefig(resFN("%s_rrs" % baseFN, dir=setdir))
_plt.close()
