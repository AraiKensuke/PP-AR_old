fx     = _N.array(_fx[kpTrl][TR0:TR1])
px     = _N.array(_px[kpTrl][TR0:TR1])

xsmp = _N.mean(Bsmpx[:, 800:1000, 2:], axis=1)
q2m  = _N.std(xsmp, axis=1)

PCS = 1
qtr = TR/PCS

lms = []
hms = []
for i in xrange(PCS):
    lows = []
    highs= []

    qtr0 = i*qtr
    qtr1 = (i+1)*qtr

    mdn = _N.median(q2m[qtr0:qtr1])
    for j in xrange(qtr0, qtr1):
        if q2m[j] < mdn:
            lows.append(_N.std(fx[j]))
        else:
            highs.append(_N.std(fx[j]))

    print "%(l).3f   %(h).3f" % {"l" : _N.mean(lows), "h" : _N.mean(highs)}



    lms.append(_N.mean(lows))
    hms.append(_N.mean(highs))

fig = _plt.figure()
_plt.scatter([0]*PCS, lms, color="grey", s=40)
_plt.scatter([1]*PCS, hms, color="red", s=40)
_plt.xticks([0, 1], ["smaller", "larger"], fontsize=20)
_plt.yticks(fontsize=20)
_plt.xlim(-0.5, 1.5)
_plt.xlabel("amplitude inferred osc.", fontsize=24)
_plt.ylabel("amplitude theta filtered LFP", fontsize=24)
for p in xrange(PCS):
    _plt.plot([0, 1], [lms[p], hms[p]], color="black", ls="--")
fig.subplots_adjust(bottom=0.15, left=0.15)
_plt.savefig("ssct-%d" % PCS)
_plt.close()
