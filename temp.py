fx     = _N.array(_fx[kpTrl][TR0:TR1])
px     = _N.array(_px[kpTrl][TR0:TR1])

q2m = _N.mean(_N.sqrt(smp_q2[:, 800:1000]), axis=1)

PCS = 6
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

_plt.figure()
_plt.scatter([0]*PCS, lms, color="grey")
_plt.scatter([1]*PCS, hms, color="red")
for p in xrange(PCS):
    _plt.plot([0, 1], [lms[p], hms[p]], color="black", ls="--")

_plt.savefig("sct")
_plt.close()
