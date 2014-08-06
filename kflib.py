#import kfpp as _kfpp
from ARcfSmplFuncs import ampAngRep, dcmpcff, betterProposal

def loadL2(setname):
    fn = resFN("lambda2.dat", dir=setname)

    if os.access(fn, os.F_OK):
        return _N.loadtxt(fn)
    return None

def runNotes(setname, ID_q2, TR0, TR1):
    fp = open(resFN("notes.txt", dir=setname), "w")
    fp.write("ID_q2=%s\n" % str(ID_q2))
    fp.write("TR0=%d\n" % TR0)
    fp.write("TR1=%d\n" % TR1)
    fp.close()

    #  ID_q2
    #  Trials using
    
def covMat(dim, cc=1):
    cm   = _N.empty((dim, dim))
    stds = _N.abs(_N.random.randn(dim))
    pij  = _N.random.rand(dim, dim) * cc   #  use upper half only
    for i in xrange(dim):
        pij[i, i] = 1
    for i in xrange(dim):
        for j in xrange(i, dim):
            cm[i, j] = pij[i, j] * stds[i] * stds[j]
            cm[j, i] = cm[i, j]
    return cm

def createDataAR(N, B, err, obsnz, trim=0):
    #  a[1]^2 + 4a[0]
    #  B[0] = -0.45
    #  B[1] =  0.9
    err = _N.sqrt(err)
    obsnz  = _N.sqrt(obsnz)
    p = len(B)

    x    = _N.empty(N)
    y    = _N.empty(N)

    #  initial few
    for i in xrange(p+1):
        x[i] = err*_N.random.randn()
        y[i] = obsnz*_N.random.randn()

    for i in xrange(p+1, N):
        x[i] = _N.dot(B, x[i-1:i-p-1:-1]) + err*_N.random.randn()
        #  y = Hx + w   where w is a zero-mean Gaussian with cov. matrix R.
        #  In our case, H is 1 and R is 1x1
        y[i] = x[i] + obsnz*_N.random.randn()

    return x[trim:N], y[trim:N]

def createDataPP(N, B, beta, u, stNz, p=1, trim=0, x=None, absrefr=0):
    #  a[1]^2 + 4a[0]
    #  B[0] = -0.45
    #  B[1] =  0.9
    if type(u) != _N.ndarray:
        u = _N.ones(N) * u
    if x == None:
        k = len(B)
        stNz = _N.sqrt(stNz)

        rands = _N.random.randn(N)
        x    = _N.empty(N)
        for i in xrange(k+1):
            x[i] = stNz*rands[i]
        for i in xrange(k+1, N):
            #  for k = 2, x[i] = B[0]*x[i-2], B[1]*x[i - 1]
            #  B[0]   is the weight of oldest time point
            #  B[k-1] is weight of most recent time point
            #        x[i] = _N.dot(B, x[i-k:i]) + err*_N.random.randn()
            x[i] = _N.dot(B, x[i-1:i-k-1:-1]) + stNz*rands[i]
    else:
        k = 0

    spks = _N.zeros(N)
    prbs = _N.zeros(N)
    fs   = _N.zeros(N)

    #  initial few

    beta0 = beta[0]
    lspk  = -2*absrefr
    for i in xrange(k, N):
        e = _N.exp(u[i] + beta0* x[i]) * dt
        prbs[i]  = (p*e) / (1 + e)
        spks[i] = _N.random.binomial(1, prbs[i])
        if spks[i] == 1:
            if i - lspk <= absrefr:
                spks[i] = 0
            else:
                lspk = i

    fs[:] = prbs / dt

    return x[trim:N], spks[trim:N], prbs[trim:N], fs[trim:N]

def createDataPPl2_old(N, B, beta, u, stNz, lambda2, p=1, trim=0, x=None, offset=None):
    #  a[1]^2 + 4a[0]
    #  B[0] = -0.45
    #  B[1] =  0.9
    if type(u) != _N.ndarray:
        u = _N.ones(N) * u
    if x == None:
        k = len(B)
        stNz = _N.sqrt(stNz)

        rands = _N.random.randn(N)
        x    = _N.empty(N)
        for i in xrange(k+1):
            x[i] = stNz*rands[i]
        for i in xrange(k+1, N):
            #  for k = 2, x[i] = B[0]*x[i-2], B[1]*x[i - 1]
            #  B[0]   is the weight of oldest time point
            #  B[k-1] is weight of most recent time point
            #        x[i] = _N.dot(B, x[i-k:i]) + err*_N.random.randn()
            x[i] = _N.dot(B, x[i-1:i-k-1:-1]) + stNz*rands[i]
    else:
        k = 0

    spks = _N.zeros(N)
    prbs = _N.zeros(N)
    fs   = _N.zeros(N)

    #  initial few

    beta0 = beta[0]
    lh    = len(lambda2)
    hst  = []    #  spikes whose history is still felt

    for i in xrange(k, N):
        if offset != None:
            e = _N.exp(u[i] + offset[i] + beta0* x[i]) * dt
        else:
            e = _N.exp(u[i] + beta0* x[i]) * dt
        prbs[i]  = (p*e) / (1 + e)

        L  = len(hst)
        lmbd = 1

        for j in xrange(L - 1, -1, -1):
            ht = hst[j]
            #  if i == 10, ht == 9, lh == 1
            #  10 - 9 -1 == 0  < 1.   Still efective
            #  11 - 9 -1 == 1         No longer effective
            if i - ht - 1 < lh:
                lmbd *= lambda2[i - ht - 1]
            else:
                hst.pop(j)
        prbs[i] *= lmbd
        spks[i] = _N.random.binomial(1, prbs[i])
        if spks[i] == 1:
            hst.append(i)
            #print "lspk   %d" % lspk

    fs[:] = prbs / dt

    if offset != None:
        return x[trim:N], offset[trim:N], spks[trim:N], prbs[trim:N], fs[trim:N]
    return x[trim:N], spks[trim:N], prbs[trim:N], fs[trim:N]

def createDataPPl2(N, B, u, stNz, lambda2, nRhythms=1, p=1, trim=0, x=None, offset=None):
    beta = _N.array([1., 0.])
    #  a[1]^2 + 4a[0]
    #  B[0] = -0.45
    #  B[1] =  0.9
    if type(u) != _N.ndarray:
        u = _N.ones(N) * u
    if x == None:
        xc   = _N.empty((nRhythms, N))
        for nr in xrange(nRhythms):
            k = len(B[nr])
            sstNz = _N.sqrt(stNz[nr])

            rands = _N.random.randn(N)

            for i in xrange(k+1):
                xc[nr, i] = sstNz*rands[i]
            for i in xrange(k+1, N):
                #  for k = 2, x[i] = B[0]*x[i-2], B[1]*x[i - 1]
                #  B[0]   is the weight of oldest time point
                #  B[k-1] is weight of most recent time point
                #        x[i] = _N.dot(B, x[i-k:i]) + err*_N.random.randn()
                xc[nr, i] = _N.dot(B[nr], xc[nr, i-1:i-k-1:-1]) + sstNz*rands[i]
    else:
        k = 0

    if nRhythms > 1:
        x = _N.sum(xc, axis=0)
    else:
        x = xc.reshape(N, )
        
    spks = _N.zeros(N)
    prbs = _N.zeros(N)
    fs   = _N.zeros(N)

    #  initial few

    beta0 = beta[0]
    lh    = len(lambda2)
    hst  = []    #  spikes whose history is still felt

    for i in xrange(k, N):
        if offset != None:
            e = _N.exp(u[i] + offset[i] + beta0* x[i]) * dt
        else:
            e = _N.exp(u[i] + beta0* x[i]) * dt
        prbs[i]  = (p*e) / (1 + e)

        L  = len(hst)
        lmbd = 1

        for j in xrange(L - 1, -1, -1):
            ht = hst[j]
            #  if i == 10, ht == 9, lh == 1
            #  10 - 9 -1 == 0  < 1.   Still efective
            #  11 - 9 -1 == 1         No longer effective
            if i - ht - 1 < lh:
                lmbd *= lambda2[i - ht - 1]
            else:
                hst.pop(j)
        prbs[i] *= lmbd
        spks[i] = _N.random.binomial(1, prbs[i])
        if spks[i] == 1:
            hst.append(i)

    fs[:] = prbs / dt

    if offset != None:
        return x[:, trim:N], offset[trim:N], spks[trim:N], prbs[trim:N], fs[trim:N]
    return xc[:, trim:N], spks[trim:N], prbs[trim:N], fs[trim:N]

def rootIsMode(x, *args):
    F    = args[0]
    fltx = args[1]
    prV  = args[2]
    B0   = args[3]   #  first component of beta
    u    = args[4]
    spks = args[5]
    t    = args[6]
#    return x - _N.dot(F, fltx[:, t - 1])[0] - prV[0, 0, t] * B0 * (spks[t] - _N.exp(u + B0 * x)*0.001)
    return x - _N.dot(F, fltx[:, t - 1])[0] - prV[0, 0, t] * B0 * (spks[t] - _N.exp(u + B0 * x)*0.001)

def d_rootIsMode(x, *args):
    F    = args[0]
    fltx = args[1]
    prV  = args[2]
    B0   = args[3]   #  first component of beta
    u    = args[4]
    spks = args[5]
    t    = args[6]

    return 1 + prV[0, 0, t] * B0 * B0 * _N.exp(u + B0 * x) * 0.001

#def plottableSpkTms(dN, ymin, ymax):
def plottableSpkTms(dN, y):
    #  for each spike time,
    ts = []
    N  = len(dN)
    for n in xrange(N):
        if dN[n] == 1:
            ts.append(n)

    x_ticks = []
    y_ticks = []
    for t in ts:
        x_ticks.append(t)
        y_ticks.append(y)
#        x_ticks.append([t, t])
#        y_ticks.append([ymin, ymax])
    return x_ticks, y_ticks

def cmplxRoots(arC):
    N   = len(arC)   # polynomial degree       a_1 B + a_2 B^2
    A   = _N.zeros((N, N))
    bBdd = True

    for col in xrange(N):
        A[0, col] = arC[col]

    for row in xrange(1, N):
        A[row, row - 1] = 1

    vals, vecs = _N.linalg.eig(A)
    vroots = _N.empty(N)

    for roots in xrange(N):
        zR = 1 / vals[roots]
        vroots[roots] = (zR * zR.conj()).real
        if vroots[roots] < 1:
            bBdd = False

    return bBdd, vroots

def saveset(name, noparam=False):
    #  u, B, singleFreqAR, dt, stNz, x, dN, prbs
    xprbsdN = _N.empty((N + 1, 3))
    xprbsdN[:, 0] = x[:]
    xprbsdN[:, 1] = prbs[:]
    xprbsdN[:, 2] = dN[:]

    _N.savetxt(resFN("xprbsdN.dat", dir=name, create=True), xprbsdN, fmt="%.5e")

    if not noparam:
        fp = open(resFN("params.py", dir=name, create=True), "w")
        fp.write("u=%.3f\n" % u)
        fp.write("beta=%s\n" % arrstr(beta))
        fp.write("ARcoeff=_N.array(%s)\n" % str(ARcoeff))
        fp.write("alfa=_N.array(%s)\n" % str(alfa))
        fp.write("#  ampAngRep=%s\n" % ampAngRep(alfa))
        fp.write("dt=%.2e\n" % dt)
        fp.write("stNz=%.3e\n" % stNz)
        fp.write("absrefr=%d\n" % absrefr)
        fp.close()

def savesetMT(model, name, psth=False):
    #  u, B, singleFreqAR, dt, stNz, x, dN, prbs
    fp = open(resFN("params.py", dir=name, create=True), "w")

    bfn = "cnt_data"
    fmt = ""    
    if model != "bernoulli":
        fmt += "% .2e "
        fmt += "%d "
        fmt *= TR
    if model=="bernoulli":
        bfn = "xprbsdN"
        fmt += "% .2e "
        fmt += "%.3f %d "
        fmt *= TR

    _N.savetxt(resFN("%s.dat" % bfn, dir=name, create=True), alldat, fmt=fmt)
    fp.write("ARcoeff=_N.array(%s)\n" % str(ARcoeff))
    fp.write("alfa=_N.array(%s)\n" % str(alfa))
    for nr in xrange(nRhythms):
        fp.write("#  ampAngRep=%s\n" % ampAngRep(alfa[nr]))
    if not psth:
        fp.write("us=[")
        for tr in xrange(TR):
            fp.write("% .3e, " % us[tr])
    fp.write("]\n")
    fp.write("stNzs=[")
    for tr in xrange(TR):
        fp.write("%s, " % str(stNzs[tr]))
        fp.write("]\n")

    fp.write("TR=%d\n" % TR)
    fp.write("N=%d\n" % N)

    if model != "bernoulli":
        fp.write("ps   = _N.array(%s)\n" % str(ps).replace("  ", ", "))
    else:
        fp.write("dt=%.2e\n" % dt)
        fp.write("lowQs=%s\n" % str(lowQs))

    fp.close()

def savesetARpn(name):
    #  u, B, singleFreqAR, dt, stNz, x, dN, prbs
    xxlxhy = _N.empty((N + 1, 4))
    xxlxhy[:, 0] = x[:]
    xxlxhy[:, 1] = loAR[:]
    xxlxhy[:, 2] = hiAR[:]
    xxlxhy[:, 3] = y[:]

    _N.savetxt(datFN("xxlxhy.dat", dir=name, create=True), xxlxhy, fmt="%.5e")

    fp = open(resFN("params.py", dir=name, create=True), "w")
    fp.write("arLO=%s\n" % str(arLO))   #  want to keep these as list
    fp.write("arHI=%s\n" % str(arHI))
    fp.write("dt=%.2e\n" % dt)
    fp.write("stNzL=%.3e\n" % stNzL)
    fp.write("stNzH=%.3e\n" % stNzH)
    fp.write("obsNz=%.3e\n" % obsNz)
    fp.write("H=%s\n" % arrstr(H))
    fp.close()

def arrstr(_arr):
    dim1 = 0
    if type(_arr) == list:
        dim1 = 1
        arr = _N.array(_arr)
    else:
        arr = _arr
    if len(arr.shape) == 1:   #  assume it is a row vector
        dim1 = 1
        cols = arr.shape[0]
        arrR = arr.reshape((1, cols))
    else:
        arrR = arr

    if dim1 == 0:
        strg = "_N.array(["
    else:
        strg = "_N.array("

    c = 0
    for r in xrange(arrR.shape[0] - 1):
        strg += "["
        for c in xrange(arrR.shape[1] - 1):
            strg += ("% .6e" % arrR[r, c]) + ", "
        strg += ("% .6e" % arrR[r, c]) + "], \n"

    #  for last row (or first, if only 1 row)
    r = arrR.shape[0] - 1
    strg += "["

    c = 0
    for c in xrange(arrR.shape[1] - 1):
        strg += ("% .6e" % arrR[r, c]) + ", "
    strg += ("% .6e" % arrR[r, arrR.shape[1] - 1]) + "]"
    if dim1 == 0:
        strg += "])"
    else:
        strg += ")"

    return strg

def quickPSTH(alldat, TR, COLS, plot=False, fn=None):
    spks = []
    N    = alldat.shape[0]
    for tr in xrange(TR):
        spks.extend(_N.where(alldat[:, COLS-1+tr*COLS] == 1)[0])

    if plot and (fn != None):
        fig = _plt.figure()
        _plt.hist(spks, bins=_N.linspace(0, N, 100), color="black")
        _plt.xlim(0, N)
        _plt.savefig(fn)
        _plt.close()

    return spks
