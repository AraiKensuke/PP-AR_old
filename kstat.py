import scipy.stats as _ss
import scipy as _sci
import matplotlib.pyplot as _plt
import numpy as _N
import utilities as _U


def poi_pdf(l, karr):
   rv = _ss.poisson(l)
   return rv.pmf(karr)

def poi_ch2test(cts):
    if _N.sum(cts) == 0:  #  all of them 0
        return 0.5    #  Poisson with rate 0
    TRIALS    = len(cts)

    rareLimL = 0
    rareLimH = 0
    obsLam   = _N.mean(cts)

    i = int(obsLam)
    while True:
        if poi_pdf(obsLam, i)*TRIALS < 1:
            rareLimH = i - 1
            break   #  inclusive of rareLim and up
        i += 1

    i = int(obsLam)
    while True and (i >= 0):
        if poi_pdf(obsLam, i)*TRIALS < 1:
            rareLimL = i + 1
            break   #  inclusive of rareLim and up
        i -= 1

    #  no bin has < 1 expcted events

    expctd   = _N.zeros(rareLimH + 1)
    for n in range(rareLimH):
        expctd[n] = poi_pdf(obsLam, n)*TRIALS
    expctd[rareLimH] = TRIALS - _N.sum(expctd[0:rareLimH])

    #  poipdf[rareLim] is at < 1.  This will be last square.
    #  so this is rareLim + 1 objects
    maxInd              = max(cts)  #  0 based index
    nbins = maxInd - 0 + 1

    hf, low, bs, out = _ss.histogram(cts, numbins=nbins, defaultlimits=(0, maxInd + 1))

    #  # of categories:  rareLim + 1    0..4 -> 1..5  (5 cats).  
    #  k == # of classes

    #  shortened version

    #  if rareLim is the last spot, then length is rareLim + 1
    #  highest count in cts is len(vals) - 1
    #  maxLim == len(vals) - 1   
    #  to accomodate last index rareLim, we need size rareLim + 1
    #  
    svals                =  _N.zeros(rareLimH + 1)
    if maxInd <= rareLimH:
        svals[0:maxInd + 1] =  hf[0:maxInd + 1]
        
    else:
        svals[0:rareLimH] =  hf[0:rareLimH]
        svals[rareLimH]       =  _N.sum(hf[rareLimH:])

    svals[rareLimL]      = _N.sum(hf[0:rareLimL+1])
    expctd[rareLimL]     = _N.sum(expctd[0:rareLimL+1])

    k                   = rareLimH - rareLimL + 1  # index of last element if index from 1
    # [0, 1], [1, 2], ... [k-2, k-1]
    #   # classes (counts) [0, 1, 2, k-2]   (k - 1) classes
    #  1...k inclusive is k classes

    chi2   = 0

    for i in xrange(rareLimL, rareLimH + 1):
        o    =  svals[i]   
        e    =  expctd[i]
        chi2  +=  (o-e)**2/e
        i    += 1

    edf  = k - 2
    pv = 1 - _ss.chi2.cdf(chi2, edf)

    return pv

#  cumulative fraction
def cumfrac(x, staircase=False, countdat=False, histogram=False, bins=None, binsAlignLeft=False):
    """
    cumulative fraction
    types of data:

    continuous data        sort data, then assign size rank to sorted points
    countdata              make a histogram of data first
    histogram              cumulatively add values of bins

    cum frac looks like a staircase when plotted w/ lines
    staircase=True will include the 
    cnts   if data is a list of counts (like spks per trial), we will most likely have many instances of, ie the number 4, in our data.  We just need to consider all 

    sanity check
    cnts = [1, 2, 3, 4, 5]  or [1, 2, 3, 4, 5, 3]
    cf = _ks.cumfrac(cnts, countdat=True, staircase=True)    
    plot(cf[:, 0], cf[:, 1])
    """

    if countdat:
        lo = min(x)
        hi = max(x)
        nbins = hi - lo + 1
        hf, low, bs, out = _ss.histogram(x, numbins=nbins, defaultlimits=(lo, hi + 1))

        datn = int(_N.sum(hf))
        if not staircase:
            cf   = _N.zeros((hi - lo + 1, 2))
            tot  = 0
            for i in xrange(len(hf)):
                tot += hf[i]
                cf[i, 0] = lo + i
                cf[i, 1] = float(tot) / datn
        else:
            cf   = _N.zeros(((hi - lo + 1)*2, 2))
            tot  = 0
            for i in xrange(len(hf)):
                cf[2*i, 0]     = lo + i
                cf[2*i + 1, 0] = lo + i
                cf[2*i, 1]     = float(tot) / datn
                tot            += hf[i]
                cf[2*i + 1, 1] = float(tot) / datn

    else:
        sx = _N.sort(x)
        N  = len(sx)

        if not staircase:  
            cf = _N.zeros((N, 2))
            cf[:, 0] = sx[:]
            cf[:, 1] = _N.linspace(0, 1, N, endpoint=False) + 1./N
        else:
            cf = _N.zeros((2*N, 2))

            yvals = _N.linspace(0, 1, N, endpoint=False)
            for n in xrange(N):
                cf[2*n, 0]      =  sx[n]
                cf[2*n + 1, 0]  =  sx[n]
                cf[2*n, 1]      =  yvals[n]
                cf[2*n + 1, 1]  =  yvals[n] + 1./N
                
    if histogram:
        tot = _N.sum(x)
        N   = len(x)
        cf  = _N.zeros((N, 2))
        ccf = 0
        if bins == None:
            bins = range(N + 1)
        for b in xrange(N):
            cf[b, 0] = bins[b + 1]
            cf[b, 1] = ccf
            ccf += x[b]
        cf[N - 1, 0] = bins[b + 1]
        cf[N - 1, 1] = tot

    return cf

def percentile(x, countdat=False):
    if countdat:
        lo = min(x)
        hi = max(x)
        nbins = hi - lo + 1
        hf, low, bs, out = _ss.histogram(x, numbins=nbins, defaultlimits=(lo, hi + 1))

        datn = int(_N.sum(hf))
        pctl   = _N.zeros((hi - lo + 1, 2))
        tot  = 0
        for i in xrange(len(hf)):
            tot += hf[i]
            pctl[i, 0] = lo + i
            pctl[i, 1] = float(tot + 1) / float(datn + 1)

    else:
        sx = _N.sort(x)
        N  = len(sx)

        pctl = _N.zeros((N, 2))

        for n in xrange(N):
            pctl[n, 0]      =  sx[n]
            pctl[n, 1]      =  float(n + 1) / float(N + 1)
            
    return pctl

def subtractDiffSupport(xy1, xy2):
    """
    subtract two functions that have a different support.
    occurs for example when the functions are the result of random sampling
    """

    p2    = 0
    ix    = 0
    lastxy1_y = 0
    lastxy2_y = 0
    
    L1 = len(xy1[:, 0])
    L2 = len(xy2[:, 0])
    subt_x = []
    subt_y = []
    for ix in xrange(L1):
        xy1_x = xy1[ix, 0]
        xy1_y = xy1[ix, 1]

        while (p2 < L2) and (xy2[p2, 0] < xy1_x):
            subt_x.append(xy2[p2, 0])
            subt_y.append(lastxy1_y - xy2[p2, 1])
            lastxy2_y     = xy2[p2, 1]
            p2 += 1

        subt_x.append(xy1_x)
        subt_y.append(xy1_y - lastxy2_y)
        lastxy1_y = xy1_y

    subtracted = _N.array([subt_x, subt_y]).T

    return subtracted

def subtract_discrete_cf(xy1, xy2):
    """
    subtract two functions that have (almost) same support.  
    """

    lo1 = int(min(xy1[:, 0]))
    hi1 = int(max(xy1[:, 0]))
    lo2 = int(min(xy2[:, 0]))
    hi2 = int(max(xy2[:, 0]))

    LO  = min([lo1, lo2])
    HI  = max([hi1, hi2])

    y1  = _N.zeros(HI - LO + 1)
    y2  = _N.zeros(HI - LO + 1)
    x1  = _N.zeros(HI - LO + 1)
    x2  = _N.zeros(HI - LO + 1)


    y1[0:lo1-LO] = 0
    
    y1[lo1-LO:lo1-LO+(hi1-lo1+1)] = xy1[:, 1]
    y1[hi1+1-LO:HI-LO+1] = 1

    y2[0:lo2-LO] = 0
    y2[lo2-LO:lo2-LO+(hi2-lo2+1)] = xy2[:, 1]
    y2[hi2+1-LO:HI-LO+1] = 1

    return range(LO, HI + 1), y1 - y2


def poissonEM_2comp(cts, ICs=20):
    """
    fit data cts with mixture Poisson distribution using EM
    EM depends on initial conditions.  ICs is # of initial conditions to try
    """
    N       = len(cts)
    K       = 2
    pik     = _N.zeros(K)
    lamk    = _N.zeros(K)
    bestf   = _N.zeros(K)
    bestp   = _N.zeros(K)

    maxLL = -10000000.
    uc      = _N.mean(cts)

    Nk      = _N.zeros(K)
    resp    = _N.zeros((N, K))

    for ic in xrange(ICs):
        pik[0]  = _N.random.rand()
        pik[1]  = 1 - pik[1]

        lamk[0] = uc * 2*_N.random.rand()
        lamk[1] = uc * 2*_N.random.rand()

        Nk[:]         = 0.
        resp[:, :]    = 0.

        llo     = -10000000000.

        while True:
            #  build responsibility array
            for n in xrange(N):
                bot = 0

                for k in xrange(K):
                    bot += pik[k] * poi_pdf(lamk[k], cts[n])

                for k in xrange(K):
                    resp[n, k] = (pik[k] * poi_pdf(lamk[k], cts[n])) / bot
            #  build responsibility array        
            for k in xrange(K):
                Nk[k] = 0
                for n in xrange(N):
                    Nk[k] += resp[n, k]

            #  update params
            for k in xrange(K):
                lamk[k] = 0
                for n in xrange(N):
                    lamk[k] += cts[n] * resp[n, k]
                lamk[k] /= Nk[k]

                pik[k] = Nk[k] / float(N)

            #  log likelihood
            ll   = 0
            for n in xrange(N):
                logarg = 0
                for k in xrange(K):
                    logarg += pik[k] * poi_pdf(lamk[k], cts[n])

                ll += _N.log(logarg)

            if (_N.abs(llo - ll) / _N.abs(_N.mean(llo + ll))) < 0.001:
                break

            noprob = True
            if _N.isinf(ll) or _N.isnan(ll):
                #  reinitialize, hit floating point error
                print "hit float error, reinit params, kstat.poissonEM_2comp()"
                noprob = False
                pik[0]  = _N.random.rand()
                pik[1]  = 1 - pik[1]
                lamk[0] = uc * 2*_N.random.rand()
                lamk[1] = uc * 2*_N.random.rand()

            if noprob:
               llo = ll    #  last log likelihood.  each step must make smaller
            if (ll > maxLL) and noprob:
                maxLL = ll
                bestf[0] = lamk[0]
                bestf[1] = lamk[1]
                bestp[0] = pik[0]
                bestp[1] = pik[1]
        
    #  I want lamk[0] to be the lower one
    if bestf[0] > bestf[1]:
       temp    = bestf[0]
       bestf[0] = bestf[1]
       bestf[1] = temp
       temp    = bestp[0]
       bestp[0]  = bestp[1]
       bestp[1]  = temp

    return bestf, bestp


def gaussianEM_2comp(x, ICs=100, cond=0.01):
    """
    fit data cts with mixture Poisson distribution using EM
    EM depends on initial conditions.  ICs is # of initial conditions to try
    """
    N       = len(x)
    K       = 2
    pik     = _N.zeros(K)   #  weights
    uk      = _N.zeros(K)   
    sk      = _N.zeros(K)   #  std deviation
    vk      = _N.zeros(K)   #  variance
    bestu   = _N.zeros(K)
    bests   = _N.zeros(K)
    bestp   = _N.zeros(K)

    s_u     = _N.mean(x)
    s_s     = _N.std(x)       
    #  need these params to check if updated component variances too small
    minDat = min(x) 
    maxDat = max(x)
    avgDist= (maxDat - minDat) / float(10*N)

    maxLL = -10000000.

    iters   = _N.zeros(ICs)
    for ic in xrange(ICs):
        pik[0]  = _N.random.rand()
        pik[1]  = 1 - pik[0]

        uk[0] = s_u * (2*_N.random.rand())
        uk[1] = s_u * (2*_N.random.rand())
        sk[0] = s_s * (2*_N.random.rand())
        sk[1] = s_s * (2*_N.random.rand())
        vk[0] = sk[0]*sk[0]
        vk[1] = sk[1]*sk[1]

        Nk      = _N.zeros(K)
        resp    = _N.zeros((N, K))

        llo     = -10000000000.

        while True:
            noprob = True
            iters[ic] += 1
            #  build responsibility array
            rvs = []

            for k in xrange(K):
                rvs.append(_ss.norm(uk[k], sk[k]))
            for n in xrange(N):
                bot = 0

                #  bot is near 0 if component std. devs too small
                for k in xrange(K):
                   bot += pik[k] * rvs[k].pdf(x[n])

                for k in xrange(K):
                   rv  = _ss.norm(uk[k], sk[k])
                   resp[n, k] = (pik[k] * rv.pdf(x[n])) / bot
            #  build responsibility array

            for k in xrange(K):
                Nk[k] = 0
                for n in xrange(N):
                    Nk[k] += resp[n, k]

            #  need new rvs
            #  update params
            for k in xrange(K):
                vk[k] = 0
                for n in xrange(N):
                   vk[k] += (x[n] - uk[k])* (x[n] - uk[k])* resp[n, k]
                vk[k] /= Nk[k]
                sk[k] = _N.sqrt(vk[k])

            for k in xrange(K):
                uk[k] = 0
                for n in xrange(N):
                    uk[k] += x[n] * resp[n, k]
                uk[k] /= Nk[k]

                pik[k] = Nk[k] / float(N)

            rvs = []
            for k in xrange(K):
                rvs.append(_ss.norm(uk[k], sk[k]))

            #  log likelihood
            ll   = 0
            for n in xrange(N):
                logarg = 0
                for k in xrange(K):
                   logarg += pik[k] * rvs[k].pdf(x[n])

                ll += _N.log(logarg)

            if (_N.abs(llo - ll) / _N.abs(llo + ll)) < cond:
                break      #  stationary ll hit.  llo is log-lik of prev. iter

            if _N.isinf(ll) or _N.isnan(ll):
                iters[ic] = 0
                #  reinitialize, hit floating point error
                print "hit float error, reinit params, kstat.gaussianEM_2comp()"
                noprob = False
                pik[0]  = _N.random.rand()
                pik[1]  = 1 - pik[0]
                uk[0] = s_u * (2*_N.random.rand())
                uk[1] = s_u * (2*_N.random.rand())
                sk[0] = s_s * (2*_N.random.rand())
                sk[1] = s_s * (2*_N.random.rand())
                vk[0] = sk[0]*sk[0]
                vk[1] = sk[1]*sk[1]
#                print "%(0).3e    %(1).3e" % {"0" : sk[0], "1" : sk[1]}

            if ll > maxLL:
                #  update best fit parameters
                if (sk[0] > avgDist) and (sk[1] > avgDist):
                    maxLL = ll        #  component dists. wide enuff, not singularity
                    bestu[0] = uk[0]
                    bestu[1] = uk[1]
                    bests[0] = sk[0]
                    bests[1] = sk[1]
                    bestp[0] = pik[0]
                    bestp[1] = pik[1]
                else:
                    noprob = False
                    iters[ic] = 0
                    pik[0]  = _N.random.rand()
                    pik[1]  = 1 - pik[0]
                    uk[0] = s_u * (2*_N.random.rand())
                    uk[1] = s_u * (2*_N.random.rand())
                    sk[0] = s_s * (2*_N.random.rand())
                    sk[1] = s_s * (2*_N.random.rand())
                    vk[0] = sk[0]*sk[0]
                    vk[1] = sk[1]*sk[1]
                    print "std. dev too small, reinit"
            if noprob:
                llo = ll
        
    #  I want lamk[0] to be the lower one
    if bestu[0] > bestu[1]:
       temp    = bestu[0]
       bestu[0] = bestu[1]
       bestu[1] = temp
       temp    = bestp[0]
       bestp[0]  = bestp[1]
       bestp[1]  = temp
       temp    = bests[0]
       bests[0]  = bests[1]
       bests[1]  = temp

    return bestu, bests, bestp, iters

def rmOutliers(arr):
    return arr
#    return srtd[1:len(arr)-1]


def c2v(cnts, blksz=50, start=None, stop=None):
    L = len(cnts)
    orig = _N.zeros(L, dtype=_N.int)
    if start == None:
        for i in range(L):
            if cnts[i] > 0:
                start = i
                break
    if stop == None:
        for i in range(L-1, -1, -1):
            if cnts[i] > 0:
                stop = i
                break

    if (start == None) or (stop == None):
        return None, None, None, None, None, None
    roi  = _N.zeros(stop - start, dtype=_N.int)
    orig = _N.zeros(stop - start, dtype=_N.int)
    roi[:] = cnts[start:stop]

    hlfblksz = blksz/2
    blks=(stop-start)/hlfblksz - 1

    #  blks  rows x 4 cols
    c2vs = _N.zeros(blks)
    for b in xrange(blks):
       i = b*hlfblksz
       if _N.sum(roi[i:i + blksz]) == 0:       #  not firing in any of these trials
          c2vs[b] = 0
       else:
          c2vs[b] = _N.std(roi[i:i + blksz])**2/_N.mean(roi[i:i + blksz])

    return _N.mean(c2vs)

def trend(ct, winsz=5, useBott=2):
    """
    does data ct have a trend?  look at 
    useBott == use lowest "useBott" numbers per block to assess trend
    """
    blks  = len(ct)/winsz
    xs    = range(blks)
    emins = []

    if useBott > 1:
        for i in xrange(blks):
            srtd = _N.sort(ct[winsz*i:winsz*(i+1)])
            emins.append(_N.sum(srtd[0:useBott]))
    else:
        for i in xrange(blks):
            emins.append(min(ct[winsz*i:winsz*(i+1)]))   
    (a_s,b_s,r,tt,stderr)=_ss.linregress(xs, emins)

    eas = abs(a_s)

    #  if slope is 0, don't even do test, just return False.
    #  often helps when count rate is very low, and most are just 0
    if eas == .0:
        return False

    TRIALS = 100
    bgr    = 0   #  slope of data surrogate bigger than data
    smins  = _N.zeros(blks)
    for tr in xrange(TRIALS):
        #  scramble
        sct = _U.shuffle(ct)

        #  sort is by default ascending
        if useBott > 1:
            for i in xrange(blks):
                srtd = _N.sort(sct[winsz*i:winsz*(i+1)])
                smins[i]  = _N.sum(srtd[0:useBott])
        else:
            for i in xrange(blks):
                smins[i] = (min(sct[winsz*i:winsz*(i+1)]))
        (a_s,b_s,r,tt,stderr)=_ss.linregress(xs, smins)

        ma_s = abs(a_s)

        if (eas < ma_s):
            bgr += 1
        
    if float(bgr) / TRIALS < 0.05:
        return True
    return False


def threshold(cnts, abestfps, twost, blksz=50, start=None, stop=None):
    """
    turn 2-state spk count data into a 0, 1 sequence.  
    cnts len L, return sequence hilos and thresh integer multiples of blksz
    abestfps is overlapping blocks of size blksz, slid every hlfblksz 
    
    if start, stop and blksz=(start - stop) specified, using whole data without slicing up
    """
    L     = len(cnts)
    if start == None:
        for i in range(L):
            if cnts[i] > 0:
                start = i
                break
    if stop == None:
        for i in range(L-1, -1, -1):
            if cnts[i] > 0:
                stop = i+1
                break

    blks=(stop-start)/blksz           #  full-size blocks

    hilos   = _N.zeros(L, dtype=_N.int)
    hilos[0:start] = -1
    hilos[stop:]   = -1

    thrshlds = _N.zeros(blks)

    for b in xrange(blks):  #  slide block along by hlfblksz steps
        i = b*blksz      #  i max is ((2*blks)-2)*hlfblksz
                           #  if blks == 3, blksz == 15
                           #  0 ... 4 (inclusive)

        f1 = abestfps[b, 0]
        f2 = abestfps[b, 1]
        w1 = abestfps[b, 2]
        w2 = abestfps[b, 3]

        mE = 100.    # maximum error  (percent, largest error is 1)
        bkt= 3*int(f2)  #  
        if bkt == 0:
            bkt = 1

        for kt in range(0, int(3*f2)):
            nE = 0     #  new error
            for k in xrange(0, kt):   #  kt - 1 (inclusive)
                nE += w2 * poi_pdf(f2, k)
            for k in xrange(kt, int(3*f2)):   #  kt - 1 (inclusive)
                nE += w1 * poi_pdf(f1, k)

            if nE < mE:
                mE = nE
                bkt= kt
        #  it is possible for bkt to be < f1 or > f2 if the weights are very
        #  unbalanced.  (for case w1 >> w2), in such a case, mathematically it 
        #  is better to put threshold above f2.  Because w1 so large, it only
        #  makes sense to classify especially large spkcts as high state.

        thrshlds[b] = bkt

    thr2st  = 0  #  avg. value of threshold using data where it makes sense
    nthr2st = 0
    for b in xrange(blks):  #  slide block along by hlfblksz steps
        if twost[b] == 1:   #  threshold makes sense
            nthr2st += 1
            thr2st += thrshlds[b]

    thr2st = max(cnts) + 1   #  all 1 state
    if nthr2st > 0:
        thr2st /= float(nthr2st)
           
    for b in xrange(blks):  #  slide block along by hlfblksz steps
        if twost[b] == 0:   #  threshold doesn't make sense
            thr2use = thr2st
        else:
            thr2use = thrshlds[b]
        for i in xrange(start+b*blksz, start+(b+1)*blksz):
            if cnts[i] < thr2use:   #  if
                hilos[i] = 0
            else:
                hilos[i] = 1
    for i in xrange(start + blks*blksz, stop):   #  outside blocks
        if cnts[i] < thr2st:
            hilos[i]  = 0
        else:
            hilos[i]  = 1

    return start, start + blksz*blks, hilos, thrshlds

def thresholdOLD(cnts, abestfps, blksz=50, start=None, stop=None):
   """
   turn 2-state spk count data into a 0, 1 sequence.  
   cnts len L, return sequence hilos and thresh integer multiples of blksz
   abestfps is overlapping blocks of size blksz, slid every hlfblksz 

   if start, stop and blksz=(start - stop) specified, using whole data without slicing up
   """
   L = len(cnts)
   orig = _N.zeros(L, dtype=_N.int)

   if start == None:
      for i in range(L):
         if cnts[i] > 0:
            start = i
            break
   if stop == None:
      for i in range(L-1, -1, -1):
         if cnts[i] > 0:
            stop = i
            break

   if (start == None) or (stop == None):
      return None, None, None, None, None, None
   roi  = _N.zeros(stop - start, dtype=_N.int)
   orig = _N.zeros(stop - start, dtype=_N.int)
   roi[:] = cnts[start:stop]
   
   hlfblksz = blksz/2
   blks=(stop-start)/blksz           #  full-size blocks

   hilos = _N.zeros(L, dtype=_N.int)
   hilos[0:start] = -1
   hilos[stop:]   = -1
   
   xs = range(start, start + blks*blksz, 1)
   kts= _N.zeros((2*blks)-1, _N.int)   #  full-size, but slid by hlfblksz
   
   for b in xrange((2*blks)-1):  #  slide block along by hlfblksz steps
      i = b*hlfblksz      #  i max is ((2*blks)-2)*hlfblksz
                           #  if blks == 3, blksz == 15
                           #  0 ... 4 (inclusive)

      f1 = abestfps[b, 0]
      f2 = abestfps[b, 1]
      w1 = abestfps[b, 2]
      w2 = abestfps[b, 3]

      mE = 10000    # maximum error
      bkt= -1

      #  diff of lo and hi small, or if fairly large, the weights are so
      #  skewed that it is likely that its really same hi and lo
      if (abs(f1 - f2) / (f1 + f2) < 0.25) or \
             ((abs(f1 - f2) / (f1 + f2) < 0.35) and
              ((w1 > 9*w2) or (w2 > 9*w1))):
         #  
         bkt = int(w1*f1 + w2*f2)
         if bkt == 0:   #  don't let kt be 0.  
            bkt = 1
      else:
         for kt in range(0, 3*int(f2)):
            nE = 0     #  new error
            for k in xrange(0, kt):   #  kt - 1 (inclusive)
               nE += w2 * poi_pdf(f2, k)
            for k in xrange(kt, int(f2*3)):   #  kt - 1 (inclusive)
               nE += w1 * poi_pdf(f1, k)

            if mE > nE:
               mE = nE
               bkt= kt

      kts[b] = bkt

    #  avg of kts over a range of size hlfblksz
   avgdKTs = _N.zeros(2*blks)    #  in size of hlfblksz
   avgdKTs[0] = kts[0]
   avgdKTs[2*blks-1] = kts[2*blks - 2]
   for b in xrange(1, 2*blks-1):
      avgdKTs[b] = 0.5*(kts[b - 1] + kts[b])

   #  assuming excitability slowly drifts, we linearly connect neighboring
   #  avgKTs to obtain drifting threshold
   thrshlds = _N.zeros(blks*blksz)
   thrshlds[0:hlfblksz]    = avgdKTs[0]
   thrshlds[(2*blks-1)*hlfblksz:2*blks*hlfblksz] = avgdKTs[2*blks-1]
   for b in xrange(1, 2*blks-1):
      thrshlds[b*hlfblksz:(b+1)*hlfblksz] = _N.linspace(avgdKTs[b-1], avgdKTs[b], hlfblksz)

   for b in xrange(2*blks):      #  blksz is
      for i in xrange(hlfblksz):
         hilos[start + b*hlfblksz + i] = 0
         if roi[b*hlfblksz + i] >= avgdKTs[b]:
            hilos[start + b*hlfblksz + i] = 1

   return start, start + 2 * blks, hilos, thrshlds

def rmLargest(mdat, frm_cols=[], n=0, largest=True):
   """
   remove n largest (smallest) values using data in frm_cols as keys
   each data in rows
   """
   if (type(frm_cols) != list) and (type(frm_cols) != tuple):
      frm_cols = [frm_cols]
      
   ndat   = mdat.shape[0]    #  number of data
   ncls   = mdat.shape[1]    #  data dimension
   keep   = range(ndat)      #  indices of data to keep

   for c in frm_cols:
      s   = _N.sort(mdat[:, c])   #  ascending
      L   = len(s)
      rmthese = []
      jstThsCol = mdat[:, c].tolist()
      for nn in xrange(n):
         if largest:
            i   = jstThsCol.index(s[L-nn-1])   #  tells me row
         else:
            i   = jstThsCol.index(s[nn])       #  tells me row
         try:
            ind = keep.index(i)
            keep.pop(ind)
         except ValueError:
            1+1
   L     = len(keep)
   retDat = _N.zeros((L, ncls), dtype=mdat.dtype)
   r     = 0

   for k in keep:
      retDat[r, :] = mdat[k, :]
      r += 1
   return retDat


def rmOutlier(mdat, per=0.0, sd=0.0, col=0, topbottom=0):
   """
   Multi dimensional data.  Check column col for top and bottom per percent of data to remove
   per is between 0 and 1
   topbottom  if using percent, if topbottom == 0, remove top and bottom per percent of data.  if topbottom == -1(1), remove bottom(top) per percent only 
   stddev  how many stddevs
   """
   if len(mdat.shape) == 1:
      mdat = mdat.reshape((len(mdat), 1))
   N, D = mdat.shape

   if per != 0.0:
      scol = _N.sort(mdat[:, col])   #  sorted col
      kL   = int(per*len(scol))
      if kL == 0:
         kL = 1
      if (topbottom <= 0):
         rmLo = scol[0:kL].tolist()
      if (topbottom >= 0):
         rmHi = scol[N-kL:N].tolist()
      if topbottom < 0:
         rmHi = []
         rmdat = _N.zeros((N-kL, D), dtype=mdat.dtype)
      elif topbottom > 0:
         rmLo = []
         rmdat = _N.zeros((N-kL, D), dtype=mdat.dtype)
      else:
         rmdat = _N.zeros((N-2*kL, D), dtype=mdat.dtype)

   elif sd != 0.0:
      u    = _N.mean(mdat[:, col])
      s    = _N.std(mdat[:, col])

      rmLo = []
      rmHi = []
      for n in xrange(N):
         if mdat[n, col] > u + sd*s:
            rmHi.append(mdat[n, col])
         elif mdat[n, col] < u - sd*s:
            rmLo.append(mdat[n, col])

      rmdat = _N.zeros((N-len(rmHi)-len(rmLo), D), dtype=mdat.dtype)

   n = 0
   for i in xrange(N):
      rmFromLo = False
      rmFromHi = False
      try:
         #  don't include in rmdat
         ind = rmLo.index(mdat[i, col])
         rmLo.pop(ind)
         rmFromLo = True
      except ValueError:
         rmFromLo = False
      try:
         #  don't include in rmdat
         ind = rmHi.index(mdat[i, col])
         rmHi.pop(ind)
         rmFromHi = True
      except ValueError:
         rmFromHi = False

      if (not rmFromLo) and (not rmFromHi):
         #  didn't find it in rmLo.  Include in rmdat
         rmdat[n, :] = mdat[i, :]
         n += 1

   if D == 1:
      if per != 0.0:
         if topbottom == 0:
            rmdat = rmdat.reshape((N-2*kL, ))
         else:
            rmdat = rmdat.reshape((N-kL, ))
      else:
         rmdat = rmdat.reshape((rmdat.shape[0], ))

   return rmdat

def anova2way(mdatRxC):
    """
    here, 1 value for each category
    # # of rows == # of yoin A categories
    # # of cols == # of yoin B categories
    """
    #  yoin A has R categories (each row a different A yoin)
    ubb =     _N.mean(mdatRxC[:, :])

    rws, cls = mdatRxC.shape
    SA  = 0
    for rw in xrange(rws):
        SA += (_N.mean(mdatRxC[rw, :]) - ubb)**2
    SA *= cls
    dfA    = rws - 1

    SB  = 0
    for cl in xrange(cls):
        SB += (_N.mean(mdatRxC[:, cl]) - ubb)**2
    SB *= rws
    dfB = cls - 1

    ST  = 0
    for rw in xrange(rws):
        for cl in xrange(cls):
            ST += (mdatRxC[rw, cl] - ubb)**2
    dfT = rws*cls - 1

    SE   = ST - SA - SB
    dfE  = (rws - 1)*(cls - 1)

    sdA2 = SA / dfA
    sdB2 = SB / dfB
    sdE2 = SE / dfE

    FA   = sdA2/sdE2
    FB   = sdB2/sdE2

    #  for B
    pvA = FandPV(dfA, dfE, FA)
    pvB = FandPV(dfB, dfE, FB)
    return pvA, pvB

def anova2wayUseMeans(datRxC, rws, cls):
    """
    Anova 
    here, several values for each category
    """
    mdatRxC = _N.zeros((rws, cls))

    for ix in xrange(rws):
        for iy in xrange(cls):
            mdatRxC[ix, iy] = _N.mean(datRxC[ix][iy])

    return anova2way(mdatRxC)

def FandPV(df1, df2, fval):
    rv = _ss.f(df1, df2)
    return 1 - rv.cdf(fval)

def entropy(x):
    """
    x can be [0, 1, 3, 2, 4, 2, 1]
    or 
    ["0", "1", "3", "2", "3", "2", "1"]
    or 
    ["00", "01", "11", "10", "11", "10", "01"]
    """
    L = len(x)
    hash = {}
    nl2 = _N.log(2)

    for l in xrange(L):
        if hash.has_key(x[l]):
            hash[x[l]] += 1
        else:
            hash[x[l]] = 1

    N       = len(hash)
    kys     = hash.keys()
    nSymbls = len(kys)

    H       = 0
    for n in xrange(len(kys)):
        p   = hash[kys[n]] / float(L)

        if p > 0:
            H += p * _N.log(p) / nl2
    H *= -1

    return H
        
def c_entropy(x, c):
    """
    sum_r p(r) H(x | r)
    x can be [0, 1, 3, 2, 4, 2, 1]
    or 
    ["0", "1", "3", "2", "3", "2", "1"]
    or 
    ["00", "01", "11", "10", "11", "10", "01"]
    """
    nl2 = _N.log(2)

    L = len(c)   #  length of c same as length of x
    hash = {}    

    for l in xrange(L):   #  loop over conditional variables
        if hash.has_key(c[l]):
            hash[c[l]] += 1
        else:
            hash[c[l]] = 1
        
    N       = len(hash)   #  number of conditional symbols
    kys     = hash.keys()
    nSymbls = len(kys)

    H       = 0

    for n in xrange(len(kys)):
        p   = hash[kys[n]] / float(L)

        #  conditional symbol is kys[n]
        xc  = []

        Hc  = 0
        for m in xrange(L):
            if c[m] == kys[n]:
                xc.append(x[m])
        Hc = entropy(xc)

        H   += p * Hc

    return H


def ranFromPercentile(pctl, minx=None):
    """
    Distribution given as CDF.  
    if minx != None, I want to sample a truncated distribution, return 
    only values above minx.
    """
    A = 1
    rminx = 0
    L = pctl.shape[0]
    if minx != None:
        try:
            irminx = _N.where(pctl[:, 0] >= minx)[0][0]
        except IndexError:
            return pctl[L-1, 0]
        rminx  = pctl[irminx, 1]
        A = 1 - rminx

    r = rminx + A*_N.random.rand()
    try:
        ret = _N.where(pctl[:, 1] >= r)[0][0]
    except IndexError:
        ret = pctl[L-1, 1]

    return pctl[ret, 0]
