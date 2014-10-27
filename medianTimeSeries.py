from kflib import createDataAR
#  TR x ITERS x N
TR    = 1
ITER  = 200
N     = 1000
wfs   = _N.empty((TR, ITER, N))

B     = _N.array([0.9, -0.45])

for tr in xrange(TR):
    for it in xrange(ITER):
        wfs[tr, it], y = createDataAR(N, B, 0.1, 0.1)

#ds    = _N.empty((ITER, ITER))

avgDs  = _N.empty(ITER)

##  Now do the distances
for tr in xrange(TR):
    for it1 in xrange(ITER):
        avgD = 0
        for it2 in xrange(it1):
            df = wfs[tr, it1] - wfs[tr, it2]
            avgD += _N.sqrt(_N.dot(df, df))
        for it2 in xrange(it1+1, ITER):
            df = wfs[tr, it1] - wfs[tr, it2]
            avgD += _N.sqrt(_N.dot(df, df))

        avgDs[it1] = avgD
            
        
