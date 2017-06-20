import pickle

N   = 200000
lfi    = _N.empty(N)     # log n!  lfi[0] = log 0!.  lfi[1] = log 1!...
logInts = _N.empty(N)     # log n

for n in xrange(1, N):    
    logInts[n] = _N.log(n)

for n in xrange(1, N):
    lfi[n] = _N.sum(logInts[1:n+1])
lfi[0] = 0   #  0! == 1.   log(0!) = 0


dmp = open("logfact.dump", "wb")
pickle.dump(lfi, dmp, -1)
dmp.close()

