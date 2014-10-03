import scipy.stats as _ss

def trPoi(lmd, a, b):
    """
    a, b inclusive
    """
    ct = a - 1
    while (ct < a) or (ct > b):
        ct = _ss.poisson.rvs(lmd)
    return ct

pT = 0.4
nT = 100
N  = 1000
order=1

#  create data
cts = _N.random.binomial(nT, pT, size=N)

pMin=0.001
pMax=0.99
Mk  = _N.mean(cts)
iMk100 = int(100*Mk)
sdk = _N.std(cts)
cv = ((sdk*sdk) / Mk)
nmin= _N.max(cts)

stdp = 0.05
stdp2= stdp**2

burn = 100
NMC  = 100

n0   = int(Mk*2)
p0   = Mk / n0



p1s = _N.empty(burn + NMC)
n1s = _N.empty(burn + NMC)

#trncnrms = _ss.truncnorm.rvs(pMin, pMax, size=burn+NMC)
for order in [1, 2]:
    print "^^^^^"
    for it in xrange(burn + NMC):
        if order == 1:
            #############  n jump
            lmd= Mk/p0

            #  proposal distribution is Poisson with mean (p0/Mk), truncated to a,b
            n1 = trPoi(lmd, a=nmin, b=iMk100)   #  mean is p0/Mk
            #  based on n1, we pick
            pu = Mk/n1
            p1 = pu + stdp*_ss.truncnorm.rvs((pMin-pu)/stdp, (pMax-pu)/stdp)
        else:  #  why is behavior
            pu = Mk/n0
            p1 = pu + stdp*_ss.truncnorm.rvs((pMin-pu)/stdp, (pMax-pu)/stdp)

            lmd= Mk/p1

            #  proposal distribution is Poisson with mean (p0/Mk), truncated to a,b
            n1 = trPoi(lmd, a=nmin, b=iMk100)   #  mean is p0/Mk
            #  based on n1, we pick

        p1s[it] = p1
        n1s[it] = n1

    print _N.mean(trncnrms)
    print _N.mean(n1s)
    print _N.mean(p1s)
    print _N.mean(p1s*n1s)

