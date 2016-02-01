import pickle

def lAR(dir):
    dmp = open("../Results/%s/smpls.dump" % dir, "rb")
    lm = pickle.load(dmp)
    dmp.close()
    
    fs   = lm["fs"]
    amps = lm["amps"]

    cmps = fs.shape[1]

    _plt.plot(fs[1:, 0], amps[1:, 0], color="black")
    _plt.plot(fs[-1, 0], amps[-1, 0], color="red", marker="*", ms=15)

    clrs = ["grey", "blue", "orange", "green", "pink", "cyan"]
    
    for i in xrange(1, cmps):
        _plt.plot(fs[:, i], amps[:, i], color=clrs[i-1])
        _plt.plot(fs[-1, i], amps[-1, i], color="red", marker="*", ms=15)

    _plt.ylabel("modulus")
    _plt.xlabel("frequency")

    _plt.savefig("../Results/%s/RootsIter.eps" % dir)
    _plt.close()



for i in xrange(1, 5):
   for j in [1, 3, 5]:
       dir="irreg_%(i)d/mcmc%(j)d" % {"i" : i, "j" : j}
       print dir
       lAR(dir)

#dir="irreg_1o/mcmcR2" % {"i" : i, "j" : j}
#lAR(dir)
