import pickle

def cmpntsAR(dir):
    dmp = open("../Results/%s/mARp.dump" % dir, "rb")
    mARp = pickle.load(dmp)
    dmp.close()
    mARp[0].getComponents
    return mARp[0]
    

#for i in xrange(1, 5):
for i in [1, ]:
   #for j in [1, 3, 5]:
    for j in [5,]:
       dir="irreg_%(i)d/mcmc%(j)d" % {"i" : i, "j" : j}
       mARp = cmpntsAR(dir)

