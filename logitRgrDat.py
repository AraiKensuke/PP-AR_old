from kassdirs import resFN

setname="rgr1ar"
###  Generate data for logit regression analysis

N   = 1000
B   = _N.array([2.2, -1.8, -1.1])    #  last component the offset 
k   = len(B)                         #  size k includes offset term
xyp = 0.3*_N.random.randn(N, k+2)    #  explanatory data
xyp[:, k-1] = 1                      #  last component the offset
Bxi = _N.dot(B, xyp[:, 0:k].T)
eBxi= _N.exp(Bxi)
rs  = _N.random.rand(N)
xyp[:, k+1]  = eBxi / (1 + eBxi)
xyp[:, k]   = _N.array(rs < xyp[:, k+1], dtype=_N.int)   #  turn into 0s and 1s


sfmt=""
for ik in xrange(k - 1):
    sfmt += "% .3e  "
sfmt += "%d  %d  %.3e"   #  offset component, response variable, prob

_N.savetxt(resFN("data.dat", dir=setname, create=True), xyp, fmt=sfmt)
