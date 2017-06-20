import cntUtil as cU

N         = 100
rn        = 130
u         = 0.01
p         = 1 / (1 + _N.exp(-u))
x         = _N.zeros(N)
model     = 1

if model == 1:
    cts       = _N.random.binomial(rn, p, size=N)
elif model == 2:
    cts       = _N.random.negative_binomial(rn, p, size=N)

iters     = 1000

dty       = _N.empty(iters, dtype=_N.int)
us        = _N.empty(iters)
rns       = _N.empty(iters, dtype=_N.int)

#  initial values
mdl0      = 2
u0        = 0.3
rn0       = 200
cU.cntmdlMCMCOnly(100, iters, u0, rn0, mdl0, cts, rns, us, dty, x)

#  
#  Fano factor

