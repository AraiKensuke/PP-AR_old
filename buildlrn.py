exf("gibbs.py")
lbd2 = _N.array([0.001, 0.08, 0.35, 1.2, 1.5, 1.1, 1.01, 1.])     #  use this for 080402,0,121

N    = 10000
y    = _N.zeros(N, dtype=_N.int)
r    = 0.05
for n in xrange(N):
    if _N.random.rand() < r:
        y[n] = 1

lrn = build_lrnLambda2(N, y, lbd2)
