import time as _tm

logs = _N.log(_N.arange(1, 5003))

N    = 100000

ns   = _N.random.rand(N)

t1   = _tm.time()
for i in xrange(N):
    _N.log(1 + int(5000*ns[i]))
t2   = _tm.time()


t3   = _tm.time()
for i in xrange(N):
    logs[1 + int(5000*ns[i])]
t4   = _tm.time()


print (t2-t1)
print (t4-t3)
