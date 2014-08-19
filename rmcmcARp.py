#  The   #  use this as a launch pad
import mcmcARp

burn = 100
NMC  = 400
model= "bernoulli"
_t0  = 0
_t1  = 1000
ID_q2= False

#  setname figured out from where I am being run 
#  sampled parameters and ut, wt
mcmcARp.run()
