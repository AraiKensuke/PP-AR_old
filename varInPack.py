import numpy as _N
import logerfc as _lfc

x         =  None

def run():
    global x
    x = _N.random.rand(5)
    _lfc.init()
