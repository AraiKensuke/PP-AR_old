import numpy as _N
import commdefs as _cd
from ARcfSmpl import ARcfSmpl

from filter import gauKer, lpFilt, bpFilt
from kassdirs import resFN
import re as _re
import os


def loadDat(setname, model, nStates, nWins, rn, t0=0, t1=None):  ##########
    #  READ parameters of generative model from file
    #  contains N, k, singleFreqAR, u, beta, dt, stNz
    x_st_cnts = _N.loadtxt(resFN("cnt_data.dat", dir=setname))

    N   = x_st_cnts.shape[0] - 1
    if t1 == None:
        t1 = N + 1
    cols= x_st_cnts.shape[1]
    x   = x_st_cnts[t0:t1, 0]
    if nStates == 2:
        mH  = x_st_cnts[t0:t1, 1]
        zT  = x_st_cnts[t0:t1, 2]
    else:
        mH  = x_st_cnts[t0:t1, 1:3]   #  assumes 3 states
        zT  = x_st_cnts[t0:t1, 3]     

    if (type(rn) == int) or (nWins == 1):  #  USE only 1 window
        nWins = 1
        if type(oWin) == int:   
            y   = x_st_cnts[t0:t1, oWin+2]   #  STILL ONLY 2 states
        else:
            y   = x_st_cnts[t0:t1, 3] + x_st_cnts[t0:t1, 4]
    else:                      #  USE only vector window
        nWins = 2
        if nStates == 2:
            y   = x_st_cnts[t0:t1, 3:]
        else:
            y   = x_st_cnts[t0:t1, 4:]
    x   = x_st_cnts[t0:t1, 0]

    if nWins == 1:
        mnCt= _N.mean(y)
    else:
        mnCt_w1= _N.mean(y[t0:t1, 0])
        mnCt_w2= _N.mean(y[t0:t1, 1])

    #  INITIAL samples
    if model=="negative binomial":
        if nWins == 1:
            kp   = (y - rn) *0.5
            p0   = mnCt / (mnCt + rn)       #  matches 1 - p of genearted
        else:
            kp_w1   = (y[t0:t1, 0] - rn[0]) *0.5
            kp_w2   = (y[t0:t1, 1] - rn[1]) *0.5
            p0_w1   = mnCt_w1 / (mnCt_w1 + rn[0])
            p0_w2   = mnCt_w2 / (mnCt_w2 + rn[0])
    else:
        if nWins == 1:
            kp  = y - rn*0.5
            p0   = mnCt / float(rn)       #  matches 1 - p of genearted
        else:
            kp_w1  = y[t0:t1, 0] - rn[0]*0.5
            kp_w2  = y[t0:t1, 1] - rn[1]*0.5
            p0_w1  = mnCt_w1 / float(rn[0])       #  matches 1 - p of genearted
            p0_w2  = mnCt_w2 / float(rn[1])       #  matches 1 - p of genearted
    #  gnerate approximate offset
    if nWins == 1:
        u0  = _N.log(p0 / (1 - p0))    #  -1*u generated
    else:
        u0_w1  = _N.log(p0_w1 / (1 - p0_w1))    #  -1*u generated
        u0_w2  = _N.log(p0_w2 / (1 - p0_w2))    #  -1*u generated

    if nWins == 2:
        return N, x, y, kp_w1, kp_w2, u0_w1, u0_w2
    else:
        return N, x, y, kp, u0
