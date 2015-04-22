import pickle
import numpy as _N
import re as _re

from kassdirs import resFN, datFN

from mcmcARpFuncs import loadL2, runNotes
import kfardat as _kfardat

import utilities as _U

import ARlib as _arl
import kfARlibMPmv as _kfar
from ARcfSmpl import ARcfSmpl, FilteredTimeseries

import commdefs as _cd

from ARcfSmplFuncs import ampAngRep, buildLims, FfromLims, dcmpcff, initF
import os

class mcmcAR:
    #  Simulation params
    processes     = 1
    setname       = None
    rs            = -1
    bFixF         = False
    burn          = None;    NMC           = None
    t0            = None;    t1            = None
    useTrials     = None;    restarts      = 0

    #  Description of model
    model         = None
    rn            = None    #  used for count data
    k             = None

    Bsmpx         = None
    smp_q2        = None
    smp_x00       = None

    dt            = None

    y             = None
    kp            = None

    x             = None   #  true latent state

    q2            = None;    q20           = None

    smpx          = None
    ws            = None
    x00           = None
    V00           = None

    #  
    _d            = None

    ##### PRIORS
    #  q2  --  Inverse Gamma prior
    a_q2         = 1e-1;          B_q2         = 1e-6
    #  initial states
    u_x00        = None;          s2_x00       = None


