import inspect
import re
import numpy as _N

def parseFN(baseFN):
    """
    (np)/(wp)_tr0-tr1_Cn_R
    """
    FN         = baseFN.split("/")[-1]

    #p          = re.compile("([a-zA-Z]+)_(\d+)-(\d+)_(\d+)_(\d+)")
    p          = re.compile("(\w+)_(\d+)-(\d+)_(\d+)_(\d+)")
    m = p.match(FN)
    
    return m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4)), int(m.group(5))


def parseFNcnt(baseFN):
    """
    go-t0,t1-0,1,2
    (np)/(wp)_tr0-tr1_Cn_R
    """
    FN         = baseFN.split("/")[-1]

    p          = re.compile(".*(\d+),(\d+)-([\d,]+)")
    m = p.match(FN)

    wins2use   = _N.array(list(m.group(3).split(",")), dtype=_N.int)
    return int(m.group(1)), int(m.group(2)), wins2use
