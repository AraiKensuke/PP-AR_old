import inspect
import re

def parseFN(baseFN):
    """
    (np)/(wp)_tr0-tr1_Cn_R
    """
    FN         = baseFN.split("/")[-1]

    #p          = re.compile("([a-zA-Z]+)_(\d+)-(\d+)_(\d+)_(\d+)")
    p          = re.compile("(\w+)_(\d+)-(\d+)_(\d+)_(\d+)")
    m = p.match(FN)
    
    return m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4)), int(m.group(5))
