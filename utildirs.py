import os

def setFN(fn, dir=None, create=False):
    """
    for programs run from the result directory
    """
    rD = ""

    if dir != None:
        lvls = dir.split("/")
        for lvl in lvls:
            rD += "%s/" % lvl
            if not os.access("%s" % rD, os.F_OK) and create:
                os.mkdir(rD)
        return "%(rd)s%(fn)s" % {"rd" : rD, "fn" : fn}
    return fn
