import pickle as _pkl

def pickleForLater():
    global restarts, use_prior, bMW, freq_lims, Cs, Cn, C, R, _t0, _t1, u, q2, F0, F_alfa_rep, smpx, ws, x00, V00, fSigMax, _d, wts, uts, Bsmpx, all_alfas, x
    #  what do I need?
    restarts           += 1
    these = dict()
    these["use_prior"] = use_prior
    these["bMW"]       = bMW
    these["freq_lims"] = freq_lims
    these["Cs"]        = Cs
    these["Cn"]        = Cn
    these["C"]         = C
    these["R"]         = R
    these["_t0"]       = _t0
    these["_t1"]       = _t1

    #  Gibbs variables
    these["u"]         = u
    these["q2"]        = q2
    these["F0"]        = F0
    these["F_alfa_rep"]= F_alfa_rep

    #  hidden variables
    these["smpx"]      = smpx
    these["ws"]        = ws
    #these["x00"]       = x00
    #these["V00"]       = V00
    these["restarts"]  = restarts
    these["fSigMax"]   = fSigMax

    fp= open(resFN("%s_restart.pkl" % baseFN, dir=setdir), "wb")
    _pkl.dump(these, fp)
    fp.close()

    # ddN= _d.N
    # pick = dict()
    # pick["wts"] = wts
    # pick["uts"] = uts
    # pick["Bsmpx"] = Bsmpx
    # pick["R"]   = R
    # pick["C"]   = C
    # pick["ddN"]   = _d.N
    # pick["allalfas"]   = allalfas
    # pick["x"]   = x
    # fp= open(resFN("%s_comps.pkl" % baseFN, dir=setname), "wb")
    # _pkl.dump(pick, fp)
    # fp.close()

def unpickle(rsdir, rs):
    #  what do I need?
    fp= open(resFN("rs=%d_restart.pkl" % rs, dir=rsdir), "rb")
    these = _pkl.load(fp)
    fp.close()
    global restarts, use_prior, bMW, freq_lims, Cs, Cn, C, R, _t0, _t1, u, q2, F0, F_alfa_rep, smpx, ws, x00, V00, k, fSigMax

    use_prior          = these["use_prior"]
    bMW                = these["bMW"]
    freq_lims          = these["freq_lims"]
    Cs                 = these["Cs"]
    Cn                 = these["Cn"]
    C                  = these["C"]
    R                  = these["R"]
    _t0                = these["_t0"]
    _t1                = these["_t1"]

    k                  = 2*C+R
    #  Gibbs variables
    u                  = these["u"]
    q2                 = these["q2"]
    F0                 = these["F0"]
    F_alfa_rep         = these["F_alfa_rep"]

    #  hidden variables
    smpx               = these["smpx"]
    ws                 = these["ws"]
    #x00                = these["x00"]
    #V00                = these["V00"]
    restarts           = these["restarts"]
    fSigMax            = these["fSigMax"]



def unpickleComps(pickleFN):
    #  what do I need?
    fp= open(resFN("comps", dir=setname), "rb")
    pick = _pkl.load(fp)
    wts = pick["wts"]
    uts = pick["uts"]
    Bsmpx = pick["Bsmpx"]
    R = pick["R"]
    C = pick["C"]
    ddN = pick["ddN"]
    x = pick["x"]
    allalfas = pick["allalfas"]
    fp.close()
