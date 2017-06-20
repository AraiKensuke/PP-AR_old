def cntmdlMCMCOnly(GibbsIter, iters, u0, rn0, dist, cts, rns, us, dty, xn, stdu=0.5):
    """
    We need starting values for rn, u0, model
    """
    # u0, rn0, dist = startingValues(cts, xoff=xn)
    # print "starting values"
    # print "$$$$$$$$$$$$$$$$$$$$$$$$"
    # print "rn=%d" % rn0
    # print "us=%.3f" % u0
    # print "md=%d" % dist
    # print "$$$$$$$$$$$$$$$$$$$$$$$$"
    # print "in cntmdlMCMCOnly"
    # print rn0
    # print dist
    # print cts
    #  proposal parameters
    stdu2= stdu**2;
    istdu2= 1./ stdu2

    Mk = _N.mean(cts) if len(cts) > 0 else 0  #  1comp if nWins=1, 2comp
    iMk = 1./Mk
    nmin= _N.max(cts) if len(cts) > 0 else 0   #  if n too small, can't generate data
    rmin= 1

    lFlB = _N.empty(2)
    rn1rn0 = _N.empty(2)

    rds  = _N.random.rand(iters)
    rdns = _N.random.randn(iters)

    p0  = 1 / (1 + _N.exp(-u0))
    p0x = 1 / (1 + _N.exp(-(u0+xn)))
    lFlB[1] = Llklhds(dist, cts, rn0, p0x)
    lBg = lFlB[1]

    cross  = False
    lls   = []
    accptd = 0

    llsV = _N.empty(40)
    #rn0 = bestrn(dist, cts, rn0, llsV, p0x)

    #print "uTH is %.3e" % uTH
    for it in xrange(iters):
        #
        if dist == _cd.__BNML__:
            uu1  = -_N.log(rn0 * iMk - 1) # mean of proposal density
            u1 = uu1 + stdu * rdns[it]
            #print "BNML   uu1  %(uu1).3e    u1  %(u1).3e" % {"uu1" : uu1, "u1" : u1}

            if u1 > uTH:       ###########   Stay in Binomial ##########
                todist = _cd.__BNML__;    cross  = False
                p1 = 1 / (1 + _N.exp(-u1))
                p1x = 1 / (1 + _N.exp(-(u1+xn)))
                lmd0= int(Mk/p1)
                rn1 = bestrn(todist, cts, lmd0, llsV, p1x)
                #print "%(1)d   %(2)d" % {"1" : lmd0, "2" : rn1}

                uu0  = -_N.log(rn1 * iMk - 1) # mean of proposal density
                # log of probability
                #print "d1  %(1) .3f    %(2) .3f" % {"1" : (u1 - uu1), "2" : (u0 - uu0)}
                #lpPR = 0.5*istdu2*(-((u1 - uu1)*(u1 - uu1)) + ((u0 - uu0)*(u0 - uu0)))  #  - (lnc0 - lnc1), lnc reciprocal of norm
                lpPR = 0.5*istdu2*(((u1 - uu1)*(u1 - uu1)) - ((u0 - uu0)*(u0 - uu0)))  #  - (lnc0 - lnc1), lnc reciprocal of norm
            else:   ########  Switch to __NBML__  ####################
                print "switch 2 NBML"
                todist = _cd.__NBML__;   cross  = True
                u1 = 2*uTH - u1  #  u1 now a parameter of NB distribution   
                p1 = 1 / (1 + _N.exp(-u1))
                p1x = 1 / (1 + _N.exp(-(u1+xn)))
                lmd0 = int((1./p1 - 1)*Mk)
                rn1 = bestrn(todist, cts, lmd0, llsV, p1x)

                uu0  = -_N.log(rn1 * iMk) # mean of proposal density
                #lpPR = 0.5*istdu2*(-(((uTH-u1) - uu1)*((uTH-u1) - uu1)) + (((uTH-u0) - uu0)*((uTH-u0) - uu0)))
                lpPR = 0.5*istdu2*((((uTH-u1) - uu1)*((uTH-u1) - uu1)) - (((uTH-u0) - uu0)*((uTH-u0) - uu0)))
        elif dist == _cd.__NBML__:
            uu1  = -_N.log(rn0 * iMk) # mean of proposal density
            u1 = uu1 + stdu * rdns[it]
            #print "NBML   uu1  %(uu1).3e    u1  %(u1).3e" % {"uu1" : uu1, "u1" : u1}

            if u1 > uTH:       ######   Stay in Negative binomial ######
                todist = _cd.__NBML__;    cross  = False
                p1 = 1 / (1 + _N.exp(-u1))
                p1x = 1 / (1 + _N.exp(-(u1+xn)))
                lmd0 = int((1./p1 - 1)*Mk)
                #print "lmd0   %d" % lmd0
                rn1 = bestrn(todist, cts, lmd0, llsV, p1x)

                #rn1 = trPoi(lmd, rmin)   #  mean is p0/Mk
                # bLargeP = (p0 > 0.3) and (p1 > 0.3)
                # if bLargeP:#    fairly large p.  Exact proposal ratio
                #     lmd= Mk*((1-0.5*(p0+p1))/(0.5*(p0+p1)))
                # else:          #  small p.  prop. ratio far from lower lim of n
                #     lmd1= Mk*((1-p1)/p1);  lmd0= Mk*((1-p0)/p0);   lmd= lmd1
                uu0  = -_N.log(rn1 * iMk) # mean of proposal density
                # log of probability

                #lpPR = 0.5*istdu2*(-((u1 - uu1)*(u1 - uu1)) + ((u0 - uu0)*(u0 - uu0)))
                lpPR = 0.5*istdu2*(((u1 - uu1)*(u1 - uu1)) - ((u0 - uu0)*(u0 - uu0)))
            else:   ########  Switch to __BNML__  ####################
                print "switch 2 BNML"
                todist = _cd.__BNML__;    cross  = True
                u1 = 2*uTH - u1  #  u in NB distribution
                p1 = 1 / (1 + _N.exp(-u1))
                p1x = 1 / (1 + _N.exp(-(u1+xn)))
                lmd0= int(Mk/p1)
                rn1 = bestrn(todist, cts, lmd0, llsV, p1x)

                #lmd = Mk/p1
                #rn1 = trPoi(lmd, nmin)   #  mean is p0/Mk
                #lmd1= Mk/p1;     lmd0= Mk*((1-p0)/p0);     lmd = lmd1
                uu0  = -_N.log(rn1 * iMk - 1) # mean of proposal density
                lpPR = 0.5*istdu2*(-(((uTH-u1) - uu1)*((uTH-u1) - uu1)) + (((uTH-u0) - uu0)*((uTH-u0) - uu0)))
                #lpPR = 0.5*istdu2*((((uTH-u1) - uu1)*((uTH-u1) - uu1)) - (((uTH-u0) - uu0)*((uTH-u0) - uu0)))

        lFlB[0] = Llklhds(todist, cts, rn1, p1x)
        #print "proposed state  ll  %(1).3e   old state  ll  %(2).3e     new-old  %(3).3e" % {"1" : lFlB[0], "2" : lFlB[1], "3" : (lFlB[0] - lFlB[1])}

        rn1rn0[0] = rn1;                   rn1rn0[1] = rn0

        ########  log of proposal probabilities

        lnPR = 0    #  we have the log part set to 1.  No change
        lPR = lnPR + lpPR
        lposRat = lFlB[0] - lFlB[1]
        lrat = lPR + lposRat
        # if lPR > 100:
        #     prRat = 2.7e+43
        # else:
        #     prRat = _N.exp(lPR)

        #  lFlB[0] - lFlB[1] >> 0  -->  new state has higher likelihood
        #posRat = 1.01e+200 if (lFlB[0] - lFlB[1] > 500) else _N.exp(lFlB[0]-lFlB[1])

        #print "posRat %(1).3e     prRat %(2).3e" % {"1" : posRat, "2" : prRat}
        #print "lrat is %f" % lrat
        #rat  = _N.exp(lrat)

        aln   = 1 if (lrat > 0) else _N.exp(lrat)
        #aln  = rat if (rat < 1)  else 1   #  if aln == 1, always accept
        if rds[it] < aln:   #  accept
            accptd += 1
            u0 = u1
            rn0 = rn1
            p0 = p1
            lFlB[1] = lFlB[0]
            #lls.append(lFlB[1])
            #print "accepted  %d" % it
            dist = todist
        lls.append(lFlB[1])

        dty[it] = dist
        us[it] = u0
        rns[it] = rn0    #  rn0 is the newly sampled value if accepted

    print "accepted %d" % accptd
    # fig = _plt.figure()
    # _plt.plot(llsV)
    # _plt.suptitle(accptd)
    # _plt.savefig("llsV%d" % GibbsIter)
    # _plt.close()
    lEn = lFlB[0]

    #print "ll Bg %(b).3e   ll En %(e).3e" % {"b" : lBg, "e" : lEn}
    return u0, rn0, dist
