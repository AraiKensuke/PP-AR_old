import patsy

x = _N.linspace(0., 1000, 1000)

B = patsy.bs(x, df=6)    #  spline basis

