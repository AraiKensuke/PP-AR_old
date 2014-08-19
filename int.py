x  = _N.linspace(0, 1, 20, endpoint=False)
xp = _N.linspace(0, 1, 10, endpoint=False)
yp = _N.sin(xp)
y  = _N.interp(x, xp, yp)
