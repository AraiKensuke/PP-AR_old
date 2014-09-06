#  ISI pctl
pctl0 = _N.array((pctl.shape[0] + 1, 2))
pctl0[0, 0] = 0
pctl0[0, 1] = 0
pctl0[1:, :] = pctl

_N.interp()
