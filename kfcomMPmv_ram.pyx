import numpy as _N
cimport numpy as _N
cimport cython
import time as _tm

"""
c functions
"""
cdef extern from "math.h":
    double sqrt(double)


###  Most expensive operation here is the SVD
@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def BSvec(double[:, ::1] iF, long N, long k, double q2, double[:, :, ::1] fx, double[:, :, ::1] fV, double[:, ::1] smXN):
    #  Backward sampling.
    #
    #  1)  find covs - only requires values calculated from filtering step
    #  2)  Only p,p-th element in cov mat is != 0.
    #  3)  genearte 0-mean normals from variances computed in 2)
    #  4)  calculate means, add to 0-mean norms (backwds samp) to p-th component
    # 

    cdef int n, i, j, ii, jj, nKK, nK, ik, n_m1_KK, i_m1_K, iik, kmk, km1, kp1, np1k
    cdef double trm1, trm2, trm3, c, Fs

    kmk = (k-1)*k
    km1 = k-1
    kp1 = k+1
    kk  = k*k

    smX_ram   = _N.empty((N+1, k))   #  where to store our samples

    smX_ram[N] = smXN[0]
    cdef double[:, ::1] smX_rammv = smX_ram   #  memory view
    cdef double* p_smX_ram = &smX_rammv[0, 0]
    cdef double[:, :, ::1] fxmv = fx   #  memory view
    cdef double* p_fx = &fxmv[0, 0, 0]

    ifV    = _N.linalg.inv(fV)

    cdef double[:, :, ::1] ifVmv = ifV
    cdef double* p_ifV         = &ifVmv[0, 0, 0]
    ####   ANALYTICAL.  
    nz_vars    = _N.empty(N+1)# "nik,nkj->nij", INAF, fV
    cdef double[::1]      nz_vars_mv  = nz_vars
    cdef double*  p_nz_vars  = &nz_vars_mv[0]
    cdef double*        p_iF = &iF[0, 0]
    cdef double iF_p1_2     = iF[k-1, 0]*iF[k-1, 0]

    norms = _N.random.randn(N+1)
    cdef double[::1] normsmv = norms
    cdef double* p_norms = &normsmv[0]

    for j in xrange(N+1):
        p_nz_vars[j] = sqrt((q2*iF_p1_2)/(1+q2*p_ifV[j*kk + kmk + km1]*iF_p1_2))*p_norms[j]

    ##  ptrs for ifV, smX_ram, iF, fx
    ###  analytical method.  only update 1 

    for n from N > n >= 0:
        nKK = n*k*k
        nK  = n*k
        np1k = (n+1)*k

        c = 1 + q2*p_ifV[nKK + kmk + km1]*iF_p1_2

        trm2 = 0
        for ik in xrange(km1):  #  shift
            p_smX_ram[nK + ik] = p_smX_ram[np1k + ik+1]
            trm2 += p_smX_ram[np1k + ik+1]*p_ifV[nKK + kmk + ik]

        #####
        Fs = 0
        trm3 = 0
        for ik in xrange(k):
            Fs += p_iF[kmk + ik]*p_smX_ram[np1k+ ik]
            trm3 += p_fx[nK + ik]*p_ifV[nKK + kmk + ik]
        trm1 = Fs*p_ifV[nKK + kmk+ km1]

        p_smX_ram[nK + km1]= Fs - q2*iF_p1_2*(trm1 + trm2 - trm3)/c + p_nz_vars[n]

    return smX_ram


