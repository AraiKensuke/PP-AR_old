import numpy as _N
cimport numpy as _N
cimport cython
import time as _tm

"""
c functions
"""
cdef extern from "math.h":
    double sqrt(double)

#Ik = _N.identity(9)
#IkN= _N.tile(Ik, (750, 1, 1))



###  Most expensive operation here is the SVD
@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def BSvec_cmp_various(double[:, ::1] F, long N, long k, double[:, ::1] GQGT, double[:, :, ::1] fx, double[:, :, ::1] fV, double[:, ::1] smXN):
    #  Backward sampling.
    #
    #  1)  find covs - only requires values calculated from filtering step
    #  2)  Only p,p-th element in cov mat is != 0.
    #  3)  genearte 0-mean normals from variances computed in 2)
    #  4)  calculate means, add to 0-mean norms (backwds samp) to p-th component
    # 

    cdef int n, i, j, ii, jj, nKK, nK, ik, n_m1_KK, i_m1_K, iik, kmk, km1, kp1, np1k
    cdef double trm1, trm2, trm3, c, Fs

    Ik      = _N.identity(k)
    IkN   =  _N.tile(Ik, (N+1, 1, 1))
    smX   = _N.empty((N+1, k))   #  where to store our samples
    smX_nv   = _N.empty((N+1, k))   #  where to store our samples
    smX_ram   = _N.empty((N+1, k))   #  where to store our samples


    smX[N] = smXN[0]
    smX_nv[N] = smXN[0]
    smX_ram[N] = smXN[0]
    cdef double[:, ::1] smXmv = smX   #  memory view
    cdef double[:, ::1] smX_rammv = smX_ram   #  memory view
    cdef double* p_smX_ram = &smX_rammv[0, 0]
    cdef double[:, :, ::1] fxmv = fx   #  memory view
    cdef double* p_fx = &fxmv[0, 0, 0]

    fFT     = _N.empty((N+1, k, k))    
    _N.dot(fV, F.T, out=fFT)  # dot([N+1 x k x k], [k, k])
    FfFTr     = _N.empty((k, k, N+1))
    _N.dot(F, fFT.T, out=FfFTr)
    iv     = _N.linalg.inv(FfFTr.T + _N.tile(GQGT, (N+1,1,1)))
    ifV    = _N.linalg.inv(fV)

    cdef double[:, :, ::1] ifVmv = ifV
    cdef double* p_ifV         = &ifVmv[0, 0, 0]
    A      = _N.empty((N+1, k, k))      # "nik,nkj->nij", fFT, iv

    for j in xrange(N+1):
       _N.dot(fFT[j], iv[j], out=A[j])

    INAF   = IkN*1.000001 - _N.dot(A, F)
    ####  EINSUM slow when mat x mat
    PtN    = _N.empty((N+1, k, k))# "nik,nkj->nij", INAF, fV
    for j in xrange(N+1):
       _N.dot(INAF[j], fV[j], out=PtN[j])

    ####   ANALYTICAL.  
    ppth_cov    = _N.empty(N+1)# "nik,nkj->nij", INAF, fV
    iF          = _N.linalg.inv(F)
    cdef double[:, ::1] iFmv = iF
    cdef double*        p_iF = &iFmv[0, 0]
    cdef double iF_p1_2     = iF[k-1, 0]*iF[k-1, 0]
    cdef double q2          = GQGT[0, 0]

    for j in xrange(N+1):
        ppth_cov[j] = (q2*iF_p1_2)/(1+q2*ifV[j, k-1, k-1]*iF_p1_2)

    ##  NOW multivariate normal
    #t1 = _tm.time()
    mvn1   = _N.random.randn(N+1, k)  #  slightly faster

    C       = _N.linalg.cholesky(PtN)    ###  REPLACE svd with Cholesky
    zrmn   = _N.einsum("njk,nk->nj", C, mvn1)
    cdef double[:, ::1] zrmnmv = zrmn
    #t2 = _tm.time()
    #print (t2-t1)

    #  out of order calculation.  one of the terms can be calculated
    INAFfx = _N.einsum("nj,nj->n", INAF[:, k-1], fx[:, :, 0])
    cdef double[::1] INAFfxmv = INAFfx
    last   = _N.zeros(k)
    last[k-1] = 1

    #  temp storage
    Asx = _N.empty(k)
    cdef double[::1] Asxmv = Asx



    t1 = _tm.time()
    """
    ##  analyticaly calculated
    for n from N > n >= 0:
        c = 1 + q2*ifV[n, k-1, k-1]*iF_p1_2
        smX_ram[n, 0:k-1] = smX_ram[n+1, 1:k] 

        Fikixi = _N.dot(iF[k-1], smX_ram[n+1]) #  (F^{-1})_{ki} * x_i
        smX_ram[n, k-1] = Fikixi * (1 - (ifV[n, k-1,k-1]*q2*iF[k-1, 0]*iF[k-1, 0])/c)
        for i in xrange(0, k-1):
            smX_ram[n, k-1] -= ((q2*iF[k-1, 0]*iF[k-1, 0])/c)*smX_ram[n+1, i+1]*ifV[n, k-1,i]

        smX_ram[n, k-1]  += (q2*iF[k-1,0]*iF[k-1,0]/c) * _N.dot(ifV[n, k-1], fx[n]) #+ _N.sqrt(ppth_cov[n])*_N.random.randn()
    """


    ##  ptrs for ifV, smX_ram, iF, fx
    ###  analytical method.  only update 1 
    kmk = (k-1)*k
    km1 = k-1
    kp1 = k+1

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

        p_smX_ram[nK + km1]= Fs - q2*iF_p1_2*(trm1 + trm2 - trm3)/c

    t2 = _tm.time()
    ###  Naive method
    for n from N > n >= 0:
        _N.dot(A[n], smX[n+1], out=Asx)
        #smXmv[n, k-1] = zrmnmv[n, k-1] + Asxmv[k-1] + INAFfxmv[n]
        smXmv[n, k-1] = Asxmv[k-1] + INAFfxmv[n]
        for i from 0 <= i < k-1:
            #smXmv[n, i] = zrmnmv[n, i] + Asxmv[i]
            smXmv[n, i] = Asxmv[i]
    t3 = _tm.time()

    for n from N > n >= 0:
        Ax    = _N.dot(A[n], smX_nv[n+1])
        #print Ax.shape
        #print (Ik - _N.dot(A[n], F)).shape
        #print fx[n].shape
        smX_nv[n] = Ax + _N.dot(Ik - _N.dot(A[n], F), fx[n, :, 0])
    t4 = _tm.time()

    ##############

    print "*************"
    print "(t2-t1)  %.4e" % (t2-t1)
    print "(t3-t2)  %.4e" % (t3-t2)
    print "(t4-t3)  %.4e" % (t4-t3)
    print smX_nv[100]
    print smX[100]
    print smX_ram[100]
    print smX_nv[300]
    print smX[300]
    print smX_ram[300]
    print "!!!!!!!!!!!!!"


    return smX
