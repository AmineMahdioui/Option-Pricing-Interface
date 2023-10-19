import numpy as np
import math

cimport numpy as np
cimport cython

cdef extern from "math.h" nogil:
    double exp(double)
    double sqrt(double)
    double pow(double, double)
    double fmax(double, double)

@cython.boundscheck(False)  
@cython.wraparound(False)
@cython.cdivision(True)
cpdef jarrow_rudd(double s, double k, double t, double v, double rf, int cp, bint am=False, int n=100, bint return_trees=False):
    """Price an option using the Jarrow-Rudd binomial model.
    
    s : initial stock price
    k : strike price
    t : expiration time
    v : volatility
    rf : risk-free rate
    cp : +1/-1 for call/put
    am : True/False for American/European
    n : binomial steps
    """
    cdef double h, u, d, drift, q
    cdef int i, j, m
    cdef np.ndarray[np.double_t, ndim=2] stkval = np.zeros((n+1,n+1))
    cdef np.ndarray[np.double_t, ndim=2] optval = np.zeros((n+1,n+1))


    #Basic calculations
    h = t/n
    u = exp((rf-0.5*pow(v,2))*h+v*sqrt(h))
    d = exp((rf-0.5*pow(v,2))*h-v*sqrt(h))
    drift = exp(rf*h)
    q = (drift-d)/(u-d)

    #Process the terminal stock price
    stkval[0,0] = s
    for i in range(n+1):
        for j in range(i + 1):
            stkval[i, j] = s * pow(d,(i-j)) * pow(u,j)

    #Backward recursion for option price
    for j in range(n+1):
        optval[n,j] = fmax(0,cp*(stkval[n,j]-k))
    for i in range(n-1,-1,-1):
        for j in range(i+1):
            optval[i,j] = ((1-q)*optval[i+1,j]+q*optval[i+1,j+1])/drift
            if am:
                optval[i,j] = fmax(optval[i,j],cp*(stkval[i,j]-k))

    if return_trees:
        return {"stock_tree":stkval,"option_tree":optval}
    return optval[0,0]

cpdef cox_ross_rubinstein(double s, double k, double t, double v, double rf, int cp, bint am=False, int n=100, bint return_trees=False):
    """Price an option using the Jarrow-Rudd binomial model.
    
    s : initial stock price
    k : strike price
    t : expiration time
    v : volatility
    rf : risk-free rate
    cp : +1/-1 for call/put
    am : True/False for American/European
    n : binomial steps
    """
    cdef double h, u, d, drift, q
    cdef int i, j, m
    cdef np.ndarray[np.double_t, ndim=2] stkval = np.zeros((n+1,n+1))
    cdef np.ndarray[np.double_t, ndim=2] optval = np.zeros((n+1,n+1))

    #Basic calculations
    h = t/n
    u = exp(v*sqrt(h))
    d = exp(-v*sqrt(h))
    drift = exp(rf*h)
    q = (drift-d)/(u-d)

    #Process the terminal stock price
    stkval[0,0] = s
    for i in range(1,n+1):
        stkval[i,0] = stkval[i-1,0]*d
        for j in range(1,i+1):
            stkval[i,j] = stkval[i-1,j-1]*u

    #Backward recursion for option price
    for j in range(n+1):
        optval[n,j] = fmax(0,cp*(stkval[n,j]-k))
    for i in range(n-1,-1,-1):
        for j in range(i+1):
            optval[i,j] = ((1-q)*optval[i+1,j]+q*optval[i+1,j+1])/drift
            if am:
                optval[i,j] = fmax(optval[i,j],cp*(stkval[i,j]-k))

    if return_trees:
        return {"stock_tree":stkval,"option_tree":optval}
    return optval[0,0]


