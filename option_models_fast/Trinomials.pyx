import math
import numpy as np
cimport numpy as np

cpdef kamrad_ritchken(double s, double k, double t, double v, double rf, int cp, bint am=False, int n=100, bint return_trees=False):
    """Price an option using the kamrad-ritchken binomial model.
    
    s : initial stock price
    k : strike price
    t : expiration time
    v : volatility
    rf : risk-free rate
    cp : +1/-1 for call/put
    am : True/False for American/European
    n : trinomial steps
    """
    # Basic calculations
    cdef double h = t/n
    cdef double mu = rf - v**2/2
    cdef double Lambda = math.sqrt(1/(1-1/3))
    cdef double alpha = math.sqrt(1 + h*(mu/v)**2)
    cdef double u = np.exp(v*alpha*Lambda * np.sqrt(h))
    cdef double d = 1.0 / u
    cdef double drift = math.exp(rf*h)
    cdef double pu = 1/(2*Lambda**2) + mu*np.sqrt(h)/(2*Lambda*v)
    cdef double pd = 1/(2*Lambda**2) - mu*np.sqrt(h)/(2*Lambda*v)
    cdef double pm = 1 - 1/Lambda**2

    # Process the terminal stock price
    cdef np.ndarray[double, ndim=2] stkval = np.zeros((n+1, 2*n+1), dtype=np.float64)
    cdef np.ndarray[double, ndim=2] optval = np.zeros((n+1, 2*n+1), dtype=np.float64)
    stkval[0,0] = s
    cdef int i, j
    for i in range(n+1):
        for j in range(2*i + 1):
            stkval[i, j] = s * u**(i-j if (i>=j) else 0) * d**((j-i) if (i<j) else 0)

    # Backward recursion for option price
    for j in range(n+1):
        optval[n,j] = max(0, cp*(stkval[n,j]-k))
    for i in range(n-1, -1, -1):
        for j in range(2*i+1):
            optval[i,j] = (pu*optval[i+1,j] + pm*optval[i+1,j+1] + pd*optval[i+1,j+2]) / drift
            if am:
                optval[i,j] = max(optval[i,j], cp*(stkval[i,j]-k))
    if return_trees:
        return {"stock_tree": stkval, "option_tree": optval}
    return optval[0,0]