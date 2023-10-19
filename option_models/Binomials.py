import math,scipy as stats,numpy as np
from scipy import stats

def jarrow_rudd(s, k, t, v, rf, cp, am=False,n=100,return_trees=False):
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
    #Basic calculations
    h = t/n
    u = math.exp((rf-0.5*math.pow(v,2))*h+v*math.sqrt(h))
    d = math.exp((rf-0.5*math.pow(v,2))*h-v*math.sqrt(h))
    drift = math.exp(rf*h)
    q = (drift-d)/(u-d)

    #Process the terminal stock price
    stkval = np.zeros((n+1,n+1))
    optval = np.zeros((n+1,n+1))
    stkval[0,0] = s
    for i in range(n+1):
        for j in range(i + 1):
            stkval[i, j] = s * u**j * d**(i-j)

    #Backward recursion for option price
    for j in range(n+1):
        optval[n,j] = max(0,cp*(stkval[n,j]-k))
    for i in range(n-1,-1,-1):
        for j in range(i+1):
            optval[i,j] = ((1-q)*optval[i+1,j]+q*optval[i+1,j+1])/drift
            if am:
                optval[i,j] = max(optval[i,j],cp*(stkval[i,j]-k))

    if return_trees:
        return {"stock_tree":stkval,"option_tree":optval}
    return optval[0,0]




def cox_ross_rubinstein(s, k, t, v, rf, cp, am=False,n=100,return_trees=False):
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
    #Basic calculations
    h = t/n
    u = math.exp(v*math.sqrt(h))
    d = math.exp(-v*math.sqrt(h))
    drift = math.exp(rf*h)
    q = (drift-d)/(u-d)

    #Process the terminal stock price
    stkval = np.zeros((n+1,n+1))
    optval = np.zeros((n+1,n+1))
    stkval[0,0] = s
    for i in range(1,n+1):
        stkval[i,0] = stkval[i-1,0]*d
        for j in range(1,i+1): 
            stkval[i,j] = stkval[i-1,j-1]*u

    #Backward recursion for option price
    for j in range(n+1):
        optval[n,j] = max(0,cp*(stkval[n,j]-k))
    for i in range(n-1,-1,-1):
        for j in range(i+1):
            optval[i,j] = ((1-q)*optval[i+1,j]+q*optval[i+1,j+1])/drift
            if am:
                optval[i,j] = max(optval[i,j],cp*(stkval[i,j]-k))

    if return_trees:
        return {"stock_tree":stkval,"option_tree":optval}
    return optval[0,0]



if __name__ == "__main__":
    S = 100.
    K = 100.
    T = 0.5
    r = 0.05
    sigma = 0.25

    print(cox_ross_rubinstein(S, K, T, r, sigma,-1,n=3,return_trees=True))
    print(jarrow_rudd(S, K, T, r, sigma,-1,n=3,return_trees=True))
