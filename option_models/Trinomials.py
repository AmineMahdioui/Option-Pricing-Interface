import math,scipy as stats,numpy as np
from scipy import stats
def kamrad_ritchken(s, k, t, v, rf, cp, am=False, n=100,return_trees=False):
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
    #Basic calculations
    h = t/n
    mu=rf-v**2/2
    Lambda= math.sqrt(1/(1-1/3))
    alpha= math.sqrt(1+h*(mu/v)**2)
    u = np.exp(v*alpha*Lambda * np.sqrt(h))
    d = 1.0 / u
    drift = math.exp(rf*h)
    pu=1/(2*Lambda**2)+mu*np.sqrt(h)/(2*Lambda*v)
    pd=1/(2*Lambda**2)-mu*np.sqrt(h)/(2*Lambda*v)
    pm=1-1/Lambda**2

    #Process the terminal stock price
    stkval = np.zeros((n+1,2*n+1))
    optval = np.zeros((n+1,2*n+1))
    stkval[0,0] = s
    for i in range(n+1):
        for j in range(2*i + 1):
            stkval[i, j] = s * u**(i-j if (i>=j) else 0) * d**((j-i) if (i<j) else 0)

    #Backward recursion for option price
    for j in range(n+1):
        optval[n,j] = max(0,cp*(stkval[n,j]-k))
    for i in range(n-1,-1,-1):
        for j in range(2*i+1):
            optval[i,j] = (pu*optval[i+1,j]+pm*optval[i+1,j+1]+pd*optval[i+1,j+2]) /drift
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

    print(kamrad_ritchken(S, K, T, r, sigma,-1,n=3,return_trees=True))
