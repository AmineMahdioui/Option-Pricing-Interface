from .Binomials import *
from .Trinomials import *
import math,scipy as stats,numpy as np
from scipy import stats

def black_scholes(s, k, t, v, rf, cp,am=False,n=100):
    """Price an option using the Black-Scholes model.
    
    s : initial stock price
    k : strike price
    t : expiration time
    v : volatility
    rf : risk-free rate
    cp : +1/-1 for call/put
    """
    d1 = (math.log(s/k)+(rf+0.5*math.pow(v,2))*t)/(v*math.sqrt(t))
    d2 = d1 - v*math.sqrt(t)
    optprice = cp*s*stats.norm.cdf(cp*d1) - \
        cp*k*math.exp(-rf*t)*stats.norm.cdf(cp*d2) 
    return optprice

