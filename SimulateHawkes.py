# Simulate Hawkes process by thinning using algorithm of Ogata,1981.

def Hawkes_CIF(lmbd, a, b, arrivals, t):
    """
    Computes conditional intensity function of process of parameters lambda,a,b for dataset 'arrivals' at time t.
    """
    f = lambda t,arriv : a*b*np.exp(-b*(t-arriv))
    return lmbd + np.sum(f(np.ones(len(arrivals),)*t, arrivals))
    
def log_prod_CIF(lmbd, a, b, arrivals):   # auxiliary function
    return np.sum([np.log(Hawkes_CIF(lmbd, a, b, arrivals[:j], arrivals[j])) for j in range(len(arrivals))])

def log_exp_term(a, b, T, arrivals):    # auxiliary function
    return -a*len(arrivals) + a*np.sum([np.exp(-b*(T-arr)) for arr in arrivals])


def simulate_expHawkes(lmbd, a, b, T):
