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


def simulate_expHawkes(lmbd, a, b, T, plot=False):
    """
    Simulate Hawkes process of parameters lmbd (lambda), a, b on interval [0,T] using thinning algorithm of Ogata,1981.
    The activation function will be exponential, see eq. 4.3.
    Setting plot=True will also return a plot of the point process realization.
    """
    t, simulated_Hawkes, unifs = 0, [], []
    eps = 10**(-10)  # epsilon
    while t < T:
        M = exp_Hawkes_CIF(lmbd, a, b, simulated_Hawkes, t+eps)
        E = np.random.exponential(scale=1/M)
        t += E 
        U = uniform.rvs(loc=0, scale=M)
        if t<T and U <= exp_Hawkes_CIF(lmbd, a, b, simulated_Hawkes, t):
            simulated_Hawkes.append(t)
            unifs.append(U)
    if plot==True:
        plt.figure(figsize=(12,8))
        plt.scatter(simulated_Hawkes, unifs, label='Accepted', c='green')
        for i in range(len(unifs)):
            plt.vlines(x = simulated_Hawkes[i], ymin=0, ymax=unifs[i])
        tvals_for_base = np.arange(0, simulated_Hawkes[0], 0.05)
        plt.plot(tvals_for_base, np.ones(len(tvals_for_base),)*lmbd, color='blue')
        for k in range(len(simulated_Hawkes)-1):
            tvals_per_arrival = np.arange(simulated_Hawkes[k], simulated_Hawkes[k+1], 0.05)
            lambda_vals = [exp_Hawkes_CIF(lmbd, a, b, simulated_Hawkes[:k+1], tvals_per_arrival[j]) for j in range(len(tvals_per_arrival))]
            plt.plot(tvals_per_arrival, lambda_vals, color='blue')
        plt.legend(loc='best')  
        plt.xlabel('x')
        plt.ylabel(r'$\lambda(t)$')
    return simulated_Hawkes  
    
