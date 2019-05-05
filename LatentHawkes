# Implement Gibbs sampler for latent variable Hawkes model described in Sec. 4.2

def sample_omega(lmbd, a, b, arrivals, T):
    '''
    
    samples = []
    for i in range(len(arrivals)):
        if i==0:
            samples.append(0)
        else:
            support = [lmbd]
            for j in range(i):
                support.append(a*b*np.exp(-b*(arrivals[i]-arrivals[j])))
            samples.append(np.random.choice(range(i+1), p = np.array(support)/np.sum(support)))
    return np.array(samples)
    
def prod_for_b(b, arrivals, t_n, n, omega):
    output = 0.0
    for j in range(n, len(arrivals)):
        if omega[j]==n:
            output += np.log(b) - b*(arrivals[j]-t_n)
    return output
