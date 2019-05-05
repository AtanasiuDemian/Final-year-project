# Implement Gibbs sampler for latent variable Hawkes model described in Sec. 4.2

def sample_omega(lmbd, a, b, arrivals, T):
    """
    Perform updating step (4.6)
    """
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
    
def prod_for_b(b, arrivals, t_n, n, omega):  # auxiliary function 
    output = 0.0
    for j in range(n, len(arrivals)):
        if omega[j]==n:
            output += np.log(b) - b*(arrivals[j]-t_n)
    return output

def LatHawkes(T, arrivals, alpha_lmbd, beta_lmbd, alpha_a, beta_a, alpha_b, beta_b, scale_b, N=10000):
    """
    Perform Gibbs sampling on parameters lambda, a, b using Gamma priors.
    Parameters a and lambda are sampled directly from their corresponding posterior distributions.
    Parameter b is samples via a Random Walk Metropolis step 
    T = length of observation
    arrivals = dataset of spikes
    alpha_lmbd, beta_lmbd, alpha_a, beta_a, alpha_b, beta_b = parameters of Gamma priors.
    scale_b = standard deviation of MCMC proposal for sampling b.
    After sampling the program returns histograms and traceplots (see Fig. 4.2) and samples without burn-in.
    """
    
    # Initial values are sampled from the prior distributions.
    curr_lmbd = gamma.rvs(a=alpha_lmbd, scale=1/beta_lmbd)
    curr_a = gamma.rvs(a=alpha_a, scale=1/beta_a)
    curr_b = gamma.rvs(a=alpha_b, scale=1/beta_b)
    curr_omega = sample_omega(curr_lmbd, curr_a, curr_b, arrivals, T)
    
    accepted_b = 0.0
    samples_lmbd, samples_a, samples_b = [], [], []
    
    for i in range(N):
            # First update lambda
            cand_lmbd = gamma.rvs(a=alpha_lmbd+np.sum(curr_omega==0), scale=1/(beta_lmbd+T))
            curr_lmbd = cand_lmbd
            samples_lmbd.append(curr_lmbd)

            # Update b
            cand_b = norm.rvs(loc=0, scale=scale_b) + curr_b
            if cand_b>0:
                b_post_at_curr = np.sum([curr_a*(np.exp(-curr_b*(T-arrivals[n]))-1)+prod_for_b(
                    curr_b, arrivals, arrivals[n], n+1, curr_omega) for n in range(len(arrivals))]) + gamma.logpdf(
                curr_b, a=alpha_b, scale=1/beta_b)
                b_post_at_cand = np.sum([curr_a*(np.exp(-cand_b*(T-arrivals[n]))-1)+prod_for_b(
                    cand_b, arrivals, arrivals[n], n+1, curr_omega) for n in range(len(arrivals))]) + gamma.logpdf(
                cand_b, a=alpha_b, scale=1/beta_b)
                ratio1 = min(0, b_post_at_cand - b_post_at_curr)
                U = uniform.rvs()
                if U <= np.exp(ratio1):
                    curr_b = cand_b
                    accepted_b += 1
            samples_b.append(curr_b)

            # Update a
            new_alpha = np.sum([np.sum(curr_omega==j+1) for j in range(len(arrivals))])
            new_beta = np.sum([1-np.exp(-curr_b*(T-t)) for t in arrivals])
            cand_a = gamma.rvs(a=new_alpha + alpha_a, scale = 1/(new_beta + beta_a))
            curr_a = cand_a
            samples_a.append(curr_a)

            # Update omega
            cand_omega = sample_omega(curr_lmbd, curr_a, curr_b, arrivals, T)
            curr_omega = cand_omega

            if i%100==0:
                print(i)

    fig, ax = plt.subplots(3,2, figsize=(15,12))
    ax[0,0].hist(samples_b, ec='black', alpha=0.6)
    ax[0,0].set_title(r'Samples of $b$')
    ax[0,0].axvline(x=b, color='red')
    ax[0,0].text(-0.1, 1.05, string.ascii_uppercase[0], transform=ax[0,0].transAxes,
            size=20, weight='bold')

    ax[0,1].plot(range(N), samples_b)
    ax[0,1].set_title(r'Samples of $b$, mean : {}, acceptance: {}'.format(round(np.mean(samples_b[int(N/3):]),3), accepted_b/N))
    ax[0,1].text(-0.1, 1.05, string.ascii_uppercase[1], transform=ax[0,1].transAxes,
            size=20, weight='bold')

    ax[1,0].hist(samples_a, ec='black', alpha=0.6)
    ax[1,0].set_title(r'Samples of $a$')
    ax[1,0].axvline(x=a, color='red')
    ax[1,0].text(-0.1, 1.05, string.ascii_uppercase[2], transform=ax[1,0].transAxes,
            size=20, weight='bold')

    ax[1,1].plot(range(N), samples_a)
    ax[1,1].set_title(r'Samples of $a$, mean : {}'.format(round(np.mean(samples_a[int(N/3):]),3)))
    ax[1,1].text(-0.1, 1.05, string.ascii_uppercase[3], transform=ax[1,1].transAxes,
            size=20, weight='bold')

    ax[2,0].hist(samples_lmbd, ec='black', alpha=0.6)
    ax[2,0].axvline(x=lmbd, color='red')
    ax[2,0].set_title(r'Samples of $\lambda$')
    ax[2,0].text(-0.1, 1.05, string.ascii_uppercase[4], transform=ax[2,0].transAxes,
            size=20, weight='bold')

    ax[2,1].plot(range(N), samples_lmbd)
    ax[2,1].set_title(r'Samples of $\lambda$, mean : {}'.format(round(np.mean(samples_lmbd[int(N/3):]),3)))
    ax[2,1].text(-0.1, 1.05, string.ascii_uppercase[5], transform=ax[2,1].transAxes,
            size=20, weight='bold')

    return samples_lmbd, samples_a, samples_b
