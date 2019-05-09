# Implementation of Particle Marginal Metropolis Hastings as described in Andrieu et al, 2010.
# Uses bootstrap filter for SMC step. Assumes model of equations (6.1) and (6.2).

def PMMH(Y, N, T, delta, alpha_b, beta_b, alpha_scale, beta_scale, scales_alpha, scale_b, scales_scale, MC_iter):
    """
    Particle Marginal Metropolis Hastings using Random Walk Metropolis for the MCMC step and bootstrap filter of Gordon et al,1993 
    for SMC step. 
    All parameters have Gamma priors.
    
    curr_alpha = uniform.rvs(loc=0, scale=1)
    curr_b = gamma.rvs(a=alpha_b, scale=1/beta_b)
    curr_scale = gamma.rvs(a=alpha_scale, scale=1/beta_scale)
    curr_particles, _, _, curr_marg_likelihood, normalized_weights = bootstrap_filter(Y, N, T, curr_alpha, curr_b, curr_scale,delta)
    curr_particle = curr_particles[np.random.choice(range(N), p=normalized_weights)]  # Sample x_{1:t}
    samples_alpha, samples_scale, samples_b = [], [], []
    accepted_alpha, accepted_scale, accepted_b = 0.0, 0.0, 0.0
    samples_particles = np.zeros((MC_iter, T+1))
    
    for i in range(MC_iter):
        if i%5==0:
            print(i, curr_alpha, curr_b, curr_scale)

        # Update alpha
        scale_alpha = np.random.choice(scales_alpha)
        cand_alpha = norm.rvs(scale=scale_alpha) + curr_alpha
        if -1<cand_alpha<1:
            cand_particles, _,_,cand_marg_likelihood, cand_normalized_weights = bootstrap_filter(Y, N, T, cand_alpha, curr_b,
                                                                                            curr_scale, delta) 
            cand_particle = cand_particles[np.random.choice(range(N), p = cand_normalized_weights)]
            ratio1 = min(0, cand_marg_likelihood + gamma.logpdf(cand_alpha, a=1, scale=1/0.75) - 
                        curr_marg_likelihood - gamma.logpdf(curr_alpha, a=1, scale=1/0.75))
            U = uniform.rvs()
            if U <= np.exp(ratio1):
                curr_alpha = cand_alpha
                curr_particle = cand_particle
                curr_marg_likelihood = cand_marg_likelihood
                accepted_alpha += 1
        samples_alpha.append(curr_alpha)

        # Update b
        #scale_b = np.random.choice(scales)
        cand_b = norm.rvs(scale=scale_b) + curr_b
        if cand_b > 0:
            cand_particles, _,_,cand_marg_likelihood, cand_normalized_weights =bootstrap_filter(Y, N, T, curr_alpha, cand_b,
                                                                                                curr_scale, delta) 
            cand_particle = cand_particles[np.random.choice(range(N), p = cand_normalized_weights)]
            ratio2 = min(0, cand_marg_likelihood + gamma.logpdf(cand_b, a=alpha_b, scale=1/beta_b) -
                         curr_marg_likelihood - gamma.logpdf(curr_b, a=alpha_b, scale=1/beta_b))
            U = uniform.rvs()
            if U <= np.exp(ratio2):
                curr_b = cand_b
                curr_particle = cand_particle
                curr_marg_likelihood = cand_marg_likelihood
                accepted_b += 1
        samples_b.append(curr_b)

        # Update sigma
        scale_scale = np.random.choice(scales_scale)
        cand_scale = norm.rvs(scale=scale_scale) + curr_scale
        if cand_scale>0:
            cand_particles, _,_,cand_marg_likelihood, cand_normalized_weights = bootstrap_filter(Y, N, T, curr_alpha, curr_b,
                                                                                            cand_scale, delta) 
            cand_particle = cand_particles[np.random.choice(range(N), p = cand_normalized_weights)]
            ratio3 = min(0, cand_marg_likelihood + gamma.logpdf(cand_scale, a=alpha_scale, scale=1/beta_scale) - 
                       curr_marg_likelihood - gamma.logpdf(curr_scale, a=alpha_scale, scale=1/beta_scale))
            U = uniform.rvs()
            if U <= np.exp(ratio3):
                curr_scale = cand_scale
                curr_marg_likelihood = cand_marg_likelihood
                curr_particle = cand_particle
                accepted_scale += 1
        samples_scale.append(curr_scale)

        samples_particles[i,:] = curr_particle

    plt.figure(figsize=(10,6))
    plt.plot(range(MC_iter), samples_alpha)
    plt.title('Samples of alpha, mean : {}, acceptance : {}'.format(round(np.mean(samples_alpha[int(MC_iter/3):]),3)
                                                                    , accepted_alpha/MC_iter))

    plt.figure(figsize=(10,6))
    plt.plot(range(MC_iter), samples_b)
    plt.title('Samples of b, mean : {}, acceptance : {}'.format(round(np.mean(samples_b[int(MC_iter/3):]),3)
                                                                    , accepted_b/MC_iter))

    plt.figure(figsize=(10,6))
    plt.plot(range(MC_iter), samples_scale)
    plt.title('Samples of scale, mean : {}, acceptance : {}'.format(round(np.mean(samples_scale[int(MC_iter/3):]),3)
                                                                    , accepted_scale/MC_iter))

    return samples_alpha, samples_b, samples_scale, samples_particles
