def bootstrap_filter(Y, N, T, alpha, b, sigmasq, delta, mult=1):
    '''
    Implement boostrap filter, as described in Doucet, de Freitas & Gordon, 2001.
    Returns particles, one-step ahead distribution, filtering distribution, log marginal likelihood p(y_{1:t})
    and normalized weights.
    See Andrieu et al,2010 for estimation of marginal likelihood p(y_{1:t}).
    This code was used for simulated decoding in Sec. 6.3, and assumes all initial hidden states are 0.
    '''
    
    log_marginal_likelihood = 0.0
    particles = np.zeros((N, T+1))
    #particles[:,0] = norm.rvs(scale=np.sqrt(scale**2/(1-alpha**2)), size=N)
    particles[:,0] = np.zeros((N,))
    preds, estimates = [], []
    pred_dist = np.zeros((N, T))
    pred_dist[:,0] = alpha*particles[:,0]+norm.rvs(scale=np.sqrt(sigmasq), size=N)
    resampled = False
    prev_weights = np.ones((N,))/N
    for t in range(1,T+1):
        
        weights = np.ones((N,))/N
        for i in range(N):
            prev_x = particles[i, t-1]
            rand_eps = norm.rvs(scale=np.sqrt(sigmasq))
            curr_x = alpha*prev_x + rand_eps
            particles[i, t] = curr_x

            cif = b*np.exp(mult*curr_x)
            weights[i] = bernoulli.pmf(Y[t-1], p=min(1, cif*delta))
        
        normalized_weights = weights/np.sum(weights)
        log_marginal_likelihood += np.log(np.mean(weights))
        estimates.append(np.dot(normalized_weights, particles[:,t]))  # filtering distribution
        # Now resample
        particles_before_resampling = particles[:, range(t+1)]    # Use range in order to create copy of array!
        for j in range(N):
            msk = np.random.choice(range(N), p=normalized_weights, replace=True)
            particles[j, 0:t+1] = particles_before_resampling[msk, 0:t+1]
        
        # Compute the one-step ahead prediction distribution of the next hidden state.
        if t<T:
            pred_dist[:,t] = alpha*particles[:,t]+norm.rvs(scale=np.sqrt(sigmasq), size=N)
            preds.append(np.mean(pred_dist[:,t]))

    return particles, pred_dist, estimates, log_marginal_likelihood, normalized_weights
