# This file contains all filters used in Section 6.4:
# 1. Bootstrap filter, with importance distribution p(x_{0:t})
# 2. SMC with importance distribution given by one-step step ahead prediction p(x_{t}|y_{1:t-1})
# 3. SMC with linear approximation of the optimal importance distribution p(x_{t}|x_{t-1}, y_{t}) - see Doucet et al, 2000.
# In all three implementations, if weights are all zero then algorithm returns marginal likelihood of -infinity. This is useful when
# run with PMMH, because a candidate with likelihood of -infinity is automatically rejected.

def bootstrap_filter(Y, N, T, alpha, b, sigmasq, delta):
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

            cif = b*np.exp(curr_x)
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


def pred_filter(Y, N, T, alpha, b, scale, delta):
    """
    Implement SMC one-step ahead using predictive distribution as importance function.
    """
    log_marginal_likelihood = 0.0
    particles = np.zeros((N,T+1))
    particles[:,0] = 0
    preds, estimates = [], []
    pred_dist = np.zeros((N, T))
    pred_dist[:,0] = alpha*particles[:,0]+norm.rvs(scale=scale, size=N)
    try:
        for t in range(1,T+1):
            weights = np.ones((N,))/N
            for i in range(N):
                prev_x = particles[i,t-1]
                msk = np.random.choice(range(N))
                curr_x = pred_dist[msk,t-1]
                particles[i,t] = curr_x

                cif = b*np.exp(curr_x)
                weights[i] = poisson.pmf(Y[t-1], mu=cif*delta)#*norm.pdf(curr_x, loc=alpha*prev_x, scale=scale)

            if np.sum(weights)<=0: 
                print('all weights are zero for alpha, b, scale : {}, {}, {}'.format(alpha, b, scale))
                log_marginal_likelihood = -float('inf')
                return particles, pred_dist, estimates, log_marginal_likelihood, weights
            normalized_weights = weights/np.sum(weights)
            log_marginal_likelihood += np.log(np.mean(weights))
            estimates.append(np.dot(normalized_weights, particles[:,t]))
            # Now resample
            particles_before_resampling = particles[:, range(t+1)]    # Use range in order to create copy of array!
            for j in range(N):
                msk = np.random.choice(range(N), p=normalized_weights, replace=True)
                particles[j, 0:t+1] = particles_before_resampling[msk, 0:t+1]

            # Predict 
            if t<T:
                pred_dist[:,t] = alpha*particles[:,t]+norm.rvs(scale=scale, size=N)
                preds.append(np.mean(pred_dist[:,t]))

        return particles, pred_dist, estimates, log_marginal_likelihood, normalized_weights
    
# Implement SMC based on approximation of optimal importance distribution. Auxiliary functions:

def second_derivative(x, b, scale):
    return -b*np.exp(x)-1/scale**2

def first_derivative(x, x_prev, y, alpha, b, scale):
    return y - b*np.exp(x)- (x - alpha*x_prev)/scale**2

def nll_of_optimal(x, x_prev, y, alpha, b, scale):
    return -(y*(np.log(b)+x) - b*np.exp(x) - (x-alpha*x_prev)**2/(2*scale**2))


def opt_approx_filter(Y, N, T, alpha, b, scale, delta):
    '''
    Implement SMC using approximate optimal importance function as described in Doucet et al,2000.
    Note : pred_dist has the same indexing as Y.
    '''
    b = b*delta
    log_marginal_likelihood = 0.0
    particles = np.zeros((N, T+1))
    #particles[:,0] = norm.rvs(scale=np.sqrt(scale**2/(1-alpha**2)), size=N)
    particles[:,0] = 0
    preds, estimates = [], []
    pred_dist = np.zeros((N, T))
    pred_dist[:,0] = alpha*particles[:,0]+norm.rvs(scale=scale, size=N)
    resampled = False
    try:
        for t in range(1,T+1):
            weights = np.ones((N,))/N
            for i in range(N):
                # Compute mode via a numerical optimization algorithm. I used L-BFGS-B because it was faster than others.
                prev_x = particles[i, t-1]
                y_t = Y[t-1]
                nr = minimize(nll_of_optimal, x0=prev_x, method='L-BFGS-B', args = (prev_x, y_t, alpha, b, scale,), 
                              options={'maxiter' : 2000}).x
                
                curr_x = norm.rvs(loc=nr, scale=np.sqrt(-1/second_derivative(nr, b, scale)))
                particles[i, t] = curr_x

                # Evaluate importance weights
                obs_pdf = poisson.pmf(y_t, mu = b*np.exp(curr_x))
                weights[i] = obs_pdf*norm.pdf(curr_x, loc=alpha*prev_x, scale=scale)/norm.pdf(curr_x,
                                                loc=nr, scale=np.sqrt(-1/second_derivative(nr, b, scale)))

            if np.sum(weights)<=0:
                print('all weights are zero for alpha, b, scale : {}, {}, {}'.format(alpha, b, scale))
                log_marginal_likelihood = -float('inf')
                return particles, pred_dist, estimates, log_marginal_likelihood, weights
            normalized_weights = weights/np.sum(weights)
            log_marginal_likelihood += np.log(np.mean(weights))
            estimates.append(np.dot(normalized_weights, particles[:,t]))
            # Now resample
            particles_before_resampling = particles[:, range(t+1)]    # Use range in order to create copy of array!
            for j in range(N):
                msk = np.random.choice(range(N), p=normalized_weights, replace=True)
                particles[j, 0:t+1] = particles_before_resampling[msk, 0:t+1]

            # Predict 
            if t<T:
                pred_dist[:,t] = alpha*particles[:,t]+norm.rvs(scale=scale, size=N)
                preds.append(np.mean(pred_dist[:,t]))

            prev_normalized_weights = normalized_weights.copy()


        return particles, pred_dist, estimates, log_marginal_likelihood, normalized_weights
