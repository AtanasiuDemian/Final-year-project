# Implement the bootstrap filter (Gordon et al,1993), which uses prior p(x_{0:t}) as importance function and resamples particles at each 
time step.

def bootstrap_filter(Y, N, T, alpha, b, sigmasq, delta, mult=1, time_sections=[]):
    '''
    Implement boostrap filter, as described in Doucet, de Freitas & Gordon, 2001.
    Returns resampled particles after N iterations, marginal likelihood p(y_{1:t})
    '''
    hists, h = np.zeros((len(time_sections), N)), 0
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
#         if t%50 == 0:
#             print(t)
            
        weights = np.ones((N,))/N
        for i in range(N):
            prev_x = particles[i, t-1]
            rand_eps = norm.rvs(scale=np.sqrt(sigmasq))
            curr_x = alpha*prev_x + rand_eps
            particles[i, t] = curr_x

            # Evaluate importance weights
#             if resampled==True or t==1:
#                 weights[i] = poisson.pmf(Y[t-1], mu=np.exp(b+curr_x))
#             else:
#                 weights[i] = prev_weights[i]*poisson.pmf(Y[t-1], mu=np.exp(b+curr_x))
            cif = b*np.exp(mult*curr_x)
            #cif = b*pillow_function(curr_x)
            weights[i] = bernoulli.pmf(Y[t-1], p=min(1, cif*delta))
        
#         if np.sum(weights)==0:
#             try:
#                 weights = prev_weights.copy()
#             except UnboundLocalError:
#                 weights = np.ones((N,))
        resampled = False
        normalized_weights = weights/np.sum(weights)
        log_marginal_likelihood += np.log(np.mean(weights))
        estimates.append(np.dot(normalized_weights, particles[:,t]))
        #ess = 1/np.sum(normalized_weights**2)
        # Now resample
        particles_before_resampling = particles[:, range(t+1)]    # Use range in order to create copy of array!
        for j in range(N):
            msk = np.random.choice(range(N), p=normalized_weights, replace=True)
            particles[j, 0:t+1] = particles_before_resampling[msk, 0:t+1]
        #prev_weights = normalized_weights.copy()
        
        # Incorporate predictions. 
#         if t==1:
#             preds.append(np.mean(norm.rvs(scale=np.sqrt(sigmasq), size=N)+alpha*np.ones((N,))*x_0))
#         elif 2<=t<=T:
#             xvals = np.arange(-7,7,0.05)
#             filt_dist_pdf = [filter_dist(x, normalized_weights, particles[:,t-1], 1) for x in xvals]
#             filt_dist_samples = accept_reject(xvals, filt_dist_pdf, N)
#             preds.append(np.mean(norm.rvs(scale=np.sqrt(sigmasq), size=N)+alpha*filt_dist_samples))
        # Predict 
        if t<T:
            pred_dist[:,t] = alpha*particles[:,t]+norm.rvs(scale=np.sqrt(sigmasq), size=N)
            preds.append(np.mean(pred_dist[:,t]))
            if t in time_sections:
                hists[h,:] = pred_dist[:,t]
                h+=1

    return particles, pred_dist, estimates, log_marginal_likelihood, normalized_weights, hists

