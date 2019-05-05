# This is the code used for simulated decoding of the triangle wave in Sec. 6.3

def triangle_wave(harmonics, tvals):
    signal = []
    for t in tvals:
        signal.append(np.sum([(-1)**i*(2*i+1)**(-2)*np.sin((2*i+1)*t) for i in range(harmonics)]))
    return signal
  
  
tvals = np.arange(0, 20, 0.02)
tri = triangle_wave(50, tvals) + norm.rvs(scale=0.1, size=len(tvals))
model = AR(tri)
model = model.fit(maxlag=1)   # fit AR(1) model

# Estimate noise term in the hidden state evolution process, see (6.1).
est_var = np.sum((model.fittedvalues-tri[1:])**2)/len(model.fittedvalues) 
