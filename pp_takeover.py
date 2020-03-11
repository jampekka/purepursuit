import math
import numpy as np
import numba

@numba.njit
def stdnormpdf(x):
    return 1/np.sqrt(2*np.pi)*np.exp(-x**2/2.0)

@numba.njit
def stdnormcdf(x):
    return (1 + math.erf(x/np.sqrt(2.0)))/2.0

@numba.njit
def trunc_norm_approx(m, s, a, b):
    alpha = (a - m)/s
    beta = (b - m)/s
    Z = stdnormcdf(beta) - stdnormcdf(alpha)
    Z = max(1e-30, Z)
    shift = (stdnormpdf(alpha) - stdnormpdf(beta))/Z

    mean = m + shift*s

    var = s**2*(1 + (alpha*stdnormpdf(alpha) - beta*stdnormpdf(beta))/Z - shift**2)
    var = max(0.0, var)
    return mean, np.sqrt(var)

@numba.njit
def takeover_liks(tots, ts, yr, dyr, alpha, noise, threshold, lag):
    m = 0.0
    s = 1e-6
    alive = 1.0
    T = threshold
    
    tot_liks = np.zeros_like(tots)
    last_tot = np.max(tots)
    
    lags = 1/np.sqrt(6)

    def lagcdf(t):
        if t <= 0.0: return 0.0
        return stdnormcdf((np.log(t) - np.log(lag))/lags)
    
    def lagpdf(t):
        if t <= 0.0: return 0.0
        return stdnormpdf((np.log(t) - np.log(lag))/lags)

    #def lagcdf(t):
    #    return stdnormcdf(t/lag)

    for i in range(len(dyr) - 1):
        if alive <= 0: break
        if s <= 0: break
        if ts[i] >= last_tot: break
        crossers = stdnormcdf((-T - m)/s) + (1 - stdnormcdf((T - m)/s))*alive
        alive -= crossers

        dt = (ts[i+1] - ts[i])
        cross_lik = crossers
        for j, tot in enumerate(tots):
            delta = tot - ts[i]
            tot_liks[j] += (lagcdf(delta + dt) - lagcdf(delta))*cross_lik/dt

        # TODO: Approximating the resulting d + e distribution
        #   would probably be somewhat more accurate
        m, s = trunc_norm_approx(m, s, -T, T)

        # x1 = alpha*(z1 + noise) + (1 - alpha)*x0
        # x1 = alpha*z1 + alpha*noise + (1 - alpha)*x0
        # x1 = alpha*(z1 - x0) + x1 + alpha*noise

        e = dyr[i] - yr[i]
        m = alpha*e + (1 - alpha)*m
        s = np.sqrt((alpha*noise)**2 + ((1 - alpha)*s)**2)
    
    return tot_liks

def takeover_liks_reparam(tots, ts, yr, dyr, alpha, std, threshold, lag):
    v = std**2
    v_noise = v*(2 - alpha)/alpha
    noise = np.sqrt(v_noise)

    return takeover_liks(tots, ts, yr, dyr, noise, alpha, threshold, lag)

def demo():
    import matplotlib.pyplot as plt
    
    dt = 1/60
    ts = np.arange(0, 20, 1/60)
    yr = np.zeros(len(ts))
    dyr = yr + np.radians(2.0)
    
    alpha = 0.05
    noise = np.radians(1.0)
    threshold = np.radians(1.0)
    lag = 0.1
    alpha, noise, threshold, lag = [0.01915796, 0.04981239, 0.02607559, 0.69824366]

    pdf = takeover_liks(ts, ts, yr, dyr, alpha, noise, threshold, lag)
    print(np.sum(pdf*dt))
    plt.plot(ts, pdf)
    plt.show()

if __name__ == '__main__':
    demo()
