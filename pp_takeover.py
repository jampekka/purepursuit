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
def threshold_norm_approx_slip(m, s, T, p):
    # TODO: Clean up this sympy generated mess
    phi = stdnormpdf
    Phi = stdnormcdf
    sigma0 = s
    mu0 = m
    alpha = (-T - mu0)/sigma0
    beta = (T - mu0)/sigma0
    
    wtf = max(1e-30, (p*Phi(alpha) - p*Phi(beta) + p - Phi(alpha) + Phi(beta)))
    mean = (mu0*p*Phi(alpha) - mu0*p*Phi(beta) + mu0*p - mu0*Phi(alpha) + mu0*Phi(beta) - p*sigma0*phi(alpha) + p*sigma0*phi(beta) + sigma0*phi(alpha) - sigma0*phi(beta))/wtf
    
    wtf = max(1e-30, (p*(2*Phi(alpha) - 1) - p*(2*Phi(beta) - 1) + 2*p - 2*Phi(alpha) + 2*Phi(beta)))
    S = -(2*sigma0*(p - 1)*(2*mu0*(phi(alpha) - phi(beta)) + sigma0*(alpha*phi(alpha) - beta*phi(beta))) + 2*(mu0**2 + sigma0**2)*(p*(Phi(beta) - 1) - p*Phi(alpha) + Phi(alpha) - Phi(beta)))/wtf
    var = S - mean**2
    var = max(0.0, var)
    std = np.sqrt(var)
    
    return mean, std

@numba.njit
def takeover_liks(tots, dt, yr, dyr, time_constant, noise, threshold, lag, bias=0.0):
    m = 0.0
    s = 1e-6 # TODO: Use the asymptotic state variance
    alive = 1.0
    T = threshold
    
    tot_liks = np.zeros_like(tots)
    last_tot = np.max(tots)
    end_t = len(yr)*dt
    
    lags = 1/np.sqrt(6)
    
    # Lognormal
    """
    def lagcdf(t):
        if t <= 0.0: return 0.0
        return stdnormcdf((np.log(t) - np.log(lag))/lags)
    """

    # Folded normal
    #def lagcdf(t):
    #    if t <= 0.0: return 0.0
    #    return math.erf(t/(lag*np.sqrt(2)))
    
    # Normal
    #def lagcdf(t):
    #    return stdnormcdf((t - bias)/lag)
    
    # Rayleigh
    #def lagcdf(t):
    #    if t <= 0.0: return 0.0
    #    return 1 - np.exp(-t**2/(2*lag**2))

    # Exponential
    def lagcdf(t):
        if t <= 0.0: return 0.0
        return 1.0 - np.exp(-t/lag)
    
    for i in range(len(dyr) - 1):
        t = dt*i
        if alive <= 0: break
        if s <= 0: break
        if t >= last_tot: break

        alpha = 1 - np.exp(-dt/time_constant)
        
        #rate = dt/lag
        #slip_p = np.exp(-1.0*rate)

        #crossers = stdnormcdf((-T - m)/s) + (1 - stdnormcdf((T - m)/s))
        crossers = stdnormcdf((-T - m)/s) + (1 - stdnormcdf((T - m)/s))
        crossers *= alive
        #crossers *= (1 - slip_p)
        alive -= crossers

        cross_lik = crossers
        for j, tot in enumerate(tots):
            delta = tot - t
            if tot < end_t:
                tot_liks[j] += (lagcdf(delta) - lagcdf(delta - dt))*cross_lik/dt
            else:
                tot_liks[j] += lagcdf(end_t - t)*cross_lik
            #if delta >= 0 and delta < dt:
            #    tot_liks[j] += cross_lik/dt

        # TODO: Approximating the resulting d + e distribution
        #   would probably be somewhat more accurate
        m, s = trunc_norm_approx(m, s, -T, T)
        #m, s = threshold_norm_approx_slip(m, s, T, slip_p)

        # x1 = alpha*(z1 + noise) + (1 - alpha)*x0
        # x1 = alpha*z1 + alpha*noise + (1 - alpha)*x0
        # x1 = alpha*(z1 - x0) + x1 + alpha*noise
        
        e = dyr[i] - yr[i]
        m = alpha*e + (1 - alpha)*m
        s = np.sqrt((alpha*noise)**2 + ((1 - alpha)*s)**2)
    
    for j, tot in enumerate(tots):
        if tot < end_t: continue
        tot_liks[j] += alive

    return tot_liks

def takeover_liks_reparam(tots, dt, yr, dyr, time_constant, std, threshold, lag, *args, **kwargs):
    v = std**2
    alpha = 1 - np.exp(-dt/time_constant)
    v_noise = v*(2 - alpha)/alpha
    noise = np.sqrt(v_noise)
    return takeover_liks(tots, dt, yr, dyr, time_constant, noise, threshold, lag, *args, **kwargs)

def takeover_liks_kalman(tots, dt, yr, dyr, std_obs, std_trans, threshold, lag):
    v_z = std_obs**2
    v_t = std_trans**2
    v_x = -v_t/2 + v_z/2 + np.sqrt(v_t**2 + 6*v_t*v_z + v_z**2)/2

    alpha = (v_x + v_t)/(v_x + v_t + v_z)
    time_constant = dt/np.log(-1/(alpha - 1))
    #v_noise = v*(2 - alpha)/alpha
    #noise = np.sqrt(v_noise)

    return takeover_liks(tots, dt, yr, dyr, time_constant, std_obs, np.sqrt(v_x)*threshold, lag)

#def reparam_to_kalman(alpha, std_x):
#    v_x = std_x**2

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
