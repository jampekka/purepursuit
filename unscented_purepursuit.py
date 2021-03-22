import numpy as np
import numba

jit = numba.njit
#jit = lambda f: f

# For compatibility with immutable APIs like JAX
@jit
def up(x, i, v): x[i] = v; return x

from scipy.stats import multivariate_normal

@jit
def normal_logpdf(x, m, s):
    return np.log(1/np.sqrt(2*np.pi)) - ((x - m)/s)**2/2.0

@jit
def multivariate_normal_logpdf(x, mu, S):
    nx = len(S)
    norm_coeff = nx*np.log(2*np.pi)+np.linalg.slogdet(S)[1]

    err = x-mu
    numerator = np.linalg.solve(S, err).T.dot(err)

    return -0.5*(norm_coeff+numerator)


@jit
def julier_sigma_weights(n):
    kappa = 3 - n
    alpha = 1e-3
    beta = 2.0
    lambda_ = alpha**2 * (n + kappa) - n
    
    c = 0.5/(n + lambda_)
    Wm = np.full(2*n+1, c)
    Wc = np.full(2*n+1, c)
    Wm = up(Wm, 0, lambda_/(n + lambda_))
    Wc = up(Wc, 0, lambda_/(n + lambda_) + (1 - alpha**2 + beta))
 
    
    return Wm, Wc

@jit
def julier_sigma_points(m, c):
    n = len(m)
    kappa = 3 - n
    alpha = 1e-3
    beta = 2.0
    lambda_ = alpha**2 * (n + kappa) - n
    
    U = np.linalg.cholesky((lambda_ + n) * c).T
    
    m = m.reshape(1, -1)
    sigmas = np.vstack((
        m,
        m + U,
        m - U
    ))

    return sigmas

@jit
def unscented_transform(sigmas, Wm, Wc):
    m = Wm@sigmas
    y = sigmas - m.reshape(1, -1)
    P = y.T@np.diag(Wc)@y
    # Force positive semidefinite
    P = (P + P.T)/2
    return m, P


@jit
def circle_pursuit_yawrate(preview, origin, radius, speed, p_x, p_y, heading):
    d_t = preview
    o_x, o_y = origin
    r = np.abs(radius)
    direction = np.sign(radius)
    v = speed
    prev_t = d_t
    preview_angle = prev_t*v/r
           
    proj_angle = np.arctan2(p_y - o_y, p_x - o_x)
    target_angle = proj_angle - direction*preview_angle
    r_x = r*np.cos(target_angle) + o_x
    r_y = r*np.sin(target_angle) + o_y

    r_x -= p_x
    r_y -= p_y
    t_x = r_x*np.cos(heading) - r_y*np.sin(heading)
    t_y = r_x*np.sin(heading) + r_y*np.cos(heading)

    target_yawrate = 2*speed*t_x/(t_x**2 + t_y**2)
    
    return target_yawrate

@jit
def step_circular_steering(dt, origin, radius, noise, time_constant, motor_time_constant, preview, speed, m, c):
    # TODO: Doing this in polar coordinates would probably give a nicer
    # approximation
    # TODO: Do full state augmentation
    alpha = 1 - np.exp(-dt/time_constant)
    motor_alpha = 1 - np.exp(-dt/motor_time_constant)
    Wm, Wc = julier_sigma_weights(len(m))
    sigmas = julier_sigma_points(m, c)
    x, z, heading, yr, desired_correction, error = sigmas.T
    
    desired_correction += alpha*(error - desired_correction)
    yr += motor_alpha*desired_correction

    x += np.sin(heading)*speed*dt
    z += np.cos(heading)*speed*dt
    heading += yr*dt
    
    ideal = circle_pursuit_yawrate(preview, origin, radius, speed, x, z, heading)
    error = ideal - yr
    sigmas = np.column_stack((x, z, heading, yr, desired_correction, error))

    m, c = unscented_transform(sigmas, Wm, Wc)
    #c[-1, -1] += noise**2
    c = up(c, (-1, -1), c[-1, -1] + noise**2)
    
    return m, c

@jit
def simulate_circular_steering(dt, N, origin, radius, noise, time_constant, motor_time_constant, preview, motor_noise, speed, x, z, heading, yr):
    states = np.zeros((N, 4))
    alpha = 1 - np.exp(-dt/time_constant)
    motor_alpha = 1 - np.exp(-dt/motor_time_constant)
    
    error = 0.0
    desired_correction = np.random.randn()*motor_noise
    for i in range(N):
        desired_correction += alpha*(error - desired_correction)
        yr += motor_alpha*desired_correction

        x += np.sin(heading)*speed*dt
        z += np.cos(heading)*speed*dt
        heading += yr*dt
        ideal = circle_pursuit_yawrate(preview, origin, radius, speed, x, z, heading)
        error = ideal - yr + np.random.randn()*noise

        states[i] = x, z, heading, yr

    return states

@jit
def circular_steerer(dt, origin, radius, noise, time_constant, motor_time_constant, preview, motor_noise, speed, x, z, heading, yr):
    Ndim = 6 # Brutally incorporating the error in the state
    m = np.zeros(Ndim)
    c = np.eye(Ndim)*1e-9 # Hack to avoid issues in the first step
    c[-2,-2] = motor_noise**2
    #m[:4] = x, z, heading, yr
    m = up(m, slice(0, 4), (x, z, heading, yr))
    
    while True:
        m, c = step_circular_steering(dt, origin, radius, noise, time_constant, motor_time_constant, preview, speed, m, c)
        yield m, c

@jit
def circular_steering(dt, N, origin, radius, noise, time_constant, motor_time_constant, preview, motor_noise, speed, x, z, heading, yr):
    Ndim = 6 # Brutally incorporating the error in the state
    m = np.zeros(Ndim)
    c = np.eye(Ndim)*1e-9 # Hack to avoid issues in the first step
    c[-2,-2] = motor_noise**2
    #m[:4] = x, z, heading, yr
    m = up(m, slice(0, 4), (x, z, heading, yr))
    
    ms = np.zeros((N, Ndim))
    cs = np.zeros((N, Ndim, Ndim))
    for i in range(N):
        m, c = step_circular_steering(dt, origin, radius, noise, time_constant, motor_time_constant, preview, speed, m, c)
        ms[i] = m
        cs[i] = c
    return ms, cs

@jit
def circular_steering_loglik(dt, origin, radius, noise, time_constant, motor_time_constant, preview, model_error, speed, x, z, heading, yr):
    Ndim = 6 # Brutally incorporating the error in the state
    m = np.zeros(Ndim)
    c = np.eye(Ndim)*1e-9 # Hack to avoid issues in the first step
    c[-2,-2] = model_error**2
    m = up(m, slice(0, 4), (x[0], z[0], heading[0], yr[0]))
    
    observations = np.vstack((x, z, heading, yr)).T

    obsdim = observations.shape[1]
    total = 0.0
    for i in range(1, len(x)):
        try:
            m, c = step_circular_steering(dt, origin, radius, noise, time_constant, motor_time_constant, preview, speed, m, c)
            lik = multivariate_normal_logpdf(observations[i], m[:obsdim], c[:obsdim,:obsdim])
        except:
            # Hack
            total = -np.inf
            break
        #lik = normal_logpdf(observations[i][3], m[3], np.sqrt(c[3,3]))
        total += lik

    return total

