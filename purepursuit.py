import numpy as np
import numba
from collections import namedtuple
import linestring

Path = namedtuple("Path", ["points", "distance"])

@numba.njit
def tangent_ttlc_old(pos, heading, speed, path, minimum=0.3):
    s = (path[:,0] - pos[0]) + 1.0j*(path[:,1] - pos[1])
    s *= np.exp(heading*1j)
    l = np.diff(s)
    s = s[:-1]
    
    a = -np.real(s)/np.real(l)
    d = np.imag(l)*a + np.imag(s)

    d[(a < 0) | (a > 1) | (d < 0)] = np.inf
    t = d/speed
    t[t < minimum] = np.inf
    return np.nanmin(t)

@numba.njit
def tangent_ttlc(pos, heading, speed, path, offset=0.0, minimum=0.3):
    ss = (path[:,0] - pos[0]) + 1.0j*(path[:,1] - pos[1])
    ss *= np.exp(heading*1j)

    ss -= offset
    #l = np.diff(s)
    #s = s[:-1]
    
    min_t = np.inf
    for i in range(len(ss) - 1):
        s = ss[i]
        e = ss[i+1]
        l = e - s

        if np.real(l) == 0: continue
        a = -np.real(s)/np.real(l)
        if a < 0 or a > 1: continue
        d = np.imag(l)*a + np.imag(s)
        t = d/speed
        if t < minimum: continue
        if t >= min_t: continue
        min_t = t
    return min_t

from numpy import sin, pi, real as re, imag as im, sqrt, log, arctan as atan
@numba.njit
def circle_ttlc(pos, heading, speed, yr, path, minimum=0.1):
    if yr == 0:
        return tangent_ttlc(pos, heading, speed, path, minimum=minimum)
    ss = (path[:,0] - pos[0]) + 1.0j*(path[:,1] - pos[1])
    #s = path - pos
    ss *= np.exp(heading*1j)
    R = speed/yr
    
    pm = np.array([-1.0, 1.0])
    min_t = np.inf
    for i in range(len(ss) - 1):
        s = ss[i]
        e = ss[i+1]
        d = e - s
        
        # Simplify this mess!
        ts = 2*atan((R*re(d) + pm*sqrt(R**2*re(d)**2 - 2*R*re(d)*im(d)*im(s) + 2*R*re(s)*im(d)**2 - re(d)**2*im(s)**2 + 2*re(d)*re(s)*im(d)*im(s) - re(s)**2*im(d)**2))/(2*R*im(d) + re(d)*im(s) - re(s)*im(d)))/yr
        for t in ts:
            if t != t: continue
            if t < 0:
                t += np.abs((2*np.pi)/yr)
            if t < minimum: continue
            if t > min_t: continue
            p = R*(1 - np.exp(-1j*t*yr))
            a = re((p - s)/d)
            if a < 0 or a > 1: continue 
            min_t = t

    return min_t

    """
    s, e = ss[:-1], ss[1:]
    d = e - s
    R = speed/yr
    pm = np.array([-1.0, 1.0]).reshape(-1, 1)
    # OMG! Simplify this mess!
    t = 2*atan((R*re(d) + pm*sqrt(R**2*re(d)**2 - 2*R*re(d)*im(d)*im(s) + 2*R*re(s)*im(d)**2 - re(d)**2*im(s)**2 + 2*re(d)*re(s)*im(d)*im(s) - re(s)**2*im(d)**2))/(2*R*im(d) + re(d)*im(s) - re(s)*im(d)))/yr
    t[t < 0] += np.abs((2*np.pi)/yr)
    t[t < minimum] = np.inf

    #p = R*(1 - cos(t*yr)) * I*R*sin(t*yr)
    p = R*(1 - np.exp(-1j*t*yr))
    a = re((p - s)/d)
    t[np.isnan(t) | (np.abs(a - 0.5) > 0.5)] = np.inf
    return np.min(t)
    """

@numba.njit
def preview_time(pos, heading, speed, yr, edges):
    # TODO: Take minimum from fov!
    #ttlc = min([circle_ttlc(pos, heading, speed, yr, edge) for edge in edges])
    ttlc = min([tangent_ttlc(pos, heading, speed, edge) for edge in edges])
    maxt = (np.pi/4)/(np.abs(yr) + 1e-9)
    #maxt = 2.5
    return min(ttlc, maxt)

@numba.njit
def pursuit_yawrate(preview, pos, heading, speed, trajectory):
    #dist = trajectory.project(pos)
    #preview_point = trajectory.interpolate(dist + preview*speed)
    p, d = trajectory
    dist = linestring.linestring_project_dist(p, d, pos)
    preview_point = linestring.linestring_interpolate(p, d, dist + preview*speed)
    
    r_x, r_y = (preview_point - pos).T

    t_x = r_x*np.cos(heading) - r_y*np.sin(heading)
    t_y = r_x*np.sin(heading) + r_y*np.cos(heading)

    target_yawrate = 2*speed*t_x/(t_x**2 + t_y**2)

    return target_yawrate

@numba.njit
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



@numba.njit
def desired_yawrate(pos, heading, speed, yr, midline, edges):
    preview = preview_time(pos, heading, speed, yr, edges)
    yr = pursuit_yawrate(preview, pos, heading, speed, midline)
    return yr

@numba.njit
def alpha_to_time_constant(alpha, dt):
    return dt/np.log(-1/(alpha - 1))

@numba.njit
def time_constant_to_alpha(tc, dt):
    return 1 - np.exp(-dt/tc)

max_yr = np.radians(35.0)
@numba.njit
def _simulate_pursuiter(dt, sensor_alpha, motor_alpha, noise, midline, edges, pos, heading, speed, yawrate):
    pos = pos.copy()
    t = 0.0
    
    sensor_tc = alpha_to_time_constant(sensor_alpha, dt)
    print(sensor_tc)


    desired_correction = 0.0
    error = 0.0
    while True:
        ideal = desired_yawrate(pos, heading, speed, yawrate, midline, edges)
        error = ideal - yawrate + np.random.randn()*noise
        desired_correction += sensor_alpha*(error - desired_correction)
        correction = desired_correction
        yield t, pos.copy(), heading, speed, yawrate, ideal
        
        # https://pdfs.semanticscholar.org/c9c4/aa52f9e1739bf926bfcffffbd6f6f12a852c.pdf
        #motor_tc < sqrt(t_h)*sqrt(tau)*asin((t_h - tau)/(t_h + tau))
        #preview = preview_time(pos, heading, speed, yawrate, edges)
        #motor_tc = np.sqrt(preview*sensor_tc)*np.arcsin((preview - sensor_tc)/(preview + sensor_tc))
        #motor_alpha = time_constant_to_alpha(motor_tc*0.5, dt)
        yawrate += motor_alpha*correction
        
        yawrate = min(max_yr, max(-max_yr, yawrate))

        direction = np.array([np.sin(heading), np.cos(heading)])
        pos += direction*speed*dt
        heading += yawrate*dt
        t += dt

def simulate_pursuiter(dt, sensor_alpha, motor_alpha, noise, track, pos, heading, speed, yawrate):
    midline = Path(track.midpath.points, track.midpath.dist)
    edges = tuple(track.edges)
    yield from _simulate_pursuiter(dt, sensor_alpha, motor_alpha, noise, midline, edges, pos, heading, speed, yawrate)

@numba.njit
def simulate_pursuiter_N(dt, N, sensor_alpha, motor_alpha, noise, midline, edges, pos, heading, speed, yawrate):
    data = []
    pursuiter = _simulate_pursuiter(dt, sensor_alpha, motor_alpha, noise, midline, edges, pos, heading, speed, yawrate)
    points, dists = midline
    for i in range(N):
        t, pos, heading, speed, yawrate, ideal = next(pursuiter)
        data.append((t, pos.copy(), heading, speed, yawrate, ideal))
    
    return data

