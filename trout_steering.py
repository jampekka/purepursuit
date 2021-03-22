from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal
from scipy.special import logit, expit
from scipy.interpolate import interp1d

from linestring import LineLoop, simplify_linestring, extrude_linestring
from purepursuit import simulate_pursuiter
import purepursuit

@dataclass
class Track:
    midpath: None
    edges: None

def get_track(midline, lanewidth, targets):
    # TODO: Handle repeal
    midpath = LineLoop(midline)

    width = np.zeros(len(midline)) + lanewidth
    # TODO: Nicer normals!
    inside = extrude_linestring(midline, -width.reshape(-1, 1)/2)
    outside = extrude_linestring(midline, width.reshape(-1, 1)/2)

    include = np.ones(len(width), dtype=bool)
    boundaries = []
    targetpoints = []
    for _, target in targets.iterrows():
        pos = target[['xcentre', 'zcentre']].values
        dist = midpath.project(pos)
        r = target.centre_radius
        #include = include & (np.abs(midpath.dist - dist) > 2*r)
        for d in np.array([-r, r]):
            #targetpoints.append([1.2*d + dist, midpath.interpolate(1.2*d + dist), lanewidth])
            targetpoints.append([1.01*d + dist, midpath.interpolate(1.01*d + dist), lanewidth])
            op = midpath.interpolate(d + dist)
            targetpoints.append([d + dist, [pos[0], op[1]], 2*r])

        op = midpath.interpolate(dist)
        boundaries.append(np.array([
            [op[0] - lanewidth/2, pos[1]],
            [pos[0] - r, pos[1]],
        ]))

        boundaries.append(np.array([
            [op[0] + lanewidth/2, pos[1]],
            [pos[0] + r, pos[1]]
                ]))
    
    td, tp, tw = zip(*targetpoints)
    dists = np.concatenate([midpath.dist, td])
    midline = np.concatenate([midpath.points, tp])
    width = np.concatenate([width, tw])

    order = np.argsort(dists)
    midline = midline[order]
    width = width[order]
    orig_midpath = midpath
    midpath = LineLoop(midline)

    track = Track(midpath, [inside, outside, *boundaries])
    track.orig_midpath = orig_midpath
    track.targets = targets
    return track

"""

def tangent_ttlc(pos, heading, speed, path, offset=0.0, minimum=0.1):
    s = (path[:,0] - pos[0]) + 1.0j*(path[:,1] - pos[1])
    s *= np.exp(heading*1j)
    s -= offset
    l = np.diff(s)
    s = s[:-1]
    
    a = -np.real(s)/np.real(l)
    d = np.imag(l)*a + np.imag(s)

    d[(a < 0) | (a > 1) | (d < 0)] = np.inf
    t = d/speed
    t[t < minimum] = np.inf
    return np.nanmin(t)

def circle_ttlc(pos, heading, speed, yr, path, minimum=0.3):
    from numpy import sin, pi, real as re, imag as im, sqrt, log, arctan as atan
    if yr == 0:
        return tangent_ttlc(pos, heading, speed, path, minimum=minimum)
    s = (path[:,0] - pos[0]) + 1.0j*(path[:,1] - pos[1])
    #s = path - pos
    s *= np.exp(heading*1j)
    
    s, e = s[:-1], s[1:]
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

def preview_time(pos, heading, speed, yr, track):
    # TODO: Take minimum from fov!
    ttlc = min(circle_ttlc(pos, heading, speed, yr, edge) for edge in track.edges)
    maxt = (np.pi/4)/(np.abs(yr) + 1e-9)
    return min(ttlc, maxt)

def pursuit_yawrate(preview, pos, heading, speed, trajectory):
    pos = np.atleast_2d(pos)
    dist = trajectory.project(pos)
    preview_point = trajectory.interpolate(dist + preview*speed)
    
    r_x, r_y = (preview_point - pos).T

    t_x = r_x*np.cos(heading) - r_y*np.sin(heading)
    t_y = r_x*np.sin(heading) + r_y*np.cos(heading)

    target_yawrate = 2*speed*t_x/(t_x**2 + t_y**2)

    return np.squeeze(target_yawrate)

def vectorizer(*args, **kwargs):
    return lambda f: np.vectorize(f, *args, **kwargs)

@vectorizer(signature="(2),(),(),()->()", excluded={4})
def desired_yawrate(pos, heading, speed, yr, track):
    preview = preview_time(pos, heading, speed, yr, track)
    yr = pursuit_yawrate(preview, pos, heading, speed, track.midpath)
    return yr

def simulate_pursuiter(dt, sensor_alpha, motor_alpha, noise, track, pos, heading, speed, yawrate):
    pos = np.array(pos, copy=True)
    t = 0.0

    desired_correction = 0.0
    error = 0.0
    while True:
        ideal = desired_yawrate(pos, heading, speed, yawrate, track)
        error = ideal - yawrate + np.random.randn()*noise
        desired_correction += sensor_alpha*(error - desired_correction)
        correction = desired_correction 
        yield t, pos.copy(), heading, speed, yawrate, ideal
        yawrate += motor_alpha*correction
        
        direction = np.array([np.sin(heading), np.cos(heading)])
        pos += direction*speed*dt
        heading += yawrate*dt
        t += dt
"""


def get_condition_track(condition):
    track = pd.read_csv('trout/track_with_edges.csv')
    midline = track[['midlinex', 'midlinez']].values
    targets = pd.read_csv('trout/targets.csv').query('condition == @condition')
    midline = simplify_linestring(midline, 0.05)
    return get_track(midline, 3, targets)


def test_steering():
    track = get_condition_track(1)
    speed = 8
    dt = 1/60
    alpha = 0.058
    #alpha = 0.1
    std = 6.33e-4
    noise = std*np.sqrt((2 - alpha)/alpha)
    print(np.degrees(noise))
    beta = 0.2
    
    alpha = 0.1
    beta = 0.1
    noise = np.radians(3.0)

    pursuiter = simulate_pursuiter(dt, alpha, beta, noise, track, track.midpath.interpolate(0), 0.0, speed, 0.0)
    data = [next(pursuiter) for i in range(int(120/dt))]
    print("Steering")
    data = [next(pursuiter) for i in range(int(120/dt))]
    print("Done")
    data = np.rec.fromrecords(data, dtype=[
        ('ts', float),
        ('pos', float, 2),
        ('heading', float),
        ('speed', float),
        ('yr', float),
        ('dyr', float)
        ])

    dist = track.midpath.project(data.pos)
    dist[np.gradient(dist) < -10] = np.nan
    plt.plot(dist, np.degrees(data.yr))
    
    plt.figure()
    plt.plot(*track.midpath.points.T, 'k-')
    for e in track.edges:
        plt.plot(*e.T, 'k-', alpha=0.5)
   
    plt.plot(data.pos[:,0], data.pos[:,1], alpha=0.7)
    plt.axis('equal')
    plt.show()

def expsmooth(x, alpha, y=0.0):
    ys = np.empty(len(x))
    for i in range(len(x)):
        y = 0.0
        y += alpha*(x[i] - y)
        ys[i] = y
    return ys

import numba

@numba.njit
def actuator(ideals, alpha, beta, desired_correction=0.0, actuated=0.0):
    out = np.empty(len(ideals))
    for i in range(len(ideals)):
        ideal = ideals[i]
        error = ideal - actuated
        desired_correction = (1 - alpha)*desired_correction + alpha*error
        
        actuated += beta*desired_correction
        actuated = min(max(actuated, -np.radians(90)), np.radians(90))
        out[i] = actuated
    return out

def compare_model(data, track, p):
    speed = 8
    dt = 1/60
    for t, td in data.groupby('trialcode'):
        ts = td.currtime.values
        pos = td[['posx_mirror', 'posz_mirror']].values
    
        heading = np.radians(td.yaw.values)
        if td.startingposition.iloc[0] < 0:
            heading -= np.pi
        yr = -np.radians(td.yawrate.values)
        dist = track.orig_midpath.project(pos)
        
        #alpha = 0.5
        #std = 6.33e-4
        #noise = std*np.sqrt((2 - alpha)/alpha)
        noise = np.radians(5.0)
        print(np.degrees(noise))
        alpha = 0.1
        beta = 0.1
        
        for i in range(10):
            pursuiter = simulate_pursuiter(dt, alpha, beta, noise, track,
                    pos[0],
                    heading[0],
                    speed,
                    yr[0])
            data = [next(pursuiter) for i in range(int(30/dt))]
            data = np.rec.fromrecords(data, dtype=[
            ('ts', float),
            ('pos', float, 2),
            ('heading', float),
            ('speed', float),
            ('yr', float),
            ('dyr', float)
            ])

            mdist = track.orig_midpath.project(data.pos)
            valid = mdist <= dist[-1]
            mdist = mdist[valid]
            data = data[valid]

            plt.plot(mdist, data.yr, color='red', alpha=0.1)
        plt.plot(dist, yr, color='black', alpha=0.5)
        #dyr = desired_yawrate(pos, heading, speed, yr, track)
        #pyr = actuator(dyr, *p)
        #plt.plot(ts - ts[0], yr, color='green', alpha=0.3)
        #plt.plot(ts - ts[0], pyr, color='red')
    plt.ylabel("Yaw rate (radians per second)")
    plt.xlabel("Track position (meters)")
    plt.show()

def plot_steering():
    condition = 1
    speed = 8

    from trout_lags import lags

    
    track = get_condition_track(condition)
    data = pd.read_parquet("trout/trout_again.parquet")
    data.query("autoflag == 0 and condition == @condition and dataset == 2", inplace=True)
    data.ID[data['ID'] == 203] = 503
    
    dt = 1/60
    for s, sd in data.groupby('ID'):
        compare_model(sd, track, lags[s])

def fit_steering():
    # TODO: More conditions
    condition = 2
    speed = 8

    track = get_condition_track(condition)
    data = pd.read_parquet("trout/trout_again.parquet")
    data.query("autoflag == 0 and condition == 1 and dataset == 2", inplace=True)
    data.ID[data['ID'] == 203] = 503
    
    dt = 1/60
    for s, sd in data.groupby('ID'):
        losses = []
        for t, td in sd.groupby('trialcode'):
            ts = td.currtime.values
            pos = td[['posx_mirror', 'posz_mirror']].values
        
            heading = np.radians(td.yaw.values)
            if td.startingposition.iloc[0] < 0:
                heading -= np.pi
            yr = -np.radians(td.yawrate.values)

            dyr = desired_yawrate(pos, heading, speed, yr, track)

            def loss(p, dyr=dyr, yr=yr):
                pred = actuator(dyr, *p)
                return pred - yr
            losses.append(loss)

        loss = lambda p: np.concatenate([loss(p) for loss in losses])
        fit = scipy.optimize.least_squares(loss, [0.1, 0.1], loss='soft_l1')
        print(f"{int(s)}: {list(fit.x)},")
        #compare_model(sd, track, fit.x)


def test_desired_yawrate():
    condition = 1
    speed = 8
    
    track = get_condition_track(condition)
    data = pd.read_parquet("trout/trout_again.parquet")
    data.query("autoflag == 0 and condition == 1 and dataset == 2", inplace=True)
    
    dt = 1/60
    time_constant = 0.01
    for t, td in data.groupby('trialcode'):
        ts = td.currtime.values
        pos = td[['posx_mirror', 'posz_mirror']].values
        
        heading = np.radians(td.yaw.values)
        if td.startingposition.iloc[0] < 0:
            heading -= np.pi
        yr = -np.radians(td.yawrate.values)
        
        smoothing = 1 - np.exp(-dt/time_constant)
        dyr_model = desired_yawrate(pos, heading, speed, yr, track)
        
        """
        impulse = np.zeros(len(ts))
        impulse[0] = 1.0
        ir = actuator(impulse, 0.05, 0.01, 0.001)
        plt.plot(ir)
        plt.show()

        hack = actuator(dyr_model, 0.1, 0.05, 0.05)
        plt.plot(ts, dyr_model, 'r--')
        plt.plot(ts, hack, 'r')
        plt.plot(ts, yr, 'g')
        plt.show()
        continue
        """

        #plt.plot(ts, td.SWA); plt.show(); continue
        #plt.plot(ts, (dyr_model - yr))
        # Horrible hackery!
        #dyr_model = dyr_model - yr
        #yr = np.concatenate(([0], np.diff(np.radians(td.SWA*90))/dt))
        #plt.plot(ts[1:], )
        #plt.show()
        #continue
        
        def lagit(x, b, scale=0.0):
            return actuator(x, *b)*np.exp(scale)
            
            m = len(b)//2
            b = list(b)
            a = [1.0] + b[m:]
            b = b[:m]
            return scipy.signal.sosfilt(scipy.signal.tf2sos(b, a), x)*np.exp(scale)
            #return scipy.signal.lfilter(b, a, x)*np.exp(scale)

        def loss(b):
            #scale, b = b[0], b[1:]
            pred = lagit(dyr_model, (b))
            #alpha = 0.0
            #return (1 - alpha)*np.mean(np.abs(pred - yr)[len(b):]) + alpha*(np.mean(np.abs(b)))
            return pred - yr
            #dyr_est = (yr[1:] - (1 - smoothing)*yr[:-1])/smoothing
            #return (yr[1:] - yr[:-1])*unsmoothing + yr[:-1] - dyr_model[1:]
        
        import scipy.optimize
        fit = scipy.optimize.least_squares(loss, ([0.1, 0.1]), loss='soft_l1')
        fit.x = (fit.x)
        yr_est = lagit(dyr_model, fit.x)
        plt.plot(ts, np.degrees(yr), 'k', label="Measured yawrate")
        plt.plot(ts, np.degrees(yr_est), 'r', label="Predicted yawrate")
        plt.plot(ts, np.degrees(dyr_model), 'y', label="Desired yawrate")
        plt.ylim(-90, 90)
        plt.legend()
        plt.figure()
        
        ts -= ts[0]
        impulse = np.zeros(len(ts))
        impulse[0] = 1
        print(fit.x)
        #ir = scipy.signal.lfilter(fit.x[1:], [1.0], impulse)*np.exp(fit.x[0])
        ir = lagit(impulse, fit.x)
        plt.plot(ts, ir)


        plt.show()

import numba
@numba.njit
def argmedioid(xs):
    min_dist = np.inf
    min_i = 0
    N = len(xs)
    for i in range(N):
        dist = 0.0
        for j in range(N):
            dist += np.linalg.norm(xs[i] - xs[j])
            if dist > min_dist:
                break
        else:
            min_dist = dist
            min_i = i
    return min_i

def plot_steering_aggregates(condition=1):
    track = get_condition_track(condition)
    speed = 8
    dt = 1/60
    
    alpha = 0.1
    beta = 0.1
    noise = np.radians(5.0)
    
    midline = purepursuit.Path(track.midpath.points, track.midpath.dist)
    edges = tuple(track.edges)

    pyr_interps = []
    syr_interps = []
    
    plp_interps = []
    slp_interps = []
    
    data = pd.read_parquet("trout/trout_again.parquet")
    data.query("autoflag == 0 and condition == @condition and dataset == 2", inplace=True)
    #data.ID[data['ID'] == 203] = 503
    for i, (t, td) in enumerate(data.groupby('trialcode')):
        #if i > 10: break

        nts = np.arange(*td.currtime.iloc[[0, -1]], dt)
        pos = td[['posx_mirror', 'posz_mirror']].values
        heading = np.unwrap(np.radians(td.yaw.values))
        
        pos = interp1d(td.currtime.values, pos, axis=0)(nts)
        heading = interp1d(td.currtime.values, heading, axis=0)(nts)
        yr = np.gradient(heading, dt)
        dist = track.orig_midpath.project(pos)

        lane_pos = track.orig_midpath.signed_error(dist, pos)
        pyr_interps.append(scipy.interpolate.interp1d(dist, yr, bounds_error=False))
        plp_interps.append(scipy.interpolate.interp1d(dist, lane_pos, bounds_error=False))

        #yr = -np.radians(td.yawrate.values)
        
        
        if td.startingposition.iloc[0] < 0:
            heading -= np.pi
        sd = purepursuit.simulate_pursuiter_N(dt, int(300/speed/dt), alpha, beta, noise, midline, edges, pos[0], heading[0], speed, yr[0])
        sd = np.rec.fromrecords(sd, dtype=[
            ('ts', float),
            ('pos', float, 2),
            ('heading', float),
            ('speed', float),
            ('yr', float),
            ('dyr', float),
            ])

        sdist = track.orig_midpath.project(sd.pos)
        slane_pos = track.orig_midpath.signed_error(sdist, sd.pos)
        syr_interps.append(scipy.interpolate.interp1d(sdist, sd.yr, bounds_error=False))
        slp_interps.append(scipy.interpolate.interp1d(sdist, slane_pos, bounds_error=False))
        #pos_interps.append(scipy.interpolate.interp1d(dist, d.yr))
        #plt.plot(dist, d.yr)
    
    #drng = np.arange(0, 300, 0.1)
    drng = np.arange(
            np.max([i.x[0] for i in pyr_interps]),
            np.min([i.x[-1] for i in pyr_interps]),
            )
    yrs = np.array([interp(drng) for interp in pyr_interps]).T
    syrs = np.array([interp(drng) for interp in syr_interps]).T
    
    plt.subplot(2,1,1)
    yr_l, yr_h = np.percentile(yrs, [5, 95], axis=1)
    plt.fill_between(drng, np.degrees(yr_l), np.degrees(yr_h),
            color='C0', alpha=0.25, label='Measured 90% range')
    pm = argmedioid(yrs.T)
    plt.plot(drng, np.degrees(yrs.T[pm]), color='C0', label='Measured medioid')


    syr_l, syr_h = np.percentile(syrs, [5, 95], axis=1)
    plt.fill_between(drng, np.degrees(syr_l), np.degrees(syr_h),
            color='C1', alpha=0.25, label='Simulated 90% range')
    sm = argmedioid(syrs.T)
    plt.plot(drng, np.degrees(syrs.T[sm]), color='C1', label='Simulated medioid')
    #plt.xlabel("Track distance (meters)")
    plt.ylabel("Yaw rate (degrees/second)")
    plt.xlim(*drng[[0, -1]])
    plt.ylim(-35, 35)
    plt.legend()

    
    plt.subplot(2,1,2)
    
    lps = np.array([interp(drng) for interp in plp_interps]).T
    slps = np.array([interp(drng) for interp in slp_interps]).T
    
    lp_l, lp_h = np.percentile(lps, [5, 95], axis=1)
    plt.fill_between(drng, (lp_l), (lp_h),
            color='C0', alpha=0.25, label='Measured 90% range')
    pm = argmedioid(lps.T)
    plt.plot(drng, (lps.T[pm]), color='C0', label='Measured medioid')


    slp_l, slp_h = np.percentile(slps, [5, 95], axis=1)
    plt.fill_between(drng, (slp_l), (slp_h),
            color='C1', alpha=0.25, label='Simulated 90% range')
    sm = argmedioid(slps.T)
    plt.plot(drng, (slps.T[sm]), color='C1', label='Simulated medioid')
    plt.xlabel("Track distance (meters)")
    plt.ylabel("Lane position (meters)")

    for _, target in track.targets.iterrows():
        pos = target[['xcentre', 'zcentre']].values
        dist = track.orig_midpath.project(pos)
        lp = track.orig_midpath.signed_error(dist, pos)
        #plt.plot(dist, lp, 'ko')
        plt.gca().add_artist(plt.Circle((dist, lp), target.centre_radius, color='k', alpha=0.3))
        plt.gca().add_artist(plt.Circle((dist, lp), target.target_radius, color='k', alpha=0.3))

        
    plt.xlim(*drng[[0, -1]])
    plt.axhline(0, alpha=0.25, color='black', linestyle='dashed')
    plt.axhline(-1.5, alpha=0.25, color='black')
    plt.axhline(1.5, alpha=0.25, color='black')
    plt.ylim(-1.75, 1.75)
    #plt.legend()



    plt.show()


if __name__ == '__main__':
    #plot_steering()
    #fit_steering()
    #test_desired_yawrate()
    test_steering()
    #plot_steering_aggregates()
