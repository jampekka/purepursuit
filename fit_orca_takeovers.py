import itertools
import numpy as np
import scipy.interpolate
import matplotlib.pyplot as plt
import pandas as pd
from pprint import pprint

from orca.reconstruct_trajectory import get_track as get_midline, get_untouched_trajectories
from linestring import LineString, simplify_linestring, extrude_linestring
from trout_steering import Track

from pp_takeover import takeover_liks, takeover_liks_reparam, takeover_liks_kalman
from purepursuit import desired_yawrate, Path
import mangleopt

speed = 8
lanewidth = 3.0
dt = 1/60
yr_error_std = np.radians(2.38)*dt

def get_track(radius, mirrored):
    midline, origin = get_midline(radius, mirrored)
    midline = simplify_linestring(midline, 0.05)
    inside = extrude_linestring(midline, -lanewidth/2)
    outside = extrude_linestring(midline, lanewidth/2)
    return Track(LineString(midline), [outside, inside])

def resample(t, x, nt):
    return scipy.interpolate.interp1d(t, x, axis=0)(nt)

from memoize import memoize

@memoize
def get_fitting_data_():
    data = get_untouched_trajectories()
    trials = list(data.values())
    
    rnd = 6
    print(trials[0].keys())
    tots = pd.DataFrame([dict(
        sab=round(t['otraj'].sab.iloc[0], rnd),
        onset_time=round(t['otraj'].onsettime.iloc[0], rnd),
        design=t['otraj'].design.iloc[0],
        takeover_time=t['takeover_time'],
        cogload=t['cogload'],
        ppid=t['ppid'],
        trialn=t['trialn'],
        autofile=t['otraj'].autofile.iloc[0],
        direction=t['otraj'].bend.iloc[0],
        ) for t in trials])
    
   
    traj_key = lambda t: (
            round(t['otraj'].sab.iloc[0], rnd),
            round(t['otraj'].onsettime.iloc[0], rnd),
            t['otraj'].autofile.iloc[0]
            )
    traj_trials = itertools.groupby(sorted(trials, key=traj_key), key=traj_key)
    traj_bank = {}
    for g, gtrials in traj_trials:
        trial = next(gtrials)
        ud = trial['traj']
        direction = trial['otraj'].bend.iloc[0]
        track = get_track(trial['otraj'].radius.iloc[0], direction < 0)
        
        ts = ud['ts'].values
        nts = np.arange(ts[0], ts[-1], dt)
        pos = ud[['world_x', 'world_z']].values
        #steering = np.radians(ud['yawrate_seconds'].values)
        heading = np.radians(ud['world_yaw'].values)
        
        pos = resample(ts, pos, nts)
        heading = resample(ts, heading, nts)
        ts = nts
        yr = (heading[1:] - heading[:-1])/dt
        yr = np.concatenate(([yr[0]], yr))
        
        desired_yawrates = np.vectorize(desired_yawrate, signature="(2),(),(),()->()", excluded={4, 5})
        #dyr = desired_yawrate(pos, heading, speed, yr, track)
        midline = Path(track.midpath.points, track.midpath.dist)
        edges = tuple(track.edges)
        dyr = desired_yawrates(pos, heading, speed, yr, midline, edges)
        
        traj_bank[g] = (ts, yr, dyr)

    return tots, traj_bank

#alpha, noise, threshold, lag = params = [0.01915796, 0.04981239, 0.02607559, 0.69824366]

#alpha, std, threshold, lag = params = [0.11295072, 0.0872724 , 1.9999986 , 0.51054798]

#alpha, std, threshold, lag = params = [0.14631508, 0.05214661, 3.00080224, 0.48796703]

#alpha, std, threshold, lag = params = [6.24502148e-02,  2.06363640e-03, 0.02, 7.11454718e-01]

#alpha, std, threshold, lag = params = [0.06206412, 0.00120368, 0.01415333, 0.68644033]

#alpha, std, threshold, lag, error = params = [5.77977952e-02, 6.33166460e-04, 8.96752143e-03, 6.56495443e-01, 0.1]


#params = [6.12162788e-02, 9.74362942e-06, 1.84053378e-04, 2.92644757e-01]

#params = [6.90382548e-02, 8.97648648e-06, 1.83966954e-04, 3.61330133e-01]

#params = [7.17224762e-02, 9.23860790e-06, 1.82851370e-04, 3.82278315e-01]

#params = [6.94159406e-02, 8.93131831e-06, 1.82899466e-04, 3.60179820e-01]

#params = [6.74236055e-02, 7.50061567e-07, 1.55989456e-05, 3.61543928e-01]

#params = [6.74236456e-02, 7.50061520e-07, 1.55989456e-05, 3.61543955e-01]

#params = [4.80249171e-02, 1.37016843e-06, 1.53283898e-05, 2.93588576e-01]

#params = [4.81182086e-02, 1.60844490e-06, 1.80142397e-05, 2.92820740e-01*0.5]

#params = [6.74236396e-02, 7.50061520e-07, 1.55989458e-05, 3.61543995e-01]

#params = [6.43858238e-02, 1.66398375e-07, 3.82226496e-06, 4.46027563e-01]
#params = [5.72306215e-02, 1.94215217e-06, 4.08340964e-05, 4.35558728e-01]
#params = [5.73948962e-02, 2.85995292e-06, 6.00279865e-05, 4.35491483e-01]
params = [0.3, 0.0120967,  0.04898352, 0.1, 0.58334232]
#params = [6.44008249e-02, 1.66437945e-07, 3.82267318e-06, 4.46043420e-01]
#alpha, noise, threshold, lag = params = 0.1, *np.exp([-2.84727728, -2.90265812, -0.25948804])

from numpy import array

# Best iteration
log_pp_params = {3: array([-2.91174571, -7.40579375, -4.63421233, -0.8788189 ]), 4: array([-2.85082045, -7.40608906, -4.71439914, -0.42177431]), 5: array([-2.877292  , -7.28378595, -4.65057163, -0.7843038 ]), 6: array([-2.5009223 , -7.5762563 , -4.53823544, -1.0485211 ]), 7: array([-2.93787276, -7.49049447, -4.6919694 , -0.59084892]), 8: array([-2.79004184, -7.54018261, -4.63096573, -0.78937079]), 9: array([-3.02501724, -7.46692746, -4.75903107, -0.31327917]), 10: array([-2.82339088, -7.50699851, -4.71504743, -0.38586959]), 12: array([-2.99527439, -7.98231311, -4.71768767, -0.44070505]), 13: array([-2.89654134, -7.36208398, -4.73543827, -0.35272135]), 14: array([-2.85100798, -7.47578188, -4.69100261, -0.53362018]), 15: array([-2.44740657, -7.67967858, -4.56726715, -0.84721225]), 16: array([-3.01136778, -7.1497029 , -4.72045063, -0.54037272]), 17: array([-3.05971904, -6.88335768, -4.68154129, -0.80860377]), 18: array([-3.07138105, -7.20686078, -4.74300224, -0.46168236]), 19: array([-2.78041682, -7.36358308, -4.66949341, -0.60590314]), 20: array([-3.0230579 , -7.36603597, -4.70990519, -0.5766791 ])}

# Last iteration
log_pp_params = {3: array([-2.9117701 , -7.40579362, -4.63421233, -0.83231792]), 4: array([-2.92001617, -7.44877072, -4.71470967, -0.42941789]), 5: array([-2.87729365, -7.28378595, -4.65057163, -0.81504775]), 6: array([-2.50093056, -7.5762563 , -4.55314192, -0.9698413 ]), 7: array([-2.93788189, -7.49049446, -4.6919694 , -0.56636383]), 8: array([-2.79004184, -7.54018261, -4.63096573, -0.79751819]), 9: array([-3.02507102, -7.46692723, -4.75903108, -0.25893789]), 10: array([-2.823343  , -7.50699845, -4.71504743, -0.41832863]), 12: array([-2.99527439, -7.98231311, -4.71768767, -0.47723406]), 13: array([-3.01714576, -7.36129108, -4.75342989, -0.24799266]), 14: array([-2.85100798, -7.47578188, -4.69100261, -0.53769845]), 15: array([-2.44740657, -7.67967858, -4.56726715, -0.84721149]), 16: array([-3.01137249, -7.14970287, -4.72045063, -0.58679456]), 17: array([-3.0597193 , -6.88335768, -4.68154129, -0.80306082]), 18: array([-3.07138135, -7.20686078, -4.74300224, -0.4952391 ]), 19: array([-2.26736554, -7.37984864, -4.66961629, -0.97040856]), 20: array([-3.02300975, -7.36603577, -4.7099052 , -0.60324193])}

# Last iteration
log_pp_params_loaded = {3: array([-2.85277682, -7.46143374, -4.67630422, -0.61990617]), 4: array([-2.53045729, -7.74899493, -4.71516869, -0.54601102]), 5: array([-2.89649078, -7.06489626, -4.71451807, -0.54116651]), 6: array([-2.65072666, -7.47913301, -4.67362922, -0.50674748]), 7: array([-3.05981664, -7.36463373, -4.66661293, -0.46042682]), 8: array([-2.68629477, -7.42502569, -4.67061035, -0.56460143]), 9: array([-3.09710284, -7.36344837, -4.71415166, -0.45015195]), 10: array([-2.64167208, -7.1454322 , -4.71544502, -0.31481502]), 12: array([-2.08318476, -8.04524666, -4.74159011, -0.88428727]), 13: array([-2.24319405, -7.27630762, -4.79258475, -0.30529264]), 14: array([-2.326923  , -7.68476374, -4.70017211, -0.70260851]), 15: array([-1.96287243, -7.73183991, -4.51770655, -0.86721602]), 16: array([-2.69127188, -7.57149945, -4.71542635, -0.40250045]), 17: array([-2.77549982, -7.15382777, -4.72187051, -0.55591431]), 18: array([-2.75659501, -7.18472566, -4.70284655, -0.51240628]), 19: array([-2.30647256, -7.47105937, -4.70165017, -0.49320897]), 20: array([-2.4843137 , -7.6784839 , -4.77127135, -0.512496  ])}


log_pp_params = {3: array([-17.64200779, -22.12813586, -11.55885736,  -1.49477744]), 4: array([ -3.03402801, -14.37556409, -11.06333586,  -1.08206868]), 5: array([ -2.7184912 , -14.69037881, -11.63122815,  -2.20497184]), 6: array([-2.68997028, -8.85508213, -5.56673971, -1.40763506]), 7: array([ -3.02619385, -13.85316111, -10.5816009 ,  -1.05821056]), 8: array([ -3.17430424, -13.44847103,  -9.93598369,  -1.73916456]), 9: array([ -2.95600116, -14.43898681, -11.24406473,  -0.729256  ]), 10: array([-18.16200854, -23.2793743 , -12.41831691,  -0.82949107]), 12: array([ -3.21204397, -15.32161087, -11.65691851,  -0.89083407]), 13: array([ -2.70654527, -12.91934181,  -9.90770679,  -0.66183533]), 14: array([ -2.92905751, -13.31545425, -10.06529468,  -1.0304206 ]), 15: array([-2.01414037, -7.63081069, -4.49539553, -1.66929827]), 16: array([ -2.93387518, -12.9294902 , -10.09421171,  -1.3749159 ]), 17: array([ -3.17415373, -12.67100798, -10.10365646,  -1.26069548]), 18: array([ -3.21805021, -13.21159782, -10.20906035,  -1.17842178]), 19: array([ -2.4451465 , -14.4955913 , -11.53826048,  -1.98047682]), 20: array([ -3.26605134, -13.55311822, -10.2533292 ,  -1.38437547])}


def get_fitting_data():
    tots, traj_bank = get_fitting_data_()
    # TODO: Could handle these as well
    #print(np.sum(~np.isfinite(tots.takeover_time)))
    #tots = tots[np.isfinite(tots.takeover_time)]
    tots = tots[tots.takeover_time > 2.5]
    #tots = tots[tots.takeover_time < 14]
    tots = tots[tots.cogload == 'None'] 

    #tots = tots[tots.ppid != 11] # Some weirdness!
    
    return tots, traj_bank

def plot():
    tots, traj_bank = get_fitting_data()
    lik_interps = []
    bi = 0
    db = 0.15
    bins = np.arange(2.5, 14, db)
    
    for g, traj in traj_bank.items():
        sab, onset_time, autofile = g
        ts = traj[0]
        tot = tots.query("sab==@sab and onset_time==@onset_time and autofile==@autofile")
        
        if len(tot) == 0: continue

        
        liks = np.zeros(len(ts))
        for s, sd in tot.groupby('ppid'):
            continue
            pparams = log_pp_params[s][:]
            share = (len(sd)/len(tot))
            liks += takeover_liks_reparam(ts, *traj, *np.exp(pparams))*(len(sd)/len(tot))
            #std_obs, threshold, lag = np.exp(pparams)
            #liks += takeover_liks_kalman(ts, *traj, std_obs, yr_error_std, threshold, lag)*share
        tmp = params[:]
        liks = takeover_liks(ts - ts[0], dt, *traj[1:], *tmp)
        
        #std_obs, threshold, lag = 0.02824987, 0.06702236, 0.91022482
        std_obs, threshold, lag = 0.02958572, 2.37495985, 0.73058758
        #liks = takeover_liks_kalman(ts, *traj, std_obs, yr_error_std, threshold, lag)

        #liks = takeover_liks_reparam(ts, *traj, *[6.08975526e-02, 2.03164559e-06, 3.86170472e-05, 2.95078476e-01, 2.72922393e-01*1e-9])
        
        share = len(tot)/len(tots)
        lik_interps.append(scipy.interpolate.interp1d(ts, liks*share, bounds_error=False, fill_value=0))
        if tot.iloc[0].design != 'balanced': continue
        #plt.hist(tot.takeover_time, weights=np.repeat(1/len(tot), len(tot)), bins=bins, color=f'C{bi}', alpha=0.25)
        #plt.hist(tot.takeover_time, weights=np.repeat(1/len(tot), len(tot)), bins=bins, color=f'C{bi}', alpha=0.25)
        #plt.plot(ts, liks*share, color=f'C{bi}')

        hist, wtfbins = np.histogram(tot.takeover_time, bins=bins, density=True)
        plt.hist(bins[:-1], bins=wtfbins, weights=hist*share, color=f'C{i}', alpha=0.25)
        plt.plot(ts, pdf*share, color=f'C{i}', label=f'SAB {sab:.1f} degrees/second')
        bi += 1
    ts = np.arange(2.5, 14, dt)
    liks = np.sum([l(ts) for l in lik_interps], axis=0)
    plt.plot(ts, liks, color='black', label='Overall')
    plt.hist(tots.takeover_time, bins=bins, color='black', alpha=0.1, density=True)
    plt.xlabel("Time from trial start (seconds)")
    plt.ylabel("Takeover density")
    plt.legend()
    plt.show()

eps = np.finfo(float).eps
def fit_data(tots, initial, prior, traj_bank):
    def likelihoods(*args):
        liks = []
        for _, trial in tots.iterrows():
            traj = traj_bank[(trial['sab'], trial['onset_time'], trial['autofile'])]
            tot = trial.takeover_time - traj[0][0]
            #lik = takeover_liks_reparam(np.array([tot]), *traj, *args)[0]
            lik = takeover_liks_reparam(np.array([tot]), dt, *traj[1:], 0.1, *args)[0]
            #lik = takeover_liks_kalman(np.array([tot]), dt, *traj[1:], *args)[0]

            #std_obs, threshold, lag = args
            #lik = takeover_liks_kalman(np.array([tot]), *traj, std_obs, yr_error_std, threshold, lag)[0]
            #lik = takeover_liks_kalman(np.array([tot]), *traj, *args)[0]
            liks.append(lik)

        return np.array(liks)
    
    best_seen = np.inf
    best_seen_args = []
    def loss(args):
        nonlocal best_seen
        nonlocal best_seen_args

        eargs = np.exp(args)
        liks = likelihoods(*eargs)
        lik = np.sum(np.log(liks + eps)) + np.log(prior(args))

        loss = -lik

        if loss < best_seen:
            best_seen = loss
            best_seen_args = eargs
            print(lik, eargs)
    
        return loss
    
    fit = scipy.optimize.minimize(loss, initial, method='nelder-mead')
    #fit = scipy.optimize.basinhopping(loss, initial, T=10,
    #        minimizer_kwargs={'method': 'nelder-mead'}
    #        )
    return fit

from sklearn.covariance import MinCovDet
from glob import glob
def estimate_transition_std():
    #data = get_untouched_trajectories()
    midlines = glob("orca/data/Midline_80_?.csv")
    des = []
    for midline in midlines:
        d = pd.read_csv(midline)
        print(list(d.columns))
        ts = d.timestamp.values
        pos = d[['World_x', 'World_z']].values
        heading = np.unwrap(np.radians(d['WorldYaw'].values))
        nts = np.arange(ts[0], ts[-1], dt)

        # Remove the straight part
        nts = nts[int(2.5/dt):]
        
        pos = resample(ts, pos, nts)
        heading = resample(ts, heading, nts)
        yr = (heading[1:] - heading[:-1])/dt
        yr = np.concatenate(([yr[0]], yr))
        
        track = get_track(d.radius.iloc[0], False)
        dyr = desired_yawrate(pos, heading, speed, yr, track)
        
        error = dyr - yr
        de = np.diff(error)
        des.append(de)

    des = np.concatenate(des)
    print(np.degrees(np.std(des)/dt))
    print(np.degrees(np.std(des)/dt))
    plt.hist(np.degrees(des/dt), bins=100)
    plt.show()

def fit_global():
    tots, traj_bank = get_fitting_data()
    
    #prior = lambda x: 1.0
    #initial = np.log([0.1, np.radians(0.1), np.radians(1.0), 1.0])
    #initial = np.log([0.3, np.radians(1.0), 3.0, 0.3])
    #initial = np.log([0.3, np.radians(1.0), 3.0, 0.3])
    initial = np.log([np.radians(1.0), np.radians(1.0), 3.0, 0.3])

    #initial = np.log([0.0285729, 0.0682464, 0.88670856])
    #initial = np.array(params[:])
    #initial = np.log(initial)

    #initial = np.log([yr_error_std*10, yr_error_std, 2.0, 0.3])

    #initial = np.log([0.1, np.radians(2.0), np.radians(4.0), 0.3, 0.1])
    
    normpdf = scipy.stats.norm.pdf
    #prior = lambda x: normpdf(np.exp(x[0]), 0.05, 0.001)*np.exp()
    prior = lambda x: (1
            #*normpdf(np.exp(x[0]), 0.3, 0.1)
            #*normpdf(np.exp(x[1]), np.radians(1.0), np.radians(0.5))
            #*np.exp(-np.exp(x[-1]))
            )
    fit = fit_data(tots, initial, prior, traj_bank)
    fit.x = np.exp(fit.x)
    pprint(fit)

def fit_hierarchical():
    tots, traj_bank = get_fitting_data()

    normlogpdf = scipy.stats.norm.logpdf
    ppid_map = {ppid: i for i, ppid in enumerate(tots.ppid.unique())}
    maxseen = -np.inf
    
    trials = []
    for _, trial in tots.iterrows():
            traj = traj_bank[(trial['sab'], trial['onset_time'], trial['autofile'])]
            tot = trial.takeover_time
            participant_i = ppid_map[trial.ppid]
            trials.append((traj, tot, participant_i))
    
    def likelihood(population_means, population_stds, participant_diffs):
        nonlocal maxseen
        #lik = normpdf(population_means)
        # TODO: Include parameter correlations
        param_lik = 0.0
        for param_i in range(len(population_means)):
            m = population_means[param_i]
            s = np.exp(population_stds[param_i])
            if ~np.isfinite(s):
                s = 1e-9
            param_lik += np.sum(normlogpdf(participant_diffs[:,param_i], 0.0, s))
        
        fit_lik = 0.0
        #for _, trial in tots.iterrows():
        for traj, tot, participant_i in trials:
            #traj = traj_bank[(trial['sab'], trial['onset_time'], trial['autofile'])]
            #tot = trial.takeover_time
            #participant_i = ppid_map[trial.ppid]
            
            param = population_means + participant_diffs[participant_i]
            param = np.exp(param)
            #lik = takeover_liks_reparam(np.array([tot]), *traj, *args)[0]
            ts, yr, dyr = traj
            lik = np.log(takeover_liks(np.array([tot - ts[0]]), dt, yr, dyr, *param)[0] + 1e-9)
            fit_lik += lik
        
        total_lik = param_lik + fit_lik
        if total_lik > maxseen:
            maxseen = total_lik
            print("Liks", total_lik, fit_lik)
            print("Means", np.exp(population_means))
            print("Stds", np.exp(population_stds))
            print("Ppdiffs")
            print(participant_diffs)
        
        return (total_lik)

    initial_means = np.log([0.1, np.radians(2.0), np.radians(4.0), 0.3])
    
    Nparam = len(initial_means)
    Nparticipants = len(ppid_map)

    initial_stds = np.log(np.repeat(np.sqrt(1/6), Nparam))
    initial_diffs = np.zeros((Nparticipants, Nparam))

    def mangle(population_means, population_stds, participant_diffs):
        return np.concatenate((
            population_means.reshape(-1),
            population_stds.reshape(-1),
            participant_diffs.reshape(-1))
            )

    def unmangle(params):
        population_means = params[:Nparam]
        population_stds = params[Nparam:2*Nparam]
        participant_diffs = params[2*Nparam:].reshape(Nparticipants, Nparam)
        return population_means, population_stds, participant_diffs

    def loss(params):
        loss = -likelihood(*unmangle(params))
        return loss

    fit = scipy.optimize.minimize(loss, mangle(initial_means, initial_stds, initial_diffs), method='nelder-mead')
    print(fit)

def fit():
    from pprint import pprint
    
    tots, traj_bank = get_fitting_data()
    #tots = tots[tots.ppid.isin(tots.ppid.unique()[:10])]
    #tots = tots[tots.design == 'balanced']

    #initial = np.log([0.1, np.radians(3), np.radians(3), 0.3])
    #initial = np.log([0.1, np.radians(3), np.radians(3), 0.0001])
    initial = np.log(params)
    #initial = np.log([0.1, np.radians(1.0), np.radians(3.0), 0.3])
    initial = np.log([np.radians(1.0), np.radians(3.0), 0.3])
    #initial = np.log([0.02705477, 0.1660191, 0.15549867])
    #initial = [-3.61057378, -1.79496228, -2.84751528]
    #initial = np.log([0.0285729, 0.0682464, 0.1*0.88670856])
    #initial = np.log([0.02858755, 2.38717505, 0.88676439])
    #initial = np.log([0.02958572, 2.37495985, 0.73058758])
    pparam = {p: initial for p in tots['ppid'].unique()}
    prior = lambda x: 1.0
    for i in range(1):
        total_loss = 0
        for p, pd in tots.groupby('ppid'):
            fit = fit_data(pd, pparam[p], prior, traj_bank)
            pparam[p] = fit.x
            loss = fit.fun
            print(p, np.exp(pparam[p]))
            total_loss += loss
        
        v = np.array(list(pparam.values()))
        #pest = MinCovDet().fit(v)
        m = np.mean(v, axis=0)
        initial = m
        cov = np.cov(v.T)
        #m = pest.location_
        #cov = pest.covariance_
        print("Iteration", i)
        print("Loss", total_loss)
        print("Mean", m)
        print("Nice mean", np.exp(m))
        print("Cors")
        print(np.corrcoef(v.T))
        print("Params")
        print(pparam)
        est = scipy.stats.multivariate_normal(m, cov, allow_singular=True)
        
        prior = lambda x: est.pdf(x) + np.exp(-x[-1])

    return
   
    def likelihoods(*args):
        liks = []
        for _, trial in tots.iterrows():
            traj = traj_bank[(trial['sab'], trial['onset_time'], trial['autofile'])]
            tot = trial.takeover_time
            lik = takeover_liks(np.array([tot]), *traj, *args)[0]
            liks.append(lik)

        return np.array(liks)

    def loss(args):
        liks = likelihoods(*np.exp(args))
        loss = -np.sum(np.log(liks + 1e-9))
        print(loss)
        return loss

    
    fit = scipy.optimize.minimize(loss, np.log([alpha, noise, threshold, lag]), method='nelder-mead')
    fit.x = np.exp(fit.x)

    print(fit)
    #pprint(mangleopt.unmangle(initial, fit.x))




if __name__ == '__main__':
    #fit()
    plot()
    #fit_global()
    #fit_hierarchical()
    #estimate_transition_std()
