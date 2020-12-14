import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate
import itertools

from orca.reconstruct_trajectory import get_track as get_midline, get_untouched_trajectories as get_all_untouched_trajectories

from unscented_purepursuit import circular_steering, circular_steering_loglik, simulate_circular_steering, circular_steerer

DT = 1/60
SPEED = 8

def get_untouched_trajectories():
    data = get_all_untouched_trajectories()
    # Get rid of outliers and early responses
    return {key: val for key, val in data.items()
            if np.max(np.abs(val['otraj']['steeringbias'])) < 4
            and val['takeover_time'] > 2.5}

def resample(t, x, nt):
    return scipy.interpolate.interp1d(t, x, axis=0)(nt)

def get_manual_steering():
    for trial in get_untouched_trajectories().values():
        od = trial['otraj']
        ud = trial['traj']
        manual_control = ud.yawrate_seconds - od.yawrate_seconds
        
        try:
            steering_start = np.flatnonzero(np.abs(manual_control.values) > 0.001)[0]
        except IndexError:
            # No takeover
            continue
        manual = od.iloc[steering_start-2:]
        
        ts = manual['ts'].values
        ots = ts
        nts = np.arange(ts[0], ts[-1], DT)
        pos = manual[['world_x', 'world_z']].values
        heading = np.radians(manual['world_yaw'].values)
        
        pos = resample(ts, pos, nts)
        heading = resample(ots, heading, nts)
        yr = (heading[1:] - heading[:-1])/DT
        ts = nts[1:]
        heading = heading[1:]
        pos = pos[1:]
        
        manual = np.rec.fromarrays([ts, pos[:,0], pos[:,1], heading, yr], names="ts,x,z,heading,yawrate")

        yield manual, trial

def fit_manual_steering(trials):
    trials = list(trials)

    def loss(param):
        total = 0.0
        param = np.exp(param)
        for manual, trial in trials:
            radius = trial['radius']*trial['direction']
            origin = radius, 16
            loglik = circular_steering_loglik(DT, origin, radius, *param, SPEED,
                    manual.x.copy(), manual.z.copy(), manual.heading.copy(), manual.yawrate.copy())
            total += loglik

        return -total
    
    initial = np.array(
        [0.1, 0.2, 0.2, 1.5, 0.01]
        )
    
    fit = scipy.optimize.minimize(loss, np.log(initial), method='powell')
    fit.x = np.exp(fit.x)
    return fit
    

def fit_per_participant_steering():
    data = get_manual_steering()
    data = [d for d in data if d[1]['cogload'] != 'None']
    key = lambda x: x[1]['ppid']
    
    data = sorted(data, key=key)
    for ppid, ppdata in itertools.groupby(data, key=key):
        fit = fit_manual_steering(ppdata)
        print(f"{ppid}: {list(fit.x)}, # {-fit.fun}")

steering_fits_noload = {
3: [0.20887707974199315, 0.05294987313442887, 0.30394260199514594, 1.4391233087113307, 0.18260599328882668], # 123232.6215582926
4: [0.4001991383802507, 0.062225740088505996, 0.23527165195267355, 1.2308617620287097, 0.03479770706158819], # 123519.7555550032
5: [0.28605517097891564, 0.04776811536677616, 0.3745977436573845, 1.2696367970916471, 0.27623227023396985], # 116676.06373637615
6: [0.194117349123435, 0.0869691840734386, 0.28961886753734084, 1.9183394869809396, 0.012307299549314844], # 118392.7087041196
7: [0.1811267408385473, 0.07831810712543613, 0.22066477366369264, 2.141707760359671, 0.029342504605309138], # 127746.5609936991
8: [0.12603113080886358, 0.2016642844308186, 0.2481752686634155, 2.1390308874422925, 0.0447118913134521], # 138170.9737601959
9: [0.3617649155497172, 0.1575841569071935, 0.5285058942512659, 1.3919422175048204, 0.03802213260472925], # 108442.69358798045
10: [0.4382995728967535, 0.057379503275585594, 0.29781502091375633, 1.4241187645495745, 0.015801564160069802], # 87166.38784598537
11: [0.34208117446208597, 0.059675677174250985, 0.2141547179681285, 1.1377616676849642, 0.17185583981883673], # 83671.92133739001
12: [0.31958529617291825, 0.14956389197911535, 0.2045926518702414, 1.2385790414193725, 0.07630886781997792], # 96873.29005287138
13: [0.4848974390710137, 0.0698315541855622, 0.19475417994096086, 1.9139023482606499, 0.029399999054440573], # 72632.06230909491
14: [0.24708666367539509, 0.10258926703925166, 0.2775901686027914, 1.6989706282847648, 0.1155640299695203], # 113461.60942128809
15: [0.3386624840548482, 0.018593764341127517, 0.42279204021740974, 1.6236360712929676, 0.06248899864775237], # 94596.22055426496
16: [0.15010223755839075, 0.05795253006741152, 0.16616767327069057, 2.458725324604743, 0.015755488428812806], # 142990.2671158318
17: [0.1854088840068683, 0.07820998819742625, 0.19897773709704344, 2.2020572266954375, 0.011303368517527623], # 146638.81012297107
18: [0.30793667590731516, 0.03232243553975776, 0.33539074703679467, 1.6028808915222366, 0.03575277967150902], # 124595.36802395672
19: [0.41742155138951736, 0.07312033457058982, 0.30714663958204313, 1.454705681221404, 0.016968288159700987], # 81507.75894248438
20: [0.28150654998334135, 0.053098274644755764, 0.28120755095839467, 1.3912277556431687, 0.030603206533028546], # 111636.5293938377
}

steering_fits_load = {
3: [0.16041897566737962, 0.11863608686525953, 0.24738681360454243, 1.6012461330187977, 0.022500913283153096], # 136590.53533617372
4: [0.3733246944635002, 0.11297239518939389, 0.212715366846773, 1.258419702100991, 0.12394224206746185], # 87416.0311845459
5: [0.15211531566721345, 0.07057412733773898, 0.3189564356968285, 1.6804197601481334, 0.10569595871520897], # 160276.71362220778
6: [0.18311660170752964, 0.03471248924591625, 0.41877012055397866, 1.9073628613035358, 0.16126538202507784], # 127355.85050037237
7: [0.1449062009497157, 0.11831992667424468, 0.2148026393140451, 2.106763684177583, 0.014370179873134102], # 131039.8752961493
8: [0.1252995797490331, 0.1483186798577729, 0.2889083661524778, 2.269727522929687, 0.0524291094784617], # 138460.20625499685
9: [0.27540041050737, 0.0796739967393797, 0.6318322526506382, 1.4456271108276701, 0.016567054836358418], # 130588.99261851133
10: [0.2554627722590304, 0.11225101779637274, 0.38935570504772726, 2.0573234980570807, 0.023519238525534787], # 117002.6916064477
11: [0.30726528910627054, 0.09565176771660712, 0.23734575968337818, 1.309238473755244, 0.14158520794587073], # 86083.82980761684
12: [0.3490064601496063, 0.1359885862903914, 0.24145208278601094, 1.4963782097486749, 0.07331081468523125], # 82436.46549761145
13: [0.4071152341382368, 0.1938386937693747, 0.1976521185716329, 2.3628783329901206, 0.16932901857511187], # 60449.15117346581
14: [0.393647263234138, 0.07864768815397395, 0.3223042649533722, 2.037371619327389, 0.015962331802028246], # 86104.21237421219
15: [0.19186233261877012, 0.044281085128725195, 0.23198209537307482, 2.2856544966664893, 0.03164583920728114], # 96326.42422432869
16: [0.2339418057578599, 0.05496020803941964, 0.16793406153118445, 1.9065267153965981, 0.029241884844455075], # 104742.21557065718
17: [0.19554792211304364, 0.12606122963777056, 0.14021425433008283, 2.1050063021889875, 0.01839901502390029], # 135107.57701928972
18: [0.23918751615559078, 0.06382986238480878, 0.33779773504588423, 1.8414554759252137, 0.1465462781368064], # 119082.48061237377
19: [0.36624185617570576, 0.06903246092694516, 0.265065466850262, 1.4615828042594787, 0.16684407257085201], # 94043.1936401319
20: [0.301271575605805, 0.048244958770871184, 0.27346203793659335, 1.5478453864183974, 0.0280269941077813], # 102502.33220446958
}

steering_fits = {
    False: steering_fits_noload,
    True: steering_fits_load
        }

def plot_manual_steering():
    for manual, trial in get_manual_steering():
        if trial['otraj'].design.iloc[0] != 'balanced':
            continue
        
        sab = round(trial['sab'], 2)
        if sab != -1.2: continue
        radius = trial['radius']*trial['direction']
        origin = radius, 16
        
        ms = []
        cs = []
        
        #noise, time_constant, motor_time_constant, preview, init_std = [0.30735563, 0.23099116, 0.16525486, 3.17188864, 0.10667357]
        
        load = trial['cogload'] != 'None'
        if load: continue
        noise, time_constant, motor_time_constant, preview, motor_noise = steering_fits[load][trial['ppid']]
        steerer = circular_steerer(DT, origin, radius, noise, time_constant, motor_time_constant, preview, motor_noise, SPEED, manual.x[0], manual.z[0], manual.heading[0], manual.yawrate[0])
        for i, (m, c) in enumerate(steerer):
            ms.append(m.copy())
            cs.append(c.copy())
            if i >= len(manual) - 2:
                break
        
        sim = simulate_circular_steering(DT, len(manual) - 1, origin, radius, noise, time_constant, motor_time_constant, preview, motor_noise, SPEED, manual.x[0], manual.z[0], manual.heading[0], manual.yawrate[0])
        
        ms = np.array(ms)
        cs = np.array(cs)
    
        plt.figure(str((trial['ppid'], sab, load)))
        
        plt.plot(manual.ts[1:], sim[:,3]*trial['direction'], color='green')

        m = ms[:,3]
        s = np.sqrt(cs[:,3,3])
        plt.plot(manual.ts[1:], manual.yawrate[1:]*trial['direction'], 'k')
        plt.plot(manual.ts[1:], m*trial['direction'], 'r')
        plt.plot(manual.ts[1:], (m - s)*trial['direction'], 'r--')
        plt.plot(manual.ts[1:], (m + s)*trial['direction'], 'r--')
        plt.show()

def plot_parameters():
    import seaborn as sns
    import pandas as pd
    params = steering_fits[False]
    params = np.array(list(params.values()))
    nonloaded = pd.DataFrame.from_records(
        params,
        columns="noise time_constant motor_time_constant preview initial_std".split()
            )

    params = steering_fits[True]
    params = np.array(list(params.values()))
    loaded = pd.DataFrame.from_records(
        params,
        columns="noise time_constant motor_time_constant preview initial_std".split()
            )

    contrasts = loaded - nonloaded
    fig, plots = plt.subplots(2, 5)
    
    for i, col in enumerate(contrasts.columns):
        ax = plots[0, i]
        ax.set_title(col)
        ax.hist(nonloaded[col], color="C0", alpha=0.5)
        ax.hist(loaded[col], color="C1", alpha=0.5)
    for i, col in enumerate(contrasts.columns):
        plots[1, i].hist(contrasts[col], density=True, color='black')

    plt.show()

def plot_balanced_steering():
    data = get_manual_steering()
    data = [d for d in data if d[1]['otraj'].design.iloc[0] == 'balanced']
    data = [d for d in data if d[1]['cogload'] == 'None']
    by_sab = lambda d: round(d[1]['sab'], 2)
    
    data = sorted(data, key=by_sab)

    ts = np.arange(2.5, 15, DT)
    plt.plot([], [], 'k-', label='Data')
    plt.plot([], [], 'k--', label='Model')
    for i, (sab, sabd) in enumerate(itertools.groupby(data, key=by_sab)):
        yrs = []
        myrs = []
        for manual, trial in sabd:
            od = trial['otraj']
            load = trial['cogload'] != 'None'
            radius = trial['radius']*trial['direction']
            origin = radius, 16
            
            noise, time_constant, motor_time_constant, preview, motor_noise = steering_fits[load][trial['ppid']]
            ms, cs = circular_steering(DT, len(manual) - 1, origin, radius, noise, time_constant, motor_time_constant, preview, motor_noise, SPEED, manual.x[0], manual.z[0], manual.heading[0], manual.yawrate[0])
            
            yr = od['yawrate_seconds']*trial['direction']
            yr_int = scipy.interpolate.interp1d(od['ts'].values, yr.values)
            yrs.append(yr_int(ts))
            
            myr = yr_int(ts)
            
            myr_int = scipy.interpolate.interp1d(manual.ts[1:], np.degrees(ms[:,3])*trial['direction'], bounds_error=False)
            start_i = ts.searchsorted(manual.ts[1])
            myr[start_i:] = myr_int(ts[start_i:])
            myrs.append(myr)
        mean_yr = np.nanmean(yrs, axis=0)
        mean_model_yr = np.nanmean(myrs, axis=0)
        plt.plot(ts, mean_yr, color=f"C{i}", label=f'SAB {sab:.1f} ⁰/s')
        plt.plot(ts, mean_model_yr, '--', color=f"C{i}")
    plt.axvline(6, color='black', linestyle='dashed')
    plt.xlim(ts[0], ts[-1])
    plt.xlabel("Time (seconds)")
    plt.ylabel("Yaw rate (degrees/second)")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    #plot_balanced_steering()
    plot_manual_steering()
    #fit_per_participant_steering()
    #plot_parameters()
