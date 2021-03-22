import pandas as pd
import numpy as np
from numpy import exp, log
from scipy.stats import norm as Normal
from collections import namedtuple
import scipy.optimize


Params = namedtuple("Params", "beta0 betaF alpha0 alphaF")

def trial_loglikr(params, trial):
    mean = params.beta0 + params.betaF*trial.ttlc + trial.onsettime
    std = exp(params.alpha0 + params.alphaF*log(trial.ttlc))
    #dist = truncated(Normal(mean, std), trial.onsettime, Inf)
    # FIXME: IS THIS CENSORING!!!
    dist = Normal(mean, std)
    censor_prob = dist.sf(trial.trial_duration)
    def loglikr(tot):
        if tot >= trial.trial_duration:
            return log(censor_prob)
        return dist.logpdf(tot) + log(1 - censor_prob)
    return loglikr

def trial_logliksr(params, trial):
    return np.vectorize(trial_loglikr(params, trial))

def trial_loglik(params, trial):
    return trial_loglikr(params, trial)(trial.takeover_time)

def data_loglik(params, trials):
    result = 0.0
    for trial in trials:
        lik = trial_loglik(params, trial)
        if lik != lik:
            print(trial)
            print(lik)
            print()
        result += lik

    return result
    #result = sum(trial_loglik(params, trial) for trial in trials)
    #return result

def fit():
    data = pd.read_parquet("orca/data/collated_steering.parq")

    trialcols = ["ppid", "radius", "sab", "cogload", "trialn"]
    bytrial = data.groupby(trialcols)

    stuff = []
    for t, trial in bytrial:
        try:
            takeover_i = np.flatnonzero(trial.autoflag.values == False)[0]
            takeover_time = trial.timestamp_trial.iloc[takeover_i]
        except IndexError:
            takeover_i = None
            takeover_time = np.inf
        row = trial.iloc[0].copy()
        row['takeover_time'] = takeover_time
        row['ttlc'] = trial.simulated_ttlc.iloc[0]
        row['trial_duration'] = trial.timestamp_trial.iloc[-1]
        stuff.append(row)
        #takeover_t = takeover_i === nothing ? Inf : trial.timestamp_trial[takeover_i]

    data = pd.DataFrame.from_records(stuff)

    # TODO: The simulated TTLC does not exist currently for
    # all trials. This was probably the case in the Plos One
    # paper, but the mechanistic model handles these as well, so
    # this should be fixed or hacked somehow for comparisons!
    data = data[~data.ttlc.isna()]

    init_params = Params(
        beta0 = 1.0,
        betaF = 1.0,
        alpha0 = 1.0,
        alphaF = 1.0
        )

    def losser(data):
        trials = [trial for i, trial in data.iterrows()]
        return lambda params: -data_loglik(Params(*params), trials) 


    # TODO: Handle more elegantly?
    data = data[data.cogload != 'None']

    for p, d in data.groupby('ppid'):
        #wtf = trial_loglik(params, data.iloc[0])
        loss = losser(d)
        fit = scipy.optimize.minimize(loss, init_params)
        param = Params(*fit.x)
        print(f"{p}: {repr(param)},")

if __name__ == "__main__":
    fit()
