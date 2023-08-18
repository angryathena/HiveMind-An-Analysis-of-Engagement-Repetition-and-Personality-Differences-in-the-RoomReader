import numpy as np
import pandas
import pandas as pd
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import treetaggerwrapper as tt
import warnings
import numpy as np
from numpy import cov
from scipy.stats import pearsonr

import pandas as pd
from scipy.stats import shapiro, kruskal, norm
import matplotlib.pyplot as plt
import statsmodels.api as sm

warnings.filterwarnings("ignore")

options = {'TPL': [1, 1, 1, 1, 1], 'TPL_open': [1, 1, 1, 0, 1], 'TPL_closed': [1, 1, 1, 1, 0], 'TP': [1, 1, 0, 1, 1],
           'PL': [0, 1, 1, 1, 1], 'T': [1, 0, 0, 1, 1], 'P': [0, 1, 0, 1, 1], 'L': [0, 0, 1, 1, 1],
           'TP_closed': [1, 1, 0, 1, 0], 'PL_closed': [0, 1, 1, 1, 0], 'T_closed': [1, 0, 0, 1, 0],
           'P_closed': [0, 1, 0, 1, 0], 'L_closed': [0, 0, 1, 1, 0],
           'TP_open': [1, 1, 0, 0, 1], 'PL_open': [0, 1, 1, 0, 1], 'T_open': [1, 0, 0, 0, 1], 'P_open': [0, 1, 0, 0, 1],
           'L_open': [0, 0, 1, 0, 1], }

second_lag = 2
second_delay = 2
utterance_lag = 1
utterance_delay = 0


def lag(measure, l, d):
    lagged = measure.tolist()
    final = np.array([0.0] * len(lagged))
    prev = []
    for i in range(0, l + d):
        lagged.insert(0, 0.0)
        lagged.pop()
        prev.append(np.copy(lagged))
    for i in range(l + d - 1, d - 1, -1):
        np.add(final, prev[i], out=final, casting="unsafe")
    return final / l


def compute_lags(type='Utterance', option='TPL'):
    l = second_lag if type == 'Second' else utterance_lag
    d = second_delay if type == 'Second' else utterance_delay

    for s in range(1, 31):
        data = pd.DataFrame()
        engagement = []
        speaking = []
        self_repetition = []
        other_repetition = []
        engagement_lag = []
        speaking_lag = []
        self_repetition_lag = []
        other_repetition_lag = []

        engagement_df = pd.read_csv('Engagement_by_' + type + '/engagement_S' + str(s) + '.csv')
        speaking_df = pd.read_csv('Speaking_by_' + type + '/speaking_S' + str(s) + '.csv')
        repetition_df = pd.read_csv('Repetition_by_' + type + '/repetition_S' + str(s) + '.csv')
        for participant in engagement_df.columns:
            engagement_participant = engagement_df[participant]
            speaking_participant = speaking_df[participant]
            self_repetition_participant = repetition_df[participant + '_self_' + option]
            other_repetition_participant = repetition_df[participant + '_other_' + option]

            engagement.extend(engagement_participant)
            speaking.extend(speaking_participant)
            self_repetition.extend(self_repetition_participant)
            other_repetition.extend(other_repetition_participant)

            engagement_lag.extend(lag(engagement_participant, l, d))
            speaking_lag.extend(lag(speaking_participant, l, d))
            self_repetition_lag.extend(lag(self_repetition_participant, l, d))
            other_repetition_lag.extend(lag(other_repetition_participant, l, d))

        data['engagement'] = engagement
        data['speaking'] = speaking
        data['other repetition'] = other_repetition
        data['self repetition'] = self_repetition

        data['engagement lag'] = engagement_lag
        data['speaking lag'] = speaking_lag
        data['other repetition lag'] = other_repetition_lag
        data['self repetition lag'] = self_repetition_lag

        data.to_csv('Lag_by_' + type + '/lag_' + option + '_S' + str(s)+'.csv', index=False)


def compute_more_lags(type = 'Utterance',l = 1, d=0):
    for s in range(1, 31):
        print(s)
        engagement = []
        speaking = []

        engagement_lag = []
        speaking_lag = []

        engagement_df = pd.read_csv('Engagement_by_' + type + '/engagement_S' + str(s) + '.csv')
        speaking_df = pd.read_csv('Speaking_by_' + type + '/speaking_S' + str(s) + '.csv')

        for participant in engagement_df.columns:
            engagement_participant = engagement_df[participant]
            engagement.extend(engagement_participant)
            engagement_lag.extend(lag(engagement_participant, l, d))

            speaking_participant = speaking_df[participant]
            speaking.extend(speaking_participant)
            speaking_lag.extend(lag(speaking_participant, l, d))

        for option in options:
            data = pd.DataFrame()
            self_repetition = []
            other_repetition = []
            self_repetition_lag = []
            other_repetition_lag = []

            repetition_df = pd.read_csv('Repetition_by_' + type + '/repetition_S' + str(s) + '.csv')
            for participant in engagement_df.columns:
                self_repetition_participant = repetition_df[participant + '_self_' + option]
                other_repetition_participant = repetition_df[participant + '_other_' + option]
                self_repetition.extend(self_repetition_participant)
                other_repetition.extend(other_repetition_participant)
                self_repetition_lag.extend(lag(self_repetition_participant, l, d))
                other_repetition_lag.extend(lag(other_repetition_participant, l, d))

            data['engagement'] = engagement
            data['speaking'] = speaking
            data['other repetition'] = other_repetition
            data['self repetition'] = self_repetition

            data['engagement lag'] = engagement_lag
            data['speaking lag'] = speaking_lag
            data['other repetition lag'] = other_repetition_lag
            data['self repetition lag'] = self_repetition_lag

            data.to_csv('More_Lags/' + type + '/lag_' + option + '_l'+str(l)+'d'+str(d)+'_S' + str(s)+'.csv', index=False)


def run_regression(Y,X):
    X = sm.add_constant(X)
    ols_model = sm.OLS(Y, X)
    ols_model = ols_model.fit()
    print(ols_model.summary())

    resid = ols_model.resid
    fitted = ols_model.fittedvalues
    plt.scatter(fitted, resid, c='hotpink')
    plt.xlabel("Fitted Values")
    plt.ylabel("Residuals")
    #plt.show()
    print(ols_model.summary())

    expected_quantiles = norm.ppf(np.linspace(0.01, 0.99, len(np.sort(resid))))
    plt.scatter(expected_quantiles, np.sort(resid), c='hotpink')
    plt.xlabel('Theoretical Quantiles')
    plt.ylabel('Sample Quantiles')
    plt.plot(np.sort(resid), np.sort(resid), color='darkorchid')
    #plt.show()

    return [ols_model.params,ols_model.pvalues]


def repetition_vs_engagement(type, option):
    big_data = []
    for s in range(1, 31):
        data = pd.read_csv('Lag_by_' + type + '/lag_' + option + '_S' + str(s) + '.csv')
        if s in [6,12] and type == 'Second':
            data = data.sample(n = 1000, random_state=42)
        #data = data[data['speaking'] == 1]
        big_data.append(data)

    all_params = []
    all_pvals = []
    for s in range(0, 30):
        Y = big_data[s]['engagement'].values.astype('float')
        X1 = big_data[s]['speaking'].values.astype('float')
        X2 = big_data[s]['self repetition'].values.astype('float')
        X3 = big_data[s]['other repetition'].values.astype('float')
        X4 = big_data[s]['speaking lag'].values.astype('float')
        X5 = big_data[s]['self repetition lag'].values.astype('float')
        X6 = big_data[s]['other repetition lag'].values.astype('float')
        X = np.column_stack((X1, X2, X3, X4, X5, X6))
        [params, pvals] = run_regression(Y,X)
        all_params.append(params)
        all_pvals.append(pvals)

    all_params = list(zip(*all_params))
    all_pvals = list(zip(*all_pvals))
    param_pval = list(zip(all_params, all_pvals))
    results = pd.DataFrame()
    for m, measure in enumerate(['constant', 'speaking','self', 'other', 'speaklag','selflag', 'otherlag']):
        for r, result in enumerate(['param', 'pval']):
            results[measure + result] = param_pval[m][r]
    results.to_csv('Results/Engagement_Repetition/'+option+'_by_' + type + '.csv', index=False)


def session_repetition_vs_engagement(type, option):
    session_data = pd.read_csv('session_data.csv')
    repetition_data = pd.read_csv('Session_Repetition_Data/'+option+'_by_'+type+'.csv')
    indices = []
    all_params = []
    all_pvals = []
    for tutor in [' no tutor']: #''
        for participant in ['',' per participant']:
            print(tutor, participant)
            indices.append('measurement' + participant + tutor)
            Y = session_data['mean engagement by '+type].values.astype('float')
            X1 = session_data['mean utterances per minute'+participant+tutor].values.astype('float')
            X2 = repetition_data['mean self-repetition'+participant+tutor].values.astype('float')
            X3 = repetition_data['mean other-repetition'+participant+tutor].values.astype('float')
            X = np.column_stack((X1, X2, X3))
            [params, pvals] = run_regression(Y, X)
            all_params.append(params)
            all_pvals.append(pvals)

    all_params = list(zip(*all_params))
    all_pvals = list(zip(*all_pvals))
    param_pval = list(zip(all_params, all_pvals))
    results = pd.DataFrame()
    for m, measure in enumerate(['constant', 'utterances per minute', 'self-repetition', 'other-repetition']):
        for r, result in enumerate(['param', 'pval']):
            results[measure + result] = param_pval[m][r]
    results['index'] =indices
    results.set_index('index', inplace=True)
    results.to_csv('Results/Session_Engagement_Repetition/'  +option + '_by_' + type+'.csv', index=True)


def session_personality_vs_engagement(type):
    session_data = pd.read_csv('session_data.csv')
    all_params = []
    all_pvals = []
    for stat in ('mean ', 'std '):
        Y = session_data['mean engagement by '+type].values.astype('float')
        X1=  session_data[stat +'openness'].values.astype('float')
        X2 = session_data[stat +'conscientiousness'].values.astype('float')
        X3 = session_data[stat +'extraversion'].values.astype('float')
        X4 = session_data[stat +'agreeableness'].values.astype('float')
        X5 = session_data[stat +'neuroticism'].values.astype('float')
        Xx = session_data['mean utterances per minute per participant'].values.astype('float')
        X = np.column_stack((X1, X2, X3, X4))
        #X = np.column_stack((X1*X1,X2*X2, X3*X3,X4*X4,X5*X5))
        [params, pvals] = run_regression(Y, X)
        all_params.append(params)
        all_pvals.append(pvals)


    all_params = list(zip(*all_params))
    all_pvals = list(zip(*all_pvals))
    param_pval = list(zip(all_params, all_pvals))
    results = pd.DataFrame()
    for m, measure in enumerate(['constant', 'openness', 'conscientiousness', 'extraversion','agreeableness','neuroticism']):
        for r, result in enumerate(['param', 'pval']):
            results[measure + result] = param_pval[m][r]
    results['index'] = ['mean', 'std']
    results.set_index('index', inplace=True)
    results.to_csv('Results/Session_Engagement_Personality/by_' + type + '.csv', index=True)


def participant_personality_vs_engagement(type):
    participant_data = pd.read_csv('participant_data.csv')
    participant_data.dropna(inplace = True)
    all_params = []
    all_pvals = []

    #Raw
    print('Raw')
    Y = participant_data['mean engagement by ' + type].values.astype('float')
    X1 = participant_data['openness']
    X2 = participant_data['conscientiousness']
    X3 = participant_data['extraversion']
    X4 = participant_data['agreeableness']
    X5 = participant_data['neuroticism']

    X1dev = participant_data['openness deviation']
    X2dev = participant_data['conscientiousness deviation']
    X3dev = participant_data['extraversion deviation']
    X4dev = participant_data['agreeableness deviation']
    X5dev = participant_data['neuroticism deviation']

    X = np.column_stack((X1, X2, X3,X4, X5, X1dev, X2dev, X3dev,X4dev, X5dev))
    [params, pvals] = run_regression(Y, X)
    all_params.append(params)
    all_pvals.append(pvals)

    # Absolute
    print('Absolute')
    X1dev, X2dev, X3dev, X4dev, X5dev = np.abs(X1dev), np.abs(X2dev), np.abs(X3dev), np.abs(X4dev), np.abs(X5dev)
    X = np.column_stack((X1, X2, X3, X4, X5, X1dev, X2dev, X3dev, X4dev, X5dev))
    [params, pvals] = run_regression(Y, X)
    all_params.append(params)
    all_pvals.append(pvals)

    all_params = list(zip(*all_params))
    all_pvals = list(zip(*all_pvals))
    param_pval = list(zip(all_params, all_pvals))
    results = pd.DataFrame()
    for m, measure in enumerate(['constant', 'openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism', 'openness deviation', 'conscientiousness deviation', 'extraversion deviation', 'agreeableness deviation', 'neuroticism deviation']):
        for r, result in enumerate(['param', 'pval']):
            results[measure + result] = param_pval[m][r]
    results['index'] = ['raw', 'absolute']
    results.set_index('index', inplace=True)
    results.to_csv('Results/Participant_Engagement_Personality/by_' + type + '.csv', index=True)


def participant_pairs(type):
    participant_data = pd.read_csv('participant_pair_data.csv')
    participant_data.dropna(inplace=True)
    all_params = []
    all_pvals = []

    # Raw
    print('Raw')
    Y = participant_data['engagement by ' + type]

    X1dev = participant_data['openness difference']
    X2dev = participant_data['conscientiousness difference']
    X3dev = participant_data['extraversion difference']
    X4dev = participant_data['agreeableness difference']
    X5dev = participant_data['neuroticism difference']

    E = participant_data['mean engagement by ' + type]
    X = np.column_stack((E, X1dev, X2dev, X3dev, X4dev))
    [params, pvals] = run_regression(Y, X)
    all_params.append(params)
    all_pvals.append(pvals)

    # Absolute
    print('Absolute')
    X1dev, X2dev, X3dev, X4dev, X5dev = np.abs(X1dev), np.abs(X2dev), np.abs(X3dev), np.abs(X4dev), np.abs(X5dev)
    print(X1dev)
    X = np.column_stack((E, X1dev, X2dev, X3dev, X4dev))
    [params, pvals] = run_regression(Y, X)
    all_params.append(params)
    all_pvals.append(pvals)

    all_params = list(zip(*all_params))
    all_pvals = list(zip(*all_pvals))
    param_pval = list(zip(all_params, all_pvals))
    results = pd.DataFrame()
    for m, measure in enumerate(
            ['constant', 'mean engagement',
             'openness difference', 'conscientiousness difference', 'extraversion difference', 'agreeableness difference',]):
        for r, result in enumerate(['param', 'pval']):
            results[measure + result] = param_pval[m][r]
    results['index'] = ['raw', 'absolute']
    results.set_index('index', inplace=True)
    results.to_csv('Results/Pairs_Engagement_Personality/by_' + type + '.csv', index=True)

def test_session(data1,data2):
    covariance = cov(data1, data2)
    print(covariance)
    correlation, pval = pearsonr(data1,data2)
    print(correlation,pval)
    plt.scatter(data1, data2, color='darkorchid')
    plt.show()



type = 'Second'
option = 'TPL'
#repetition_vs_engagement(type,option)
for type in ['Second', 'Utterance']: #['Second','Utterance']:
    #print(type)
    for option in options:
        #print(option)
        #compute_lags(type,option)
        #repetition_vs_engagement(type,option)
        #session_repetition_vs_engagement(type, option)
        continue
    #session_personality_vs_engagement(type)
    #participant_pairs(type)
    #participant_personality_vs_engagement(type)
    continue

#for l in [1,10,100]:
    #for d in [0,5,50]:
        #print(l,d)
        #compute_more_lags('Utterance',l,d)