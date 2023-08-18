import pandas as pd
import numpy as np
import math

options = {'TPL': [1, 1, 1, 1, 1], 'TPL_open': [1, 1, 1, 0, 1], 'TPL_closed': [1, 1, 1, 1, 0], 'TP': [1, 1, 0, 1, 1],
           'PL': [0, 1, 1, 1, 1], 'T': [1, 0, 0, 1, 1], 'P': [0, 1, 0, 1, 1], 'L': [0, 0, 1, 1, 1],
           'TP_closed': [1, 1, 0, 1, 0], 'PL_closed': [0, 1, 1, 1, 0], 'T_closed': [1, 0, 0, 1, 0],
           'P_closed': [0, 1, 0, 1, 0], 'L_closed': [0, 0, 1, 1, 0],
           'TP_open': [1, 1, 0, 0, 1], 'PL_open': [0, 1, 1, 0, 1], 'T_open': [1, 0, 0, 0, 1], 'P_open': [0, 1, 0, 0, 1],
           'L_open': [0, 0, 1, 0, 1], }
alpha = 0.1

def engagement_repetition():
    for type in ['Second', 'Utterance']:
        count = pd.DataFrame()
        value = pd.DataFrame()
        for m, measure in enumerate(['speaking', 'self', 'other', 'speaklag', 'selflag', 'otherlag']):
            count['indices'] = ['TPL', 'TP', 'PL', 'T', 'P', 'L']
            value['indices'] = ['TPL', 'TP', 'PL', 'T', 'P', 'L']
            for word in ['','_open','_closed']:
                count_temp = []
                value_temp = []
                for tpl in ['TPL','TP','PL','T','P','L']:
                    df = pd.read_csv('Results/Engagement_Repetition/'+tpl+word+'_by_'+type+'.csv')
                    significant = df[df[measure+'pval']<=alpha]
                    count_temp.append(len(significant))
                    try:
                        value_temp.append((len(significant[significant[measure + 'param'] > 0.0]) / len(
                            significant) * 10000 // 100) / 100)
                    except:
                        value_temp.append(0)
                count[word] = count_temp
                value[word] = value_temp
            count.set_index('indices',inplace=True)
            value.set_index('indices', inplace=True)
            count.to_csv('Processed_Results/Engagement_Repetition/'+type+'/Count/'+measure+'.csv', index=True)
            value.to_csv('Processed_Results/Engagement_Repetition/' + type + '/Value/' + measure + '.csv', index=True)

def session_engagement_repetition():
    for type in ['Second', 'Utterance']:
        count = pd.DataFrame()
        value = pd.DataFrame()
        for m, measure in enumerate(['utterances per minute', 'self-repetition', 'other-repetition']):
            count['indices'] = ['TPL', 'TP', 'PL', 'T', 'P', 'L']
            value['indices'] = ['TPL', 'TP', 'PL', 'T', 'P', 'L']
            for word in ['','_open','_closed']:
                count_temp = []
                value_temp = []
                for tpl in ['TPL','TP','PL','T','P','L']:
                    df = pd.read_csv('Results/Session_Engagement_Repetition/'+tpl+word+'_by_'+type+'.csv',index_col=0)
                    significant = df[df[measure+'pval']<=alpha]
                    count_temp.append(len(significant.index.tolist()))
                    value_temp.append(significant.index.tolist())
                count[word] = count_temp
                value[word] = value_temp
            count.set_index('indices',inplace=True)
            value.set_index('indices', inplace=True)
            count.to_csv('Processed_Results/Session_Engagement_Repetition/'+type+'/Count/'+measure+'.csv', index=True)
            value.to_csv('Processed_Results/Session_Engagement_Repetition/' + type + '/Value/' + measure + '.csv', index=True)

session_engagement_repetition()
engagement_repetition()