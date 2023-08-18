import os
import numpy as np
import pandas as pd
import glob

def engagement_by_second():
    path = './EngAnno_3'
    file_list = glob.glob(path + '/*.csv')

    data = []
    for i in range(30):
        data.append(pd.DataFrame())

    for file in file_list:
        session = int(file.split('S')[1].split('_')[0]) - 1

        participant = 'P' + str(file.split('P')[1].split('_')[0])
        df = pd.read_csv(file, skiprows=9, header=None)
        df.drop(0,axis = 1,inplace=True)
        data[session][participant] = df[1]

    #for i in range(0,30):
        #data[i].to_csv('Engagement_by_Second/engagement_S'+str(i+1)+'.csv', index_label=False, index = False)
    print(data)

def engagement_by_utterace():
    for session in range(1, 31):
        print(session)
        # Read session files
        df = pd.read_csv('Sessions/S' + str(session) + '.csv')
        engagement = pd.read_csv('Engagement_by_Second/engagement_S' + str(session) + '.csv')

        # List of all speakers in session
        speakers = engagement.columns

        # Initialise engagementU dataframe at 0 for each utterance
        engagement_utt = pd.DataFrame()
        for speaker in speakers:
                engagement_utt[speaker] = [0] * len(df)

        # Iterate through session lines
        for ind in df.index:
            # Take start and end time to update the entire span
            start = int(float(df['start'][ind]) + 0.5)
            end = int(float(df['end'][ind]) + 1.5)

            for speaker in speakers:
                # Record speaking times
                engagement_utt[speaker][ind] = sum(engagement[speaker][start:end])/(end-start)

        engagement_utt.to_csv('Engagement_by_Utterance/engagement_S' + str(session) + '.csv', index=False)

engagement_by_utterace()