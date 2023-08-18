import pandas as pd
import numpy as np
import math
from matplotlib import pyplot as plt

options = {'TPL': [1, 1, 1, 1, 1], 'TPL_open': [1, 1, 1, 0, 1], 'TPL_closed': [1, 1, 1, 1, 0], 'TP': [1, 1, 0, 1, 1],
           'PL': [0, 1, 1, 1, 1], 'T': [1, 0, 0, 1, 1], 'P': [0, 1, 0, 1, 1], 'L': [0, 0, 1, 1, 1],
           'TP_closed': [1, 1, 0, 1, 0], 'PL_closed': [0, 1, 1, 1, 0], 'T_closed': [1, 0, 0, 1, 0],
           'P_closed': [0, 1, 0, 1, 0], 'L_closed': [0, 0, 1, 1, 0],
           'TP_open': [1, 1, 0, 0, 1], 'PL_open': [0, 1, 1, 0, 1], 'T_open': [1, 0, 0, 0, 1], 'P_open': [0, 1, 0, 0, 1],
           'L_open': [0, 0, 1, 0, 1], }

participant_session = {}
overall_length = []
overall_notutor_length = []
mean_participant_length = []
std_participant_length = []
mean_participant_notutor_length = []
std_participant_notutor_length = []

total_utterances = []
total_notutor_utterances = []
mean_participant_utterances = []
mean_participant_notutor_utterances = []
std_participant_utterances = []
std_participant_notutor_utterances = []

time_between = []
time_between_notutor = []
mean_participant_wait_time = []
mean_participant_notutor_wait_time = []
std_participant_wait_time = []
std_participant_notutor_wait_time = []

engagement_second = []
engagement_utterance = []

mean_openness = []
mean_conscientiousness = []
mean_extravesion = []
mean_agreeableness = []
mean_neuroticism = []
std_openness = []
std_conscientiousness = []
std_extravesion = []
std_agreeableness = []
std_neuroticism = []
tutor_openness_difference = []
tutor_conscientiousness_difference = []
tutor_extraversion_difference = []
tutor_agreeableness_difference = []
tutor_neuroticism_difference = []

ID_list = []
tutor_list = []

personality = pd.read_csv('personality_normalised.csv')
participant_data = personality.copy()
participant_data.set_index('Participant ID', inplace=True)
participant_repetition_data = personality.copy()
participant_repetition_data.set_index('Participant ID', inplace=True)

session_repetition_data = pd.DataFrame()

def dataset_features():
    data = pd.DataFrame()
    participants_per_session = []
    tutor = []
    time = []
    utterances = []

    for s in range(1, 31):
        session = pd.read_csv('Sessions/S' + str(s) + '.csv')
        end = session['end']
        time.append(end.tolist()[-1])
        utterances.append(len(session))
        participants = session['speaker'].unique()
        participants_per_session.append(len(participants) - 1)
        tutor.append(1 if 'T001' in participants else 2)

    data['participants_per_session'] = participants_per_session
    data['tutor'] = tutor
    data['time'] = time
    data['utterances'] = utterances
    data.to_csv('dataset_features.csv', index = False)


def session_data():
    # Iterate through sessions
    for s in range(1, 31):

        # Reading session & engagement data
        session = pd.read_csv('Sessions/S' + str(s) + '.csv')
        start = session['start']
        end = session['end']
        length = np.array(end) - np.array(start)
        duration = end.tolist()[-1]
        engagementU = pd.read_csv('Engagement_by_Utterance/engagement_S' + str(s) + '.csv')
        engagementS = pd.read_csv('Engagement_by_Second/engagement_S' + str(s) + '.csv')

        # Getting participant lists with and without the tutor
        participants_notutor = engagementU.columns
        participants = session['speaker'].unique()
        tutor = [p for p in participants if p not in participants_notutor][0]
        tutor_list.append(tutor)
        ID_list.extend(participants)

        # Temporary session lists
        # Length (time) of utterance
        session_participant_length = []
        session_participant_notutor_length = []

        # Number of utterances per minute
        total_utterances.append(60 * len(session) / duration)
        total_notutor_utterances.append(60 * len(session[session['speaker'] != tutor]) / duration)
        session_participant_utterances = []
        session_participant_notutor_utterances = []

        # Time between utterances
        end_next = end.tolist()
        end_next.insert(0, 0)
        end_next.pop()
        inbetween = np.array(start) - np.array(end_next)
        time_between.append(np.mean(inbetween))
        time_between_notutor.append(np.mean(inbetween[session['speaker'] != tutor]))
        session_participant_wait_time = []
        session_participant_notutor_wait_time = []

        #Personality traits
        session_engagement_second = []
        session_engagement_utterance = []
        session_openness = []
        session_conscientiousness = []
        session_extraversion = []
        session_agreeableness = []
        session_neuroticism = []

        for p in participants:
            # Recording participant session number
            participant_session[p] = s

            # PERSONALITY
            [[agreeableness, conscientiousness, extraversion, neuroticism, openness]] = np.array(personality.loc[
                                                                                                     personality[
                                                                                                         'Participant ID'] == p, [
                                                                                                         'agreeableness',
                                                                                                         'conscientiousness',
                                                                                                         'extraversion',
                                                                                                         'neuroticism',
                                                                                                         'openness']])
            if not math.isnan(openness):
                session_openness.append(openness)
                session_conscientiousness.append(conscientiousness)
                session_extraversion.append(extraversion)
                session_agreeableness.append(agreeableness)
                session_neuroticism.append(neuroticism)

            # UTTERANCE LENGTH
            session_participant_length.append(np.mean(length[session['speaker'] == p]))

            # UTTERANCE PER MINUTE
            session_participant_utterances.append(len(session.loc[session['speaker'] == p]))
            
            # INBETWEEN TIME
            participant_data.loc[p, 'inbetween time'] = np.mean(inbetween[session['speaker'] == p])
            session_participant_wait_time.append(np.mean(inbetween[session['speaker'] == p]))

            # Recording the sans-tutor metrics
            if p in participants_notutor:
                #LENGTH
                participant_data.loc[p, 'length'] = np.mean(length[session['speaker'] == p])
                session_participant_notutor_length.append(np.mean(length[session['speaker'] == p]))
                
                #UTTERANCE PER MINUTE
                participant_data.loc[p, 'utterances per minute'] = 60 * len(session.loc[session['speaker'] == p]) / duration
                session_participant_notutor_utterances.append(len(session.loc[session['speaker'] == p]))
                
                #INBETWEEN TIME
                session_participant_notutor_wait_time.append(np.mean(inbetween[session['speaker'] == p]))

                #ENGAGEMENT
                session_engagement_second.append(np.mean(engagementS[p]))
                session_engagement_utterance.append(np.mean(engagementU[p]))
                participant_data.loc[p,'mean engagement by Utterance'] = np.mean(engagementU[p])
                participant_data.loc[p, 'mean engagement by Second'] = np.mean(engagementS[p])

        # PERSONALITY
        mean_openness.append(np.mean(session_openness))
        mean_conscientiousness.append(np.mean(session_conscientiousness))
        mean_extravesion.append(np.mean(session_extraversion))
        mean_agreeableness.append(np.mean(session_agreeableness))
        mean_neuroticism.append(np.mean(session_neuroticism))
        std_openness.append(np.std(session_openness))
        std_conscientiousness.append(np.std(session_conscientiousness))
        std_extravesion.append(np.std(session_extraversion))
        std_agreeableness.append(np.std(session_agreeableness))
        std_neuroticism.append(np.std(session_neuroticism))

        [[tutor_agreeableness, tutor_conscientiousness, tutor_extraversion, tutor_neuroticism, tutor_openness]] = np.array(personality.loc[
                                                                                                 personality[
                                                                                                     'Participant ID'] == tutor, [
                                                                                                     'agreeableness',
                                                                                                     'conscientiousness',
                                                                                                     'extraversion',
                                                                                                     'neuroticism',
                                                                                                     'openness']])
        '''tutor_openness_difference.append(np.mean(session_openness)- tutor_openness)
        tutor_conscientiousness_difference.append(np.mean(session_conscientiousness) - tutor_conscientiousness)
        tutor_extraversion_difference.append(np.mean(session_extraversion) - tutor_extraversion)
        tutor_agreeableness_difference.append(np.mean(session_agreeableness) - tutor_agreeableness)
        tutor_neuroticism_difference.append(np.mean(session_neuroticism) - tutor_neuroticism)'''
        tutor_openness_difference.append(tutor_openness)
        tutor_conscientiousness_difference.append(tutor_conscientiousness)
        tutor_extraversion_difference.append(tutor_extraversion)
        tutor_agreeableness_difference.append(tutor_agreeableness)
        tutor_neuroticism_difference.append(tutor_neuroticism)

        # Adding personality data to the participant dataframe
        for p in participants_notutor:
            other_participants_openness = session_openness.copy()
            other_participants_conscientiousness = session_conscientiousness.copy()
            other_participants_extraversion = session_extraversion.copy()
            other_participants_agreeableness = session_agreeableness.copy()
            other_participants_neuroticism = session_neuroticism.copy()

            if not math.isnan(participant_data.loc[p, 'openness']):
                other_participants_openness.remove(participant_data.loc[p, 'openness'])
                other_participants_conscientiousness.remove(participant_data.loc[p, 'conscientiousness'])
                other_participants_extraversion.remove(participant_data.loc[p, 'extraversion'])
                other_participants_agreeableness.remove(participant_data.loc[p, 'agreeableness'])
                other_participants_neuroticism.remove(participant_data.loc[p, 'neuroticism'])

            participant_data.loc[p, 'openness deviation'] = participant_data.loc[p, 'openness'] - np.mean(other_participants_openness)
            participant_data.loc[p, 'conscientiousness deviation'] = participant_data.loc[p, 'conscientiousness'] - np.mean(other_participants_conscientiousness)
            participant_data.loc[p, 'extraversion deviation'] = participant_data.loc[p, 'extraversion'] - np.mean(other_participants_extraversion)
            participant_data.loc[p, 'agreeableness deviation'] = participant_data.loc[p, 'agreeableness'] - np.mean(other_participants_agreeableness)
            participant_data.loc[p, 'neuroticism deviation'] = participant_data.loc[p, 'neuroticism'] - np.mean(other_participants_neuroticism)
            participant_data.loc[p, 'openness tutor difference'] = participant_data.loc[p, 'openness'] - tutor_openness
            participant_data.loc[p, 'conscientiousness tutor difference'] = participant_data.loc[p, 'conscientiousness'] - tutor_conscientiousness
            participant_data.loc[p, 'extraversion tutor difference'] = participant_data.loc[p, 'extraversion'] - tutor_extraversion
            participant_data.loc[p, 'agreeableness tutor difference'] = participant_data.loc[p, 'agreeableness'] - tutor_agreeableness
            participant_data.loc[p, 'neuroticism tutor difference'] = participant_data.loc[p, 'neuroticism'] - tutor_neuroticism

        # UTTERANCE LENGTH
        mean_participant_length.append(np.mean(session_participant_length))
        std_participant_length.append(np.std(session_participant_length))
        mean_participant_notutor_length.append(np.mean(session_participant_notutor_length))
        std_participant_notutor_length.append(np.std(session_participant_notutor_length))
        overall_length.append(np.mean(length))
        overall_notutor_length.append(np.mean(length[session['speaker'] != tutor]))
        
        # UTTERANCE PER MINUTE
        session_participant_utterances = 60 * np.array(session_participant_utterances) / duration
        session_participant_notutor_utterances = 60 * np.array(session_participant_notutor_utterances) / duration
        mean_participant_utterances.append(np.mean(session_participant_utterances))
        mean_participant_notutor_utterances.append(np.mean(session_participant_notutor_utterances))
        std_participant_utterances.append(np.std(session_participant_utterances))
        std_participant_notutor_utterances.append(np.std(session_participant_notutor_utterances))

        #INBETWEEN TIME
        mean_participant_wait_time.append(np.mean(session_participant_wait_time))
        mean_participant_notutor_wait_time.append(np.mean(session_participant_notutor_wait_time))
        std_participant_wait_time.append(np.std(session_participant_wait_time))
        std_participant_notutor_wait_time.append(np.std(session_participant_notutor_wait_time))

        # ENGAGEMENT
        engagement_second.append(np.mean(session_engagement_second))
        engagement_utterance.append(np.mean(session_engagement_utterance))

    #Adding all the session data to a dataframe and saving it
    session_data = pd.DataFrame()

    # Utterance length
    session_data['mean length'] = overall_length
    session_data['mean length no tutor'] = overall_notutor_length
    session_data['mean participant length'] = mean_participant_notutor_length
    session_data['std participant length'] = std_participant_length
    session_data['mean participant length no tutor'] = mean_participant_notutor_length
    session_data['std participant length no tutor'] = std_participant_notutor_length

    # Number of utterances (per minute)
    session_data['mean utterances per minute'] = total_utterances
    session_data['mean utterances per minute no tutor'] = total_notutor_utterances
    session_data['mean utterances per minute per participant'] = mean_participant_utterances
    session_data['std utterances per minute per participant'] = std_participant_utterances
    session_data['mean utterances per minute per participant no tutor'] = mean_participant_notutor_utterances
    session_data['std utterances per minute per participant no tutor'] = std_participant_notutor_utterances

    # Time between utterances
    session_data['mean inbetween time'] = time_between
    session_data['mean inbetween time no tutor'] = time_between_notutor
    session_data['mean participant inbetween time'] = mean_participant_wait_time
    session_data['std inbetween time'] = std_participant_wait_time
    session_data['mean participant inbetween time no tutor'] = mean_participant_notutor_wait_time
    session_data['std inbetween time no tutor'] = std_participant_notutor_wait_time

    # Engagement
    session_data['mean engagement by Second'] = engagement_second
    session_data['mean engagement by Utterance'] = engagement_utterance

    # Personality traits
    session_data['mean openness'] = mean_openness
    session_data['std openness'] = std_openness
    session_data['mean conscientiousness'] = mean_conscientiousness
    session_data['std conscientiousness'] = std_conscientiousness
    session_data['mean extraversion'] = mean_extravesion
    session_data['std extraversion'] = std_extravesion
    session_data['mean agreeableness'] = mean_agreeableness
    session_data['std agreeableness'] = std_agreeableness
    session_data['mean neuroticism'] = mean_neuroticism
    session_data['std neuroticism'] = std_neuroticism
    session_data['openness tutor difference'] = tutor_openness_difference
    session_data['conscientiousness tutor difference'] = tutor_conscientiousness_difference
    session_data['extraversion tutor difference'] = tutor_extraversion_difference
    session_data['agreeableness tutor difference'] = tutor_agreeableness_difference
    session_data['neuroticism tutor difference'] = tutor_neuroticism_difference
    
    

    session_data.to_csv('session_data.csv', index=False)
    participant_data.to_csv('participant_data.csv', index=True)


def repetition_data(option = '_TPL', type = 'Utterance'):
    self_repetition = []
    other_repetition = []
    self_repetition_notutor = []
    other_repetition_notutor = []
    self_repetition_participant = []
    other_repetition_participant = []
    self_repetition_participant_notutor = []
    other_repetition_participant_notutor = []
    for s in range(1, 31):
        # Reading session files
        repetition = pd.read_csv('Repetition_by_'+type+'/repetition_S' + str(s) + '.csv')
        speaking = pd.read_csv('Speaking_by_'+type+'/speaking_S' + str(s) + '.csv')
        engagement = pd.read_csv('Engagement_by_Utterance/engagement_S' + str(s) + '.csv')

        # Getting participant lists with and without the tutor
        participants_notutor = engagement.columns
        participants = speaking.columns

        # Temporary session lists
        session_self_repetition = []
        session_other_repetition = []
        session_self_repetition_notutor = []
        session_other_repetition_notutor = []
        session_self_repetition_participant = []
        session_other_repetition_participant = []
        session_self_repetition_participant_notutor = []
        session_other_repetition_participant_notutor = []

        # Iterating through session participants
        for p in participants:

            # Repetition scores while participant speaks
            session_self_repetition.extend(repetition.loc[speaking[p]==1,p + '_self_'+ option])
            session_other_repetition.extend(repetition.loc[speaking[p]==1,p + '_other_'+ option])

            # Average repetition score while speaking
            session_self_repetition_participant.append(np.mean(repetition.loc[speaking[p]==1,p + '_self_'+ option]))
            session_other_repetition_participant.append(np.mean(repetition.loc[speaking[p]==1,p + '_other_'+ option]))
            participant_repetition_data.loc[p, 'mean self-repetition'+option+'_by_' + type] = np.mean(repetition.loc[speaking[p]==1,p + '_self_'+ option])
            participant_repetition_data.loc[p, 'mean other-repetition' +option+'_by_' + type] = np.mean(repetition.loc[speaking[p]==1,p + '_other_'+ option])

            if p in participants_notutor:
                session_self_repetition_notutor.extend(repetition.loc[speaking[p] == 1, p + '_self_' + option])
                session_other_repetition_notutor.extend(repetition.loc[speaking[p] == 1, p + '_other_' + option])
                session_self_repetition_participant_notutor.append(
                    np.mean(repetition.loc[speaking[p] == 1, p + '_self_' + option]))
                session_other_repetition_participant_notutor.append(
                    np.mean(repetition.loc[speaking[p] == 1, p + '_other_' + option]))

        # Recording the average values for each measure
        self_repetition.append(np.mean(session_self_repetition))
        other_repetition.append(np.mean(session_other_repetition))
        self_repetition_notutor.append(np.mean(session_self_repetition_notutor))
        other_repetition_notutor.append(np.mean(session_other_repetition_notutor))
        self_repetition_participant.append(np.mean(session_self_repetition_participant))
        other_repetition_participant.append(np.mean(session_other_repetition_participant))
        self_repetition_participant_notutor.append(np.mean(session_self_repetition_participant_notutor))
        other_repetition_participant_notutor.append(np.mean(session_other_repetition_participant_notutor))

    # Adding values to dataframe and saving
    session_repetition_data['mean self-repetition'] = self_repetition
    session_repetition_data['mean other-repetition'] = other_repetition
    session_repetition_data['mean self-repetition no tutor'] = self_repetition_notutor
    session_repetition_data['mean other-repetition no tutor'] = other_repetition_notutor
    session_repetition_data['mean self-repetition per participant'] = self_repetition_participant
    session_repetition_data['mean other-repetition per participant'] = other_repetition_participant
    session_repetition_data['mean self-repetition per participant no tutor'] = self_repetition_participant_notutor
    session_repetition_data['mean other-repetition per participant no tutor'] = other_repetition_participant_notutor

    session_repetition_data.to_csv('Session_Repetition_Data/'+option+'_by_' + type + '.csv', index=False)


def participant_pairs():
    participant_pair_data = pd.DataFrame()
    pair = []
    engagement_by_Second = []
    engagement_by_Utterance = []
    mean_engagement_by_Utterance = []
    mean_engagement_by_Second = []
    openness_difference = []
    conscientiousness_difference = []
    extraversion_difference = []
    agreeableness_difference = []
    neuroticism_difference = []
    openness = []
    conscientiousness = []
    extraversion = []
    agreeableness = []
    neuroticism = []

    for s in range(1, 31):
        speakingS = pd.read_csv('Speaking_by_Second/speaking_S' + str(s) + '.csv')
        speakingU = pd.read_csv('Speaking_by_Utterance/speaking_S' + str(s) + '.csv')
        engagementS = pd.read_csv('Engagement_by_Second/engagement_S' + str(s) + '.csv')
        engagementU = pd.read_csv('Engagement_by_Utterance/engagement_S' + str(s) + '.csv')
        participants_notutor = engagementU.columns

        
        for p1 in participants_notutor:
            for p2 in participants_notutor:
                if p1 == p2:
                    continue
                pair.append(p1+'_'+p2)
                engagement_by_Utterance.append(np.mean(engagementU.loc[speakingU[p2]==1,p1]))
                engagement_by_Second.append(np.mean(engagementS.loc[speakingS[p2]==1, p1]))
                mean_engagement_by_Utterance.append(np.mean(engagementU[p1]))
                mean_engagement_by_Second.append(np.mean(engagementS[p1]))
                openness_difference.append(float(personality.loc[personality['Participant ID'] == p1, 'openness'])-float(personality.loc[personality['Participant ID'] == p2, 'openness']))
                conscientiousness_difference.append(float(personality.loc[personality['Participant ID'] == p1, 'conscientiousness'] )- float(personality.loc[personality['Participant ID'] == p2, 'conscientiousness']))
                extraversion_difference.append(float(personality.loc[personality['Participant ID'] == p1, 'extraversion'] )- float(personality.loc[personality['Participant ID'] == p2, 'extraversion']))
                agreeableness_difference.append(float(personality.loc[personality['Participant ID'] == p1, 'agreeableness'] )- float(personality.loc[personality['Participant ID'] == p2, 'agreeableness']))
                neuroticism_difference.append(float(personality.loc[personality['Participant ID'] == p1, 'neuroticism'] )- float(personality.loc[personality['Participant ID'] == p2, 'neuroticism'] ))
                openness.append(
                    float(personality.loc[personality['Participant ID'] == p1, 'openness']))
                conscientiousness.append(
                    float(personality.loc[personality['Participant ID'] == p1, 'conscientiousness']))
                extraversion.append(
                    float(personality.loc[personality['Participant ID'] == p1, 'extraversion']) )
                agreeableness.append(
                    float(personality.loc[personality['Participant ID'] == p1, 'agreeableness']))
                neuroticism.append(
                    float(personality.loc[personality['Participant ID'] == p1, 'neuroticism']) )

    participant_pair_data['pair'] = pair
    participant_pair_data['engagement by Utterance'] = engagement_by_Utterance
    participant_pair_data['engagement by Second'] = engagement_by_Second
    participant_pair_data['mean engagement by Utterance'] = mean_engagement_by_Utterance
    participant_pair_data['mean engagement by Second'] = mean_engagement_by_Second
    participant_pair_data['openness'] = openness
    participant_pair_data['conscientiousness'] = conscientiousness
    participant_pair_data['extraversion'] = extraversion
    participant_pair_data['agreeableness'] = agreeableness
    participant_pair_data['neuroticism'] = neuroticism
    participant_pair_data['openness difference'] = openness_difference
    participant_pair_data['conscientiousness difference'] = conscientiousness_difference
    participant_pair_data['extraversion difference'] = extraversion_difference
    participant_pair_data['agreeableness difference'] = agreeableness_difference
    participant_pair_data['neuroticism difference'] = neuroticism_difference
    participant_pair_data.to_csv('participant_pair_data.csv', index=False)


def plot_session():
    fig, ax = plt.subplots(3)
    ax[0].plot(range(1, 31), overall_length, label='Mean utterance time')
    ax[0].plot(range(1, 31), overall_notutor_length, label='Mean utterance time excluding tutor')
    ax[0].errorbar(range(1, 31), mean_participant_length, yerr=std_participant_length, label='Participant-weighted')
    ax[0].errorbar(range(1, 31), mean_participant_notutor_length, yerr=std_participant_notutor_length,
                   label='Participant-weighted, tutor excluded')

    ax[1].plot(range(1, 31), total_utterances, label='Mean utterance per second')
    ax[1].plot(range(1, 31), total_notutor_utterances, label='Mean utterance per second excluding tutor')
    ax[1].errorbar(range(1, 31), mean_participant_utterances, yerr=std_participant_utterances, label='Participant-weighted')
    ax[1].errorbar(range(1, 31), mean_participant_notutor_utterances, yerr=std_participant_notutor_utterances,
                   label='Participant-weighted, tutor excluded')

    ax[2].plot(range(1, 31), time_between, label='Time between utterances')
    ax[2].plot(range(1, 31), time_between_notutor, label='Time btween utterances excluding tutor')
    ax[2].errorbar(range(1, 31), mean_participant_wait_time, yerr=std_participant_wait_time, label='Participant-weighted')
    ax[2].errorbar(range(1, 31), mean_participant_notutor_wait_time, yerr=std_participant_notutor_wait_time,
                   label='Participant-weighted, tutor excluded')

    ax[0].legend(ncol=2)
    ax[1].legend(ncol=2)
    ax[2].legend(ncol=2)
    plt.show()


def engagement_full():
    engagement = pd.DataFrame()
    for s in range(1,31):
        engagementS = pd.read_csv('Engagement_by_Second/engagement_S' + str(s) + '.csv')
        engagement[s] = engagementS.mean(axis=1).groupby(engagementS.index // (len(engagementS)/ 500)).mean()
    # Display the resulting DataFrame
    engagement.to_csv("full_engagement.csv", index = False)


#dataset_features()
#engagement_full()
#session_data()
#plot_session()
for option in options:
    for type in ['Utterance','Second']:
        #print(option, type)
        #repetition_data(option,type)
        continue

#participant_repetition_data.to_csv('participant_repetition_data.csv', index=False)

#participant_pairs()


