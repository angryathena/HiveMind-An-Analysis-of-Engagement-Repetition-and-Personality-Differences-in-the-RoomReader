import numpy as np
import pandas
import pandas as pd
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import treetaggerwrapper as tt
import warnings

warnings.filterwarnings("ignore")

tagger = tt.TreeTagger(TAGLANG='en', TAGDIR='TreeTagger')
ADJECTIVE = ['AJ0', 'AJC', 'AJS']
ADVERB = ['AV0', 'AVP', 'AVQ']
ARTICLE = ['AT0']
CONJUNCTION = ['CJC', 'CJS', 'CJT', 'TO0']
DETERMINER = ['DPS', 'DT0', 'DTQ' ]
INTERJECTION = ['ITJ']
NOUN = ['NN0', 'NN1', 'NN2', 'NP0']
NUMERAL = ['CRD', 'ORD']
PRONOUN = ['PNI', 'PNP', 'PNQ', 'PNX', 'POS',]
PREPOSITION = ['PRF', 'PRP']
PUNCTUATION = ['PUL', 'PUN', 'PUQ', 'PUR','SENT']
UNCLASSIFIED = ['UNC']
VERB = ['VBD', 'VHD', 'VVD', 'VDD', 'VM0', 'VBI', 'VHI', 'VDI', 'VVG', 'VVI', 'VBG', 'VDG', 'VHG', 'VBN', 'VDN', 'VHN', 'VVN']

closed_class = ['AVP', 'AVQ','AT0','CJC', 'CJS', 'CJT', 'TO0','DPS', 'DT0', 'DTQ','CRD', 'ORD','PNI', 'PNP', 'PNQ',
                'PNX', 'POS','PRF', 'PRP','VBB','VBD','VBG','VBI','VBN','VBZ','VDB','VDD','VDG','VDI','VDN','VDZ','VHB','VHD','VHG','VHI','VHN','VHZ','VM0']
open_class = ['AV0','AJ0', 'AJC', 'AJS','AVO','ITJ','NN0', 'NN1', 'NN2', 'NP0',
              'VVB','VVD','VVG','VVI', 'VVN']

pronoun_transformation = {('i', 'PNP', 'i'): ['P', 'PNP', 'P'], ('me', 'PNP', 'i'): ['P', 'PNP', 'P'],('my', 'DPS', 'i'): ['D', 'DPS', 'D'], ('mine', 'PNP', 'mine'): ['PP', 'PNP', 'PP'],('myself', 'PNX', 'myself'): ['sR', 'PNX', 'sR'],
                          ('we', 'PNP', 'we'): ['P', 'PNP', 'sP'], ('us', 'PNP', 'we'): ['P', 'PNP', 'P'],('our', 'DPS', 'we'): ['D', 'DPS', 'D'], ('ours', 'PNP', 'ours'): ['PP', 'PNP', 'PP'],('ourselves', 'PNX', 'ourselves'): ['pR', 'PNX', 'pR'],
                          ('you', 'PNP', 'you'): ['P', 'PNP', 'sP'], ('your', 'DPS', 'you'): ['D', 'DPS', 'D'],('yours', 'PNP', 'yours'): ['PP', 'PNP', 'PP'],('yourself', 'PNX', 'yourself'): ['sR', 'PNX', 'sR'],
                          ('yous', 'PNP', 'yous'): ['P', 'PNP', 'P'],('yourselves', 'PNX', 'yourselves'): ['pR', 'PNX', 'pR']}
options = {'_TPL':[1,1,1,1,1],'_TPL_open':[1,1,1,0,1],'_TPL_closed':[1,1,1,1,0],'_TP':[1,1,0,1,1],'_PL':[0,1,1,1,1],'_T':[1,0,0,1,1],'_P':[0,1,0,1,1],'_L':[0,0,1,1,1],
           '_TP_closed':[1,1,0,1,0],'_PL_closed':[0,1,1,1,0],'_T_closed':[1,0,0,1,0],'_P_closed':[0,1,0,1,0],'_L_closed':[0,0,1,1,0],
           '_TP_open':[1,1,0,0,1],'_PL_open':[0,1,1,0,1],'_T_open':[1,0,0,0,1],'_P_open':[0,1,0,0,1],'_L_open':[0,0,1,0,1],}

#closed_class = [PRONOUN,ARTICLE,PREPOSITION,DETERMINER,CONJUNCTION,]
#open_class = [NOUN,ADJECTIVE,VERB,ADVERB,INTERJECTION,UNCLASSIFIED, NUMERAL]

def preprocessing():
    file_list = glob.glob('./transcriptions txt/*.txt')
    timestamps = []
    for file in file_list:
        # file = './transcriptions txt/S01.txt'
        session = int(file.split('S')[1].split('.')[0])
        line_list = []
        with open(file, 'r') as f:
            lines = f.readlines()[1:]
        for line in np.copy(lines):
            fields = line.split('\t')
            if fields[0] != 'Section' and fields[0] != 'Sections' and not 'Para' in fields[0] and not 'Check' in fields[
                0] and not fields[11] == "\n":
                fields[11] = fields[11].replace("\n", '')
                fields[3] = float(fields[3])
                line_list.append(fields)
        lines = line_list
        lines_T = np.array(lines).T.tolist()
        df = pd.DataFrame()
        # session_data['id'] =lines_T[0]
        df['speaker'] = lines_T[1]
        df['start'] = lines_T[3]
        df['end'] = lines_T[6]
        df['duration'] = lines_T[9]
        df['content'] = lines_T[11]
        df['start'] = df['start'].astype("float")
        df.sort_values(by='start', inplace=True)
        print(session)
        df.to_csv('Sessions/S' + str(session) + '.csv', index=False)


def proces_tags(tags):
    for t, tag in enumerate(tags):
        try:
            # bringing each word to lowercase
            tag[0] = tag[0].lower()
            tag = tuple(tag)
            # Converting first and second person pronouns to the same tag.
            tags[t] = pronoun_transformation[tag]
        except:
            # Skipping words that are not first or second person pronouns
            True


def distance(texts, option = [1,1,1,1,1]):
    [token, part_of_speech, lemma, closed, open] = option
    tags = [tagger.tag_text(text) for text in texts]
    tags = [[tag.split('\t') for tag in tagset] for tagset in tags]
    for tag in tags:
        proces_tags(tag)

    exclusions = PUNCTUATION.copy()
    if not closed:
        exclusions.extend(closed_class)
    if not open:
        exclusions.extend(open_class)
    if token and part_of_speech and lemma:
        tags = [[tag for tag in tagset if tag[1] not in exclusions] for tagset in tags]
    elif token and part_of_speech:
        tags = [[tag[0:2] for tag in tagset if tag[1] not in exclusions] for tagset in tags]
    elif lemma and part_of_speech:
        tags = [[tag[1:3] for tag in tagset if tag[1] not in exclusions] for tagset in tags]
    elif part_of_speech:
        tags = [[tag[1] for tag in tagset if tag[1] not in exclusions] for tagset in tags]
    elif token:
        tags = [[tag[0] for tag in tagset if tag[1] not in exclusions] for tagset in tags]
    elif lemma:
        tags = [[tag[2] for tag in tagset if tag[1] not in exclusions] for tagset in tags]


    set1, set2 = tags[0], tags[1]
    # Computing Jaccard coefficient between the two utterances represented as two tag sets
    intersection = [tuple(element) for element in set1 if element in set2]
    union = [tuple(element) for element in set1 + set2]

    try:
        return len(set(intersection)) / len(set(union))
    except:
        return 0


def repetition():
    for session in range(30,31):
        print(session)
        # Read session files
        df = pd.read_csv('Sessions/S' + str(session) + '.csv')
        engagement = pd.read_csv('Engagement_by_Second/engagement_S' + str(session) + '.csv')
        # List of all speakers in session
        speakers = df['speaker'].unique()

        # Initialise repetition dataframe at 0 for each second that has an engagementU value recorded
        repetition = pd.DataFrame()
        for speaker in speakers:
            for option in options:
                for type in ['_self','_other']:
                    repetition[speaker+type + option] = [0.0] * len(engagement)

        repetition_utt  = pd.DataFrame()
        for speaker in speakers:
            for option in options:
                for type in ['_self','_other']:
                    repetition_utt[speaker+type + option] = [0.0] * len(df)

        # Initialise the register which will update with each utterance
        register = {}
        for speaker in speakers:
            register[speaker] = ''

        # Iterate through session lines
        for ind in df.index:
            # Take start and end time to update the entire span
            start = int(float(df['start'][ind]) + 0.5)
            end = int(float(df['end'][ind]) + 1.5)

            # Iterate through all speakers in the register, including tutor
            for speaker in speakers:
                # Compute score between current utterance and the last utterance of each speaker
                for option in options:
                    score = distance([register[speaker], df['content'][ind]], options[option])
                    # Add score to self or other repetition count.
                    if df['speaker'][ind] == speaker:
                        repetition_utt[(df['speaker'][ind] + '_self' + option)][ind] +=score
                        for i in range(start, end):
                            try:
                                repetition[(df['speaker'][ind]+'_self'+option)][i] += score
                            except:
                                True
                    else:
                        score = score/(len(speakers)-1)
                        repetition_utt[(df['speaker'][ind] + '_other' + option)][ind] += score
                        for i in range(start, end):
                            try:
                                repetition[(df['speaker'][ind]+'_other'+option)][i] += score
                            except:
                                True
            # Update register
            register[df['speaker'][ind]] = df['content'][ind]

        repetition.to_csv('Repetition_by_Second/repetition_S'+str(session)+'.csv', index = False)
        repetition_utt.to_csv('Repetition_by_Utterance/repetition_S' + str(session) + '.csv', index=False)


def speaking():
    for session in range(30, 31):
        print(session)
        # Read session files
        df = pd.read_csv('Sessions/S' + str(session) + '.csv')
        engagement = pd.read_csv('Engagement_by_Second/engagement_S' + str(session) + '.csv')

        # List of all speakers in session
        speakers = df['speaker'].unique()

        # Initialise repetition dataframe at 0 for each second that has an engagementU value recorded
        speaking = pd.DataFrame()
        for speaker in speakers:
                speaking[speaker] = [0] * len(engagement)

        speaking_utt = pd.DataFrame()
        for speaker in speakers:
                speaking_utt[speaker] = [0] * len(df.index)

        # Iterate through session lines
        for ind in df.index:
            speaking_utt[df['speaker'][ind]][ind] = 1
            # Take start and end time to update the entire span
            start = int(float(df['start'][ind]) + 0.5)
            end = int(float(df['end'][ind]) + 1.5)

            # Record speaking times
            for i in range(start, end):
                speaking[df['speaker'][ind]][i] = 1

        speaking_utt.to_csv('Speaking_by_Utterance/speaking_S' + str(session) + '.csv', index=False)
        speaking.to_csv('Speaking_by_Second/speaking_S' + str(session) + '.csv', index=False)

# Generate files for each session ordered by start time of the utterance
#preprocessing()

# Iterate through all sessions to record repetitions
repetition()

#Iterate through sessions and record speaking
speaking()
