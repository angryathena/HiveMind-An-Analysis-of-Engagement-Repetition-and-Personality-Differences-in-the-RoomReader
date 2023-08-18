import pandas as pd


answers = pd.read_csv('personality_answers.csv')
answers.set_index('Participant ID',inplace=True)
#answers.dropna(inplace=True)

grading = pd.read_csv('bfi.csv')
grading.set_index('statement', inplace=True)
grading = grading[['agreeableness', 'conscientiousness', 'extraversion', 'neuroticism','openness']]

personality = answers.dot(grading)
personality.to_csv('personality.csv')


for trait in personality.columns:
    personality[trait] = personality[trait] - min(personality[trait])
    personality[trait] = personality[trait].div(max(personality[trait])).round(2)
personality.to_csv('personality_normalised.csv')