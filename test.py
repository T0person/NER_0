import spacy
from main import load_csv
from os import getenv
import random

nlp = spacy.load("./output/model-best")

path = getenv("TRAIN_PATH")

df = load_csv('test.csv')  # Начальный (тренировочный) датафрейм

print(f'Введите строку или введите 0:')

while True:
    text = input()
    
    if text == '0':
        id = random.randint(0, len(df))
        text = df.loc[id]['text']
        # print(df.iloc[id]['essence'])
        # print(text)
    
    doc = nlp(text)    

    for ent in doc.ents:
        print(ent.text, ent.start_char, ent.end_char, ent.label_)