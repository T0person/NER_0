from ctypes import alignment
import os
import string
import pandas as pd
import re
from os import getenv
import spacy
from spacy.tokens import DocBin


def load_csv(_path) -> pd.DataFrame:
    """
    Функция открытия "csv" файла с определенными названиями столбцов

    Args:
        _path (str): Путь к файлу

    Returns:
        pd.DataFrame: DataFrame
    """

    _df = pd.read_csv(_path, names=['id', 'essence', 'text'], encoding='utf-8').reset_index(drop=True)

    return _df


def drop_duplicate(_df: pd.DataFrame) -> pd.DataFrame:
    """
    Функция удаления дубликатов в датафрейме

    Args:
        _df (pd.DataFrame): Датафрейм

    Returns:
        pd.DataFrame: Датафрейм без дубликатов
    """
    last = _df.shape
    _df = _df.drop_duplicates(subset=['essence'], keep='last')
    new = _df.shape
    print("Дубликаты удалены!") if new != last else print("Дубликаты не найдены!")
    return _df


def _generate_training_data(_reg : str, _text='', _training_data=[], essence="PRIMARY") -> list:
    """
    Функция подготовки данных для обучения Spacy

    Args:
        _reg (re.Match): Регулярка с сущностью
        _training_data (list): Список тренировочных данных

    Returns:
        list: Список размеченных сущностей
    """
        
    if _reg == None:
        return _training_data
    
    _total = list(re.finditer(_reg, _text, re.I))
    _data = []

    if len(_total) == 0:
        lol =1
    for _essence in _total:
        _data.append((_essence.start(), _essence.end(), essence))
            
    _doc = (_text, _data)

    _training_data.append(_doc)

    return _training_data


def find_essences(_df: pd.DataFrame, create_test_file=False):

    # Проход по всему датафрейму
    for id, row in _df.iterrows():

        row['text'] = row['text'].replace('&#32;', ' ')
        row['text'] = row['text'].replace('  ', ' ')
        
        _first_word = f"\\b.*?{row['essence']}(?:.*? -|.*? —)"
        _first_word_text = re.match(_first_word, row['text'])
        
        # Регулярные выражения для переборов сущностей
        reg_essence = f"\\b{row['essence']}(?:[-.]?\\w+?\\b|\\w*?\\b)"
        reg_prefix = f"\\b\\w+?{row['essence']}.*?\\b" # Слова с приставкой
        reg_prefix_with_dash = f"\\b\\w+?{row['essence']}.*?\\b -|\\b\\w+?{row['essence']}.*?\\b —"


        # Для главной сущности выбирается первое вхождение, поэтому разбиваем на смысловые предложения
        sub_text = row['text'].split(',')

        # Поиск регулярного выражения
        total_text_braked = re.search(f"\\({row['essence']}\\)", sub_text[0], re.I)  # Ищет скобку в тексте
        total_text_prefix = re.search(reg_prefix, row['text'], re.I) # Ищет слово с приставкой
        total_text_prefix_with_dash = re.search(reg_prefix_with_dash, row['text'], re.I) # Ищет слово с приставкой

        # Разбиение имени и фамилии
        total_text_fio = None
        total_split_fio = row['essence'].split()
        if "Гурзуф" in row['essence'] or "Брюссель" in row['essence'] or "Pfizer" in row['essence']:
            lol =1
        if len(total_split_fio) == 2:
            fio = f'{total_split_fio[0]} \\s\\w{2,}?\\s {total_split_fio[1]}'
            total_text_fio = re.search(fio, row['text'], re.I)

        if _first_word_text:
            _generate_training_data(reg_essence, row['text'])

        elif total_text_braked:
            _generate_training_data(reg_essence, row['text'])

        elif total_text_fio:
            _generate_training_data(total_text_fio.group(), row['text'])

        # Вычисляем те строки, которые не находятся и вставляем сущность
        elif row['essence'].lower() not in row['text'].lower():
            # Изменяем датасет и добавляем к основным сущностям
            _temp_text = f"{row['essence']} - {row['text']}"
            # print(_df.loc[id]['text'])
            _df.at[id, 'text'] = _temp_text
            # _df.iloc[id]['text'] = _temp_text
            if "+" in row['essence']:
                row['essence'] = row['essence'].replace("+", "[\\+]")
            _generate_training_data(row['essence'], _temp_text)

        elif total_text_prefix_with_dash:
            _generate_training_data(reg_prefix, row['text'])
            
        elif total_text_prefix:
            _generate_training_data(reg_prefix, row['text'])
            
        else:
            _generate_training_data(reg_essence, row['text'], essence='MENTION')
    if create_test_file:
        _df.to_csv("./test.csv", header=True, index=False)
    return _generate_training_data(None)

def create_cfg(_data, _file_path):
    nlp = spacy.blank("xx")

    db = DocBin()
    for text, annotations in _data:
        doc = nlp(text)
        ents = []
        for start, end, label in annotations:
            span = doc.char_span(start, end, label=label, alignment_mode='contract')
            ents.append(span)
        doc.ents = ents
        db.add(doc)
    db.to_disk(_file_path)
    

if __name__ == "__main__":
    train_path = getenv("TRAIN_PATH")
    test_path = getenv("TEST_PATH")

    train_df = load_csv(train_path)  # Начальный (тренировочный) датафрейм

    train_df = drop_duplicate(train_df)  # Удаление дубликатов

    training_data = find_essences(train_df, create_test_file=True)

    create_cfg(training_data, "./train.spacy")
    
    test_df = load_csv(test_path)  # Начальный (тренировочный) датафрейм

    test_df = drop_duplicate(test_df)  # Удаление дубликатов

    test_data = find_essences(test_df)
    
    create_cfg(test_data, "./dev.spacy")
    
