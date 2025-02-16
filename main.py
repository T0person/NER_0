from ctypes import alignment
import string
import pandas as pd
import re
from os import getenv
import spacy
from spacy.tokens import DocBin


def load_csv(_path: str) -> pd.DataFrame:
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
    
    _total = list(re.finditer(_reg, _text))
    _data = []
        
    for _essence in _total:
    
        _data.append((_essence.start(), _essence.end(), essence))
            
        _doc = (_text, _data)

        _training_data.append(_doc)

    return _training_data


def find_essences(_df: pd.DataFrame):
    _df_lower = _df.copy()
    _df_lower['essence'] = _df_lower['essence'].str.lower()
    _df_lower['text'] = _df_lower['text'].str.lower()

    # Проход по всему датафрейму
    for id, row in _df_lower.iterrows():

        row['text'] = row['text'].replace('&#32;', ' ')
        row['text'] = row['text'].replace('  ', ' ')
        
        _first_word = f"\\b.*?{row['essence']}(?:.*? -|.*? —)"
        _first_word_text = re.match(_first_word, row['text'])
        
        
        # Регулярные выражения для переборов сущностей
        reg_with_dash = f"\\b{row['essence']}\\b —|\\b{row['essence']}\\b -"
        reg_dash = f'(?:{row['essence']}\\b.+? —|{row['essence']}\\b.+? -)'
        reg_mention = f"\\b{row['essence']}(?:\\b|.*?\\b)"
        reg_primary = f"\\b{row['essence']}(?:\\b|.*?\\b)"
        reg_prefix = f"\\b\\w+?{row['essence']}.*?\\b" # Слова с приставкой
        reg_prefix_with_dash = f"\\b\\w+?{row['essence']}.*?\\b -|\\b\\w+?{row['essence']}.*?\\b —"


        # Для главной сущности выбирается первое вхождение, поэтому разбиваем на смысловые предложения
        sub_text = row['text'].split(',')

        # Поиск регулярного выражения
        total_text_with_dash = re.search(reg_with_dash, row['text'])  # Ищет начало
        total_text_dash = re.search(reg_dash, row['text'])  # Ищет сущность и - (большое тире)
        total_text_braked = re.search(f"\\({row['essence']}\\)", sub_text[0])  # Ищет скобку в тексте
        total_text_prefix = re.search(reg_prefix, row['text']) # Ищет слово с приставкой
        total_text_prefix_with_dash = re.search(reg_prefix_with_dash, row['text']) # Ищет слово с приставкой

        # Разбиение имени и фамилии
        total_text_fio = None
        total_split_fio = row['essence'].split()
        
        if len(total_split_fio) == 2:
            fio = f'{total_split_fio[0]} \\s\\w{2,}?\\s {total_split_fio[1]}'
            total_text_fio = re.search(fio, row['text'])

        if row['essence'] == 'циан':
            print('lelel')
        if _first_word_text:
            _generate_training_data(reg_primary, row['text'])
        # # Первый проход - Ищет сущность и —/- (длинный/короткий дефис)
        # if total_text_with_dash:
        #     _generate_training_data(reg_primary, row['text'])


        # # Третий проход с поиском зависимых слов и —/- (длинный/короткий дефис)
        # elif total_text_dash:
        #     _generate_training_data(reg_primary, row['text'])


        elif total_text_braked:
            _generate_training_data(reg_primary, row['text'])

        elif total_text_fio:
            _generate_training_data(total_text_fio.group(), row['text'])

        # Вычисляем те строки, которые не находятся и вставляем сущность
        elif row['essence'] not in row['text']:
            # Изменяем датасет и добавляем к основным сущностям
            _temp_text = f"{row['essence']} - {row['text']}"
            _generate_training_data(row['essence'], _temp_text)

        elif total_text_prefix_with_dash:
            _generate_training_data(reg_prefix, row['text'])
            
        elif total_text_prefix:
            _temp_text = f"{row['essence']} - {row['text']}"
            _generate_training_data(reg_primary, _temp_text)
            
        else:
            print(reg_mention)
            _generate_training_data(reg_mention, row['text'], essence='MENTION')
    return _generate_training_data(None)


if __name__ == "__main__":
    train_path = getenv("TRAIN_PATH")

    train_df = load_csv(train_path)  # Начальный (тренировочный) датафрейм

    train_df = drop_duplicate(train_df)  # Удаление дубликатов

    training_data = find_essences(train_df)

    nlp = spacy.blank("xx")

    db = DocBin()
    for text, annotations in training_data:
        doc = nlp(text)
        ents = []
        if len(annotations) > 1:
            lol =1
        for start, end, label in annotations:
            print(text[start:end])
            span = doc.char_span(start, end, label=label, alignment_mode='contract')
            ents.append(span)
        doc.ents = ents
        db.add(doc)
    db.to_disk("./train.spacy")

    # primary_df, mentions_df = filling_ids(train_df) # Основные сущности и дополнительные

    # train_mentions = create_train_data(mentions_df) # Тренировочная выборка

    # train_df.loc[train_df.duplicated(),:] # Проверка дубликатов
