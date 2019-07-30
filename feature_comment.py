import pandas as pd
import fasttext, re, json, random, pickle
from pyvi.ViTokenizer import tokenize
import numpy as np
from string import punctuation
from langdetect import detect
from sklearn.datasets import dump_svmlight_file
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

negative_emoticons = {':(', '☹', '❌', '👎', '👹', '💀', '🔥', '🤔', '😏', '😐', '😑', '😒', '😓', '😔', '😕', '😖',
                      '😞', '😟', '😠', '😡', '😢', '😣', '😤', '😥', '😧', '😨', '😩', '😪', '😫', '😭', '😰', '😱',
                      '😳', '😵', '😶', '😾', '🙁', '🙏', '🚫', '>:[', ':-(', ':(', ':-c', ':c', ':-<', ':っC', ':<',
                      ':-[', ':[', ':{', ':((', ':((('}

positive_emoticons = {'=))', 'v', ';)', '^^', '<3', '☀', '☺', '♡', '♥', '✌', '✨', '❣', '❤', '🌝', '🌷', '🌸',
                      '🌺', '🌼', '🍓', '🎈', '🐅', '🐶', '🐾', '👉', '👌', '👍', '👏', '👻', '💃', '💄', '💋',
                      '💌', '💎', '💐', '💓', '💕', '💖', '💗', '💙', '💚', '💛', '💜', '💞', ':-)', ':)', ':D', ':o)',
                      ':]', ':3', ':c)', ':>', '=]', '8)', ':)))', '😆', ':))'}

def normalize_rating(df):
    df['rating'] = df['rating'].apply(lambda x: remove_label(x))
    df['rating'] = df['rating'].apply(lambda x: int(round(float(x) / 2)))
    return df

def remove_label(x):
    if x == 1.0:
        return 1.1
    else:
        return x

def string_format(_string):
    try:
        if detect(_string) == 'vi':
            _string = _string.lower()
            _string = re.sub(r"([.!?,'/()])", r" \1 ", _string)
        else:
            _string = ''
    except Exception:
        _string = _string.lower()
        _string = re.sub(r"([.!?,'/()])", r" \1 ", _string)
    return _string
    
def tokenize_content(_string):
    _string = string_format(_string)
    _string =  tokenize(_string)
    # for token in _string.split(' '):
    #     if token in positive_emoticons:
    #         _string = _string.replace(token, 'pos')
    #     if token in negative_emoticons:
    #         _string = _string.replace(token, 'neg')
    #     if token in punctuation:
    #         _string = _string.replace(token, 'punc')
    return _string

def get_equal_train(df):
    df_list = {}
    train = []
    for i in range(5,11):
        df_list[i] = df[df['rating'] == i]
        train.append(df_list[i])

    train_df = pd.concat(train, axis=0)
    train_df.to_json('data_train.json', orient='records')



def prepare_data_fasttext(filepath):
    df = pd.read_json(filepath)
    df = df.drop(list(df[df['rating'].isnull()].index), axis =0)
    df = df.sample(frac=1)
    df = normalize_rating(df)
    df['comment'] = df['comment'].apply(lambda x: tokenize_content(x))
    df = df.drop(list(df[df['comment'] == ''].index), axis =0)
    df.to_json('data_tokenized.json', orient='records')
    comments = list(df['comment'])
    ratings = list(df['rating'])
    with open('train_emo.txt', 'w') as ftrain, open('test_emo.txt', 'w') as ftest:
        for i in range(len(comments)):
            fasttext_line = "__label__{} {}".format(ratings[i], comments[i].strip())
            if random.random() <= 0.10:
                ftest.write(fasttext_line+'\n')
            else:
                ftrain.write(fasttext_line+'\n')
    ftrain.close()
    ftest.close()

def prepare_data_starspace(filepath):
    df = pd.read_json(filepath)
    df = df.drop(list(df[df['rating'].isnull()].index), axis =0)
    df = normalize_rating(df)
    df['comment'] = df['comment'].apply(lambda x: tokenize_content(x))
    df.to_json('data_tokenized_emo.json', orient='records')
    comments = list(df['comment'])
    ratings = list(df['rating'])
    with open('train_emo.txt', 'w') as ftrain:
        for i in range(len(comments)):
            fasttext_line = "__label__{} {}".format(ratings[i], comments[i].strip())
            ftrain.write(fasttext_line+'\n')
    ftrain.close()

def create_stopwords():
    stopwords = []
    with open('stopwords.txt', 'r') as f:
        for line in f.readlines():
            word = line.strip()
            stopwords.append('_'.join(word.split(' ')))
    f.close()
    return stopwords

def check_model(review):
    prep_review = tokenize_content(review)

    classifier = fasttext.load_model('reviews_model.bin')
    label, prob = classifier.predict(prep_review, 1)
    label = label[0][9:]
    # print(label)
    if int(label) != 5:
        print('score from {} to {} with {}% confidence'.format(int(label) * 2 - 1, int(label) * 2 + 1, int(prob[0] * 100)))
        print()
    else:
        print('score from {} to {} with {}% confidence'.format(int(label) * 2 - 1, int(label) * 2, int(prob[0] * 100)))
        print()

stopwords = create_stopwords()

# prepare_data_fasttext('commentCrawler/data_new.json')
temp = input("Start: ")
while temp not in ["n", "N"]:
    review = input("Write review to predict score range: ")
    check_model(review)
    temp = input("Start: ")




# df = pd.read_json('data_tokenized.json')
# train_X, test_X, train_Y, test_Y = train_test_split(df['comment'],df['rating'],test_size=0.3, stratify = df['rating'])
# TfidfVect = TfidfVectorizer(stop_words=stopwords)
# TfidfVect.fit(train_X)

# train_X = TfidfVect.transform(train_X)
# test_X = TfidfVect.transform(test_X)
# dump_svmlight_file(train_X, train_Y, 'train_liblinear.txt', zero_based=False)
# dump_svmlight_file(test_X, test_Y, 'test_liblinear.txt',zero_based=False)
# print('done dump to file')
