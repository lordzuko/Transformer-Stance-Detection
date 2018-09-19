import os
import codecs
import csv
import numpy as np
import pandas as pd

from tqdm import tqdm
from readwrite import reader
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from utils import iter_data
seed = 3535999445



def stance(data_dir, n_valid=0.05):
    map_st = {
        'NONE':2,
        'AGAINST': 1,
        'FAVOR':0
    }

    text, target, st, _ = reader.readTweetsOfficial(os.path.join(data_dir,'semeval2016-task6-train+dev.txt'))
    test_X1, test_X2, test_Y, _ = reader.readTweetsOfficial(os.path.join(data_dir,'semeval2016-task6-testdata-gold/SemEval2016-Task6-subtaskA-testdata-gold.txt'))
    test_Y = [map_st[y] for y in test_Y]

    tr_target, va_target, tr_text, va_text, tr_st, va_st  = train_test_split(target, text, st, test_size=n_valid, random_state=seed)

    trX1, trX2 = [], []
    trY = []

    for x1,x2,y in zip(tr_text, tr_target, tr_st):
        trX1.append(x1)
        trX2.append(x2)
        trY.append(map_st[y])

    vaX1, vaX2 = [], []
    vaY = []

    for x1, x2, y in zip(va_text, va_target, va_st):
        vaX1.append(x1)
        vaX2.append(x2)
        vaY.append(map_st[y])

    trY = np.asarray(trY, dtype=np.int32)
    vaY = np.asarray(vaY, dtype=np.int32)
    test_Y = np.asarray(test_Y, dtype=np.int32)

    return (trX1, trX2, trY), (vaX1, vaX2, vaY), (test_X1, test_X2) #, test_Y)


def create_pretraing_data_file():
    embed_train_file = 'data/dataset_tweet_encode/prepared_data_embed_training.txt'

    if not os.path.exists(embed_train_file):
        tweets, targets, labels, ids = reader.readTweetsOfficial(
            "data/semeval2016-task6-train+dev.txt")
        tweets_trump, targets_trump, labels_trump, ids_trump = reader.readTweetsOfficial(
            "data/downloaded_Donald_Trump.txt", "utf-8", 1)
        tweets_unlabelled = reader.readTweets("data/additionalTweetsStanceDetectionBig.json")

        with codecs.open(embed_train_file, 'w') as w:
            for tweet in tqdm(tweets, desc='semeval: ', total=len(tweets)):
                w.write(''.join(tweet) + '\n')
            for tweet in tqdm(tweets_trump, desc='trump: ', total=len(tweets_trump)):
                w.write(''.join(tweet) + '\n')
            for tweet in tqdm(tweets_unlabelled, desc='tweet dump: ', total=len(tweets_unlabelled)):
                w.write(''.join(tweet) + '\n')

if __name__ == "__main__":
    create_pretraing_data_file()