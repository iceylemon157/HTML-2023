import os
import numpy as np
import pandas as pd
# import tensorflow as tf
import sklearn
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
# from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.ensemble import ExtraTreesRegressor
# import sklearn.preprocessing.s

def read_data():
    df = pd.read_csv('train.csv')

    print(list(df.keys()[:18]))

    dic = {}

    for key in list(df.keys()[:18]) + ['Comments']:
        if key[0] == 'D':
            continue
        x = df[key]

        nan_rows = df[df[key].isnull()]
        for id in nan_rows['id']:
            dic[id] = 1

    print('len', len(df['id']) - len(dic))

def new_data(filename):
    '''
    Remove the columns that are bad
    ['Danceability', 'Energy', 'Key', 'Loudness', 'Speechiness',
       'Acousticness', 'Instrumentalness', 'Liveness', 'Valence', 'Tempo',
       'Duration_ms', 'Views', 'Likes', 'Stream', 'Album_type', 'Licensed',
       'official_video', 'id', 'Track', 'Album', 'Uri', 'Url_spotify',
       'Url_youtube', 'Comments', 'Description', 'Title', 'Channel',
       'Composer', 'Artist'].

    '''

    all_keys = ['Danceability', 'Energy', 'Key', 'Loudness', 'Speechiness',
       'Acousticness', 'Instrumentalness', 'Liveness', 'Valence', 'Tempo',
       'Duration_ms', 'Views', 'Likes', 'Stream', 'Album_type', 'Licensed',
       'official_video', 'id', 'Track', 'Album', 'Uri', 'Url_spotify',
       'Url_youtube', 'Comments', 'Description', 'Title', 'Channel',
       'Composer', 'Artist']

    df = pd.read_csv(filename + '.csv')
    new_df = df.drop(all_keys[-11:], axis=1)

    to_drop = ['Key', 'Album_type', 'Licensed', 'official_video', 'Likes', 'Views', 'id', 'Stream']
    # to_drop = ['Danceability', 'Loudness', 'Liveness', 'Tempo', 'Instrumentalness', 'Energy']

    new_df = new_df.drop(to_drop, axis=1)

    # new_df = df[['Danceability', 'Energy', 'Loudness', 'Speechiness', 'Liveness', 'Tempo']]

    new_df.to_csv('new_' + filename + '.csv', index=False)

def fix_missing_data(filename):

    new_data(filename)
    

    if os.path.exists('imputed_' + filename + '.npy') and os.path.exists('imputed_' + filename + '.csv'):
        print('both file already exists')
        try:
            with open('keys.txt', 'r') as f:
                print('keys in latest file:', f.read())
        except:
            print('keys.txt does not exist')
        print('new keys to replace:', list(pd.read_csv('new_' + filename + '.csv').keys()))
        opt = input('do you want to overwrite? (y/n): ')
        if opt != 'y':
            return


    df = pd.read_csv('new_' + filename + '.csv')
    # print('here is the fucking keys', df.keys())
    imputer = IterativeImputer(estimator=ExtraTreesRegressor(max_features='sqrt'))
    imputer.fit(df)
    imputed_df = imputer.transform(df)
    # print(imputed_df)

    # print('what is this', imputed_df[0])

    # print(imputed_df[:, 0].shape, imputed_df[:, 1:].shape)

    # mapped to 0-1
    scaled_data = sklearn.preprocessing.scale(imputed_df[:, 1:])
    imputed_df = np.concatenate((imputed_df[:, 0].reshape(-1, 1), scaled_data), axis=1)

    np.save('imputed_' + filename + '.npy', imputed_df)

    new_df = pd.DataFrame(imputed_df, columns=df.keys())
    new_df.to_csv('imputed_' + filename + '.csv', index=False)

    with open('keys.txt', 'w') as f:
        f.write(str(list(new_df.keys())))

if __name__ == '__main__':
    fix_missing_data('train')

