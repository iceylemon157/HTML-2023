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
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer

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

    to_drop = ['Key', 'Album_type', 'Licensed', 'official_video', 'Likes', 'Views', 'id', 'Stream', 'Duration_ms']
    # to_drop = ['Danceability', 'Loudness', 'Liveness', 'Tempo', 'Instrumentalness', 'Energy']

    new_df = new_df.drop(to_drop, axis=1)

    # new_df = df[['Danceability', 'Energy', 'Loudness', 'Speechiness', 'Liveness', 'Tempo']]

    new_df.to_csv('new_' + filename + '.csv', index=False)

def evil_data_pruning(filename):

    all_keys = ['Danceability', 'Energy', 'Key', 'Loudness', 'Speechiness',
       'Acousticness', 'Instrumentalness', 'Liveness', 'Valence', 'Tempo',
       'Duration_ms', 'Views', 'Likes', 'Stream', 'Album_type', 'Licensed',
       'official_video', 'id', 'Track', 'Album', 'Uri', 'Url_spotify',
       'Url_youtube', 'Comments', 'Description', 'Title', 'Channel',
       'Composer', 'Artist']

    df = pd.read_csv(filename)

    print(df.shape)
    df.drop(df[df['Liveness'] > 10].index, inplace=True)
    print(df.shape)
    df.drop(df[df['Tempo'] < -2].index, inplace=True)
    print(df.shape)
    # df = df.drop(df[df['Tempo'] > 200].index, inplace=True)
    df.drop(df[df['Speechiness'] > 8].index, inplace=True)
    print(df.shape)
    df.drop(df[df['Loudness'] < -6].index, inplace=True)
    print(df.shape)
    df.drop(df[df['Instrumentalness'] > 10].index, inplace=True)
    print(df.shape)

    df.to_csv('evil_' + filename, index=False)

def fix_missing_data_train(filename='train'):

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
    imputer = IterativeImputer()
    imputer.fit(df)
    imputed_df = imputer.transform(df)
    print(imputed_df.shape)
    # print(imputed_df)

    print('what is this', imputed_df[0])

    # print(imputed_df[:, 0].shape, imputed_df[:, 1:].shape)


    # scaled_data = sklearn.preprocessing.scale(imputed_df[:, 1:])
    print('type of imputed_df', type(imputed_df))
    scaler = RobustScaler()
    if filename == 'train':
        scaled_data = scaler.fit_transform(imputed_df[:, 1:])
        scaled_data = np.concatenate((imputed_df[:, 0].reshape(-1, 1), scaled_data), axis=1)
    else:
        scaled_data = scaler.fit_transform(imputed_df)



    np.save('imputed_' + filename + '.npy', scaled_data)

    new_df = pd.DataFrame(scaled_data, columns=df.keys())
    new_df.to_csv('imputed_' + filename + '.csv', index=False)

    print(filename)

    evil_data_pruning('imputed_' + filename + '.csv')


    evil_df = pd.read_csv('evil_imputed_' + filename + '.csv')

    print('below should be head of evil_df')
    evil_np_array = evil_df.to_numpy()
    print('type of evil_df', type(evil_np_array))
    print('first line of evil np array', evil_np_array[0])

    normlizer = Normalizer()
    if filename == 'train':
        norm_data = normlizer.fit_transform(evil_np_array[:, 1:])
        evil_np_array = np.concatenate((evil_np_array[:, 0].reshape(-1, 1), norm_data), axis=1)
    else:
        norm_data = normlizer.fit_transform(evil_np_array)
        evil_np_array = norm_data

    np.save('evil_imputed_' + filename + '.npy', evil_np_array)

    new_df = pd.DataFrame(evil_np_array, columns=df.keys())
    new_df.to_csv('evil_imputed_' + filename + '.csv', index=False)

    with open('keys.txt', 'w') as f:
        f.write(str(list(new_df.keys())))

def fix_missing_data_test(filename='test'):

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
    imputer = IterativeImputer()
    imputer.fit(df)
    imputed_df = imputer.transform(df)


    # scaled_data = sklearn.preprocessing.scale(imputed_df[:, 1:])
    scaler = RobustScaler()
    scaled_data = scaler.fit_transform(imputed_df)

    np.save('imputed_' + filename + '.npy', scaled_data)

    new_df = pd.DataFrame(scaled_data, columns=df.keys())
    new_df.to_csv('imputed_' + filename + '.csv', index=False)

    normlizer = Normalizer()
    norm_data = normlizer.fit_transform(scaled_data)
    evil_np_array = norm_data

    np.save('evil_imputed_' + filename + '.npy', evil_np_array)

    new_df = pd.DataFrame(evil_np_array, columns=df.keys())
    new_df.to_csv('evil_imputed_' + filename + '.csv', index=False)

    with open('keys.txt', 'w') as f:
        f.write(str(list(new_df.keys())))

if __name__ == '__main__':
    fix_missing_data_train('train')
    fix_missing_data_test('test')
    # evil_data_pruning('imputed_train.csv')
