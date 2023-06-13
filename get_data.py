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
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
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
    # new_df = df.drop(all_keys[-12:], axis=1)

    # to_drop = ['Key', 'Album_type', 'Licensed', 'official_video', 'Likes', 'Views', 'id', 'Stream', 'Duration_ms']
    to_drop = ['id', 'Track', 'Album', 'Uri', 'Url_spotify', 'Url_youtube', 'Description', 'Title', 'Channel', 'Composer', 'Artist']
    to_drop.extend(['Comments', 'official_video', 'Licensed', 'Key', 'Album_type', 'Views', 'Likes', 'Stream', 'Instrumentalness'])

    # to_drop.extend(['Loudness', 'Duration_ms', 'Liveness'])
    # to_drop = ['Danceability', 'Loudness', 'Liveness', 'Tempo', 'Instrumentalness', 'Energy']

    new_df = df.drop(to_drop, axis=1)

    if 'Album_type' in new_df.keys():
        new_df.loc[new_df['Album_type'] == 'single', 'Album_type'] = np.float32(0)
        new_df.loc[new_df['Album_type'] == 'compilation', 'Album_type'] = np.float32(0.5)
        new_df.loc[new_df['Album_type'] == 'album', 'Album_type'] = np.float32(1)

    # new_df.loc[new_df['Licensed'] == 'True', 'Licensed'] = np.float32(1)
    # new_df.loc[new_df['Licensed'] == 'False', 'Licensed'] = np.float32(0)

    # new_df.loc[new_df['official_video'] == 'True', 'official_video'] = np.float32(1)
    # new_df.loc[new_df['official_video'] == 'False', 'official_video'] = np.float32(0)

    # print(new_df.keys())

    # new_df = df[['Danceability', 'Energy', 'Loudness', 'Speechiness', 'Liveness', 'Tempo']]

    new_df.to_csv('new_' + filename + '.csv', index=False)

def composer_encoder(encoding_type='freq'):
    cdf = pd.read_csv('train.csv')

    composer_name = list(cdf['Composer'].unique())
    composer_name = [x for x in composer_name if type(x) == str]
    composer_name = sorted(composer_name)

    if encoding_type == 'freq':
        dic = cdf['Composer'].value_counts().to_dict()
        for key in dic.keys():
            dic[key] = dic[key] / len(cdf['Composer'])
    
    if encoding_type == 'target':
        dic = {}
        for key in composer_name:
            dic[key] = cdf[cdf['Composer'] == key]['Danceability'].mean()
 

    return dic

def get_composer(filename='train', encoding_type='target'):

    cdf = pd.read_csv(filename + '.csv')
    df = pd.read_csv('new_' + filename + '.csv')

    composer_name = list(cdf['Composer'].unique())
    composer_name = [x for x in composer_name if type(x) == str]
    composer_name = sorted(composer_name)

    if encoding_type == 'target':
        dic = composer_encoder(encoding_type)
        a = []
        for i in range(len(cdf['Composer'])):
            a.append(dic[cdf['Composer'][i]] if type(cdf['Composer'][i]) == str else np.float32(0))
        df['Composer'] = a

    if encoding_type == 'freq':
        dic = composer_encoder(encoding_type)
        a = []
        for i in range(len(cdf['Composer'])):
            a.append(dic[cdf['Composer'][i]] if type(cdf['Composer'][i]) == str else np.float32(0))
        df['Composer'] = a

    if encoding_type == 'onehot':
        # importance = ['Energy', 'Loudness', 'Speechiness', 'Acousticness', 'Liveness', 'Valence', 'Tempo', 'Duration_ms', "Finneas O'Connell", 'Juicy J', 'Louis Bell', 'Ludwig Göransson', 'Mike Dean', 'Noah "40" Shebib', 'Ricky Reed', 'Terrace Martin', 'Yeti Beats', 'Big Thief', 'Billie Eilish', 'Chance the Rapper', 'Drake', 'Dua Lipa', 'Halsey', 'Travis Scott']
        for key in composer_name:
            if type(key) != str: # or key not in importance:
                continue
            a = []
            for i in range(len(cdf['Composer'])):
                if cdf['Composer'][i] == key:
                    a.append(np.float32(1))
                else:
                    a.append(np.float32(0))
            df[key] = a

    df.to_csv('new_' + filename + '.csv', index=False)

def artist_encoder(encoding_type='target'):
    cdf = pd.read_csv('train.csv')
    artist_name = list(cdf['Artist'].unique())
    artist_name = [x for x in artist_name if type(x) == str]
    artist_name = sorted(artist_name)
 

    if encoding_type == 'target':
        dic = {}
        for key in artist_name:
            dic[key] = cdf[cdf['Artist'] == key]['Danceability'].mean()

    return dic


def get_artist(filename='train', encoding_type='target'):
    cdf = pd.read_csv(filename + '.csv')
    df = pd.read_csv('new_' + filename + '.csv')
    artist_name = list(cdf['Artist'].unique())
    artist_name = [x for x in artist_name if type(x) == str]
    artist_name = sorted(artist_name)

    if encoding_type == 'target':
        dic = artist_encoder(encoding_type)
        a = []
        for i in range(len(cdf['Artist'])):
            a.append(dic[cdf['Artist'][i]] if type(cdf['Artist'][i]) == str else np.float32(0))
        df['Artist'] = a

    if encoding_type == 'onehot':
        importance = ['Energy', 'Loudness', 'Speechiness', 'Acousticness', 'Liveness', 'Valence', 'Tempo', 'Duration_ms', "Finneas O'Connell", 'Juicy J', 'Ludwig Göransson', 'Mike Dean', 'Noah "40" Shebib', 'Ricky Reed', 'Terrace Martin', 'Yeti Beats', 'A$AP Rocky', 'Alicia Keys', 'Anderson .Paak', 'Angel Olsen', 'Arcade Fire', 'Avicii', 'Big Thief', 'Billie Eilish', 'Bon Iver', 'Chance the Rapper', 'Daft Punk', 'Drake', 'Dua Lipa', 'Foals', 'Frank Ocean', 'Halsey', 'Harry Styles', 'Justice', 'Kanye West', 'Kaytranada', 'Kurt Vile', 'LCD Soundsystem', 'Lorde', 'Mac DeMarco', 'Nicki Minaj', 'Pharrell Williams', 'Radiohead', 'Shawn Mendes', 'Skrillex', 'St. Vincent', 'Swedish House Mafia', 'Tame Impala', 'The National', 'Travis Scott', 'Ty Segall', 'Vampire Weekend']
        for key in artist_name:
            if type(key) != str or key not in importance:
                continue
            a = []
            for i in range(len(cdf['Artist'])):
                if cdf['Artist'][i] == key:
                    a.append(np.float32(1))
                else:
                    a.append(np.float32(0))
            df[key] = a

    df.to_csv('new_' + filename + '.csv', index=False)

def album_type_encoder(encoding_type='target'):
    cdf = pd.read_csv('train.csv')
    album_type = ['single', 'compilation', 'album']

    if encoding_type == 'target':
        dic = {}
        for key in album_type:
            dic[key] = cdf[cdf['Album_type'] == key]['Danceability'].mean()

    return dic


def get_album_type(filename='train', encoding_type='target'):
    cdf = pd.read_csv(filename + '.csv')
    df = pd.read_csv('new_' + filename + '.csv')
    album_type = ['single', 'compilation', 'album']

    if encoding_type == 'target':
        dic = album_type_encoder(encoding_type)
        a = []
        for i in range(len(cdf['Album_type'])):
            a.append(dic[cdf['Album_type'][i]] if type(cdf['Album_type'][i]) == str else np.float32(0))
        df['Album_type'] = a

    if encoding_type == 'onehot':
        for key in album_type:
            if type(key) != str:
                continue
            a = []
            for i in range(len(cdf['Album_type'])):
                if cdf['Album_type'][i] == key:
                    a.append(np.float32(1))
                else:
                    a.append(np.float32(0))
            df[key] = a

    df.to_csv('new_' + filename + '.csv', index=False)

def data_preprocessing(scaler_type='std', encoding_type='freq'):

    try:
        with open('keys.txt', 'r') as f:
            print('keys in latest file:', f.read())
    except:
        print('keys.txt does not exist')
    print('new keys to replace:', list(pd.read_csv('new_' + 'train' + '.csv').keys()))
    opt = input('do you want to overwrite? (y/n): ')
    if opt != 'y':
        return

    new_data('train')
    new_data('test')
    get_composer('train', encoding_type=encoding_type)
    get_composer('test', encoding_type=encoding_type)
    get_artist('train', encoding_type=encoding_type)
    get_artist('test', encoding_type=encoding_type)
    # get_album_type('train', encoding_type=encoding_type)
    # get_album_type('test', encoding_type=encoding_type)

    importance = ['Energy', 'Loudness', 'Speechiness', 'Acousticness', 'Liveness', 'Valence', 'Tempo', 'Duration_ms', "Finneas O'Connell", 'J. Cole', 'Juicy J', 'Ludwig Göransson', 'Mike Dean', 'Ricky Reed', 'Terrace Martin', 'Yeti Beats', 'A$AP Rocky', 'Big Thief', 'Billie Eilish', 'Chance the Rapper', 'Daft Punk', 'Drake', 'Dua Lipa', 'Halsey', 'Radiohead', 'Shawn Mendes', 'Travis Scott', 'Vampire Weekend']
    new_train_df = pd.read_csv('new_' + 'train' + '.csv')
    new_test_df = pd.read_csv('new_' + 'test' + '.csv')
    keys = list(new_train_df.keys())
    for key in keys:
        if key == 'Danceability':
            continue
        if key not in importance:
            new_train_df = new_train_df.drop(columns=[key])
            new_test_df = new_test_df.drop(columns=[key])
    
    new_train_df.to_csv('new_' + 'train' + '.csv', index=False)
    new_test_df.to_csv('new_' + 'test' + '.csv', index=False)


    new_train_df = pd.read_csv('new_' + 'train' + '.csv')
    keys = list(new_train_df.keys())

    print(keys)

    return

    # train data processing

    # imputed_data is a np array
    y_train = new_train_df['Danceability'].to_numpy().reshape(-1, 1)
    new_train_df = new_train_df.drop(columns=['Danceability'])
    imputer = IterativeImputer()
    imputer.fit(new_train_df)
    imputed_train_data = imputer.transform(new_train_df)

    # scaled_data is a np array
    if scaler_type == 'std':
        scaler = StandardScaler()
    elif scaler_type == 'minmax':
        scaler = MinMaxScaler()
    scaler.fit(imputed_train_data)
    scaled_train_data = scaler.transform(imputed_train_data)

    scaled_train_data = np.concatenate((y_train, scaled_train_data), axis=1)

    imputed_train_df = pd.DataFrame(scaled_train_data, columns=keys)
    imputed_train_df.to_csv(f'{scaler_type}_{encoding_type}_imputed_train.csv', index=False)

    # test data processing

    new_test_df = pd.read_csv('new_' + 'test' + '.csv')

    print(new_test_df.keys())

    imputed_test_data = imputer.transform(new_test_df)
    scaled_test_data = scaler.transform(imputed_test_data)

    new_test_df = pd.DataFrame(scaled_test_data, columns=new_test_df.keys())
    new_test_df.to_csv(f'{scaler_type}_{encoding_type}_imputed_test.csv', index=False)

    ''' commented until fix
    evil_data_pruning('imputed_' + filename + '.csv')

    evil_df = pd.read_csv('evil_imputed_' + filename + '.csv')
    evil_np_array = evil_df.to_numpy()
    '''

    with open('keys.txt', 'w') as f:
        f.write(str(list(new_train_df.keys())))



if __name__ == '__main__':
    # fix_missing_data_train('train')
    # fix_missing_data_test('test')
    # evil_data_pruning('imputed_train.csv')
    data_preprocessing('std', 'target')
    data_preprocessing('std', 'onehot')
    data_preprocessing('minmax', 'target')
    data_preprocessing('minmax', 'onehot')
    # evil_data_pruning('imputed_train.csv')
    # evil_data_pruning('imputed_test.csv')

    # get_composer()

    # df = pd.read_csv('train.csv')
    # print(len(df['Artist'].unique()))


