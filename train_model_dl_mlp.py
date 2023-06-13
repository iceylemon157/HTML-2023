import tqdm
import numpy as np
import pandas as pd
# import keras
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.models import Sequential
from keras.callbacks import EarlyStopping
# from keras.wrappers.scikit_learn import KerasClassifier
from scikeras.wrappers import KerasRegressor
from keras import regularizers

from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_predict, KFold
from sklearn.metrics import SCORERS, mean_absolute_error as mae
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from xgboost import XGBRegressor, XGBClassifier

from get_data import *
from visual import *

def train_model(x_train, y_train, x_test, y_test, scaler_type='std'):

    # param_grid = {
    #     'max_depth': [5, 6, 7, 8, 9, 10],
    #     'n_estimators': [100, 250, 500, 750, 1000],
    #     'max_features': [0.5, 0.6, 0.7, 0.8, 0.9],
    #     'min_samples_split': [2, 4, 6, 8, 10],
    #     'min_samples_leaf': [1, 2, 3, 4, 5],
    #     'bootstrap': [True, False],
    #     'oob_score': [True, False],
    # }

    # gsc = GridSearchCV(
    #     estimator=RandomForestClassifier(random_state=24),
    #     param_grid=param_grid,
    #     cv=5, scoring='neg_mean_absolute_error', verbose=1, n_jobs=20)


    # gsc_result = gsc.fit(x_train, y_train)

    # best of XGBRegressor
    # model = XGBRegressor(random_state=24, n_estimators=1000, max_depth=1, n_jobs=20, verbose=1)


    # model = MLPRegressor(random_state=3224, max_iter=324, early_stopping=True, learning_rate_init=0.000157, hidden_layer_sizes=(64, 128, 256, 256, 256, 128, 64), verbose=True, activation='relu', validation_fraction=0.157, n_iter_no_change=10)
    # model = MLPClassifier(random_state=3224, max_iter=1000, early_stopping=True, learning_rate_init=0.0000157, validation_fraction=0.157, n_iter_no_change=10, hidden_layer_sizes=(256, 256, 256, 256, 256, 256, 256, 256, 256), verbose=True, activation='relu')
    # model = MLPClassifier(random_state=24, max_iter=1000, early_stopping=True, learning_rate_init=0.000157, validation_fraction=0.2, n_iter_no_change=100, hidde
    # n_layer_sizes=(100, 100, 100, 100, 100, 100, 100, 100, 100, 100))
    # model = RandomForestClassifier(random_state=24, n_estimators=3157, max_depth=7, n_jobs=20, verbose=1, min_samples_split=20, min_samples_leaf=2, max_features='log2')
    # model = RandomForestRegressor(random_state=24, n_estimators=8157, max_depth=7, n_jobs=20, verbose=1, min_samples_split=2, min_samples_leaf=1, max_features=0.6, oob_score=True)
    # model.fit(x_train, y_train)

    # df = pd.read_csv('new_test.csv')
    # print(list(model.feature_importances_)

    '''

    train_predictions = model.predict(x_train)

    predictions = model.predict(x_test)
    predictions = np.round(predictions).astype(np.int32)
    for i in range(len(predictions)):
        if predictions[i] < 0:
            predictions[i] = 0
        elif predictions[i] > 9:
            predictions[i] = 9

    print('train', mae(y_train, train_predictions))
    print('test', mae(y_test, predictions))

    if scaler_type == 'std':
        train_predictions = StandardScaler().fit_transform(train_predictions.reshape(-1, 1))
        predictions = StandardScaler().fit_transform(predictions.reshape(-1, 1))
    elif scaler_type == 'minmax':
        train_predictions = MinMaxScaler().fit_transform(train_predictions.reshape(-1, 1))
        predictions = MinMaxScaler().fit_transform(predictions.reshape(-1, 1))


    # x_train = np.concatenate((x_train, train_predictions), axis=1)
    # x_test = np.concatenate((x_test, predictions), axis=1)
    '''

    model = None
    premodel = model
    model = XGBRegressor(random_state=24, n_estimators=10000, max_depth=8, n_jobs=20, learning_rate=0.005)
    # model = XGBClassifier()
    # print('parameters', model.get_params())
    
    model.fit(x_train, y_train, eval_set=[(x_test, y_test)], early_stopping_rounds=10, verbose=True, eval_metric='mae')

    temp = pd.read_csv('new_test.csv')

    print('feature importances')
    print(list(temp.keys()))
    print(list(model.feature_importances_))
    for a, b in zip(list(temp.keys()), list(model.feature_importances_)):
        print(f'feature: {a}\t\t importance: {b}')


    train_predictions = model.predict(x_train)

    predictions = model.predict(x_test)
    predictions = np.round(predictions).astype(np.int32)
    for i in range(len(predictions)):
        if predictions[i] < 0:
            predictions[i] = 0
        elif predictions[i] > 9:
            predictions[i] = 9


    print('train', mae(y_train, train_predictions))
    print('test', mae(y_test, predictions))


    return premodel, model

def make_prediction(premodel, model, scaler_type):

    x = pd.read_csv(f'new_test.csv').to_numpy()

    if premodel:
        pre_predictions = premodel.predict(x)
        if scaler_type == 'std':
            pre_predictions = StandardScaler().fit_transform(pre_predictions.reshape(-1, 1))
        elif scaler_type == 'minmax':
            pre_predictions = MinMaxScaler().fit_transform(pre_predictions.reshape(-1, 1))
    # x = np.concatenate((x, pre_predictions), axis=1)

    base_id = 17170
    predictions = model.predict(x)
    # predictions = predictions - 0.5
    predictions = np.round(predictions).astype(np.int32).reshape(-1, 1)

    for i in range(len(predictions)):
        if predictions[i] < 0:
            predictions[i] = 0
        elif predictions[i] > 9:
            predictions[i] = 9

    predictions = np.concatenate((np.array([np.arange(base_id, base_id + len(predictions))]).T, predictions), axis=1)

    df = pd.DataFrame(predictions, columns=['id', 'Danceability'])
    df.to_csv('predictions.csv', index=False)
    print(df.head(10))
    print('successfully saved to predictions.csv')

if __name__ == '__main__':

    scaler_type = 'std'
    x = pd.read_csv(f'new_train.csv')
    x_train, x_test = train_test_split(x, test_size=0.2, random_state=1126)

    y_train = x_train['Danceability'].to_numpy().astype(np.int32)
    y_test = x_test['Danceability'].to_numpy().astype(np.int32)

    x_train = x_train.drop(['Danceability'], axis=1).to_numpy()
    x_test = x_test.drop(['Danceability'], axis=1).to_numpy()

    premodel, model = train_model(x_train, y_train, x_test, y_test, scaler_type)
    make_prediction(None, model, scaler_type)
    print_prediction_stats()