import numpy as np
import pandas as pd
# import keras
import matplotlib.pyplot as pyplot
from tensorflow import keras
from keras.models import Sequential
from keras.callbacks import EarlyStopping
# from keras.wrappers.scikit_learn import KerasClassifier
from scikeras.wrappers import KerasRegressor
from keras import regularizers

from sklearn.model_selection import train_test_split, cross_val_predict, KFold
from sklearn.metrics import mean_absolute_error as mae

from xgboost import XGBRegressor

from get_data import fix_missing_data
from liblinear.liblinearutil import *
from get_data import fix_missing_data

from sklearn.preprocessing import PolynomialFeatures

def train_model(x, y, test_x, test_y, C=0.1):

    x = PolynomialFeatures(degree=4, include_bias=False).fit_transform(x)
    test_x = PolynomialFeatures(degree=4, include_bias=False).fit_transform(test_x)

    best_model = None
    best_mae = 100000

    for log_lambda in range(-5, 5):
        lamb = 10 ** log_lambda
        C = 1 / (2 * lamb)

        model = train(y, x, f'-s 0 -c {C} -e 0.00001 -q')
        y_train_predict, _, _ = predict(y, x, model, '-q')
        y_test_predict, _, _ = predict(test_y, test_x, model, '-q')
        print(y_train_predict[:100])
        print(mae(y_train_predict, y))
        print(mae(y_test_predict, test_y))

        if mae(y_test_predict, test_y) < best_mae:
            best_mae = mae(y_test_predict, test_y)
            best_model = model

    save_model('best_model', best_model)
    
    return best_model

def make_prediction(model):
    fix_missing_data('test')

    x = pd.read_csv('imputed_test.csv').to_numpy()

    print(x.shape)

    base_id = 17170
    predictions = np.array(predict([], x, model, '-q')[0]).astype(np.int32).reshape(-1, 1)
    predictions = np.concatenate((np.array([np.arange(base_id, base_id + len(predictions))]).T, predictions), axis=1)

    df = pd.DataFrame(predictions, columns=['id', 'Danceability'])
    df.to_csv('predictions.csv', index=False)
    print('successfully saved to predictions.csv')



if __name__ == '__main__':

    fix_missing_data('train')

    x = pd.read_csv('imputed_train.csv')
    x_train, x_test = train_test_split(x, test_size=0.2, random_state=1126)
    # x_train['Stream'] = x_train['Stream'].apply(lambda x: np.log(x))

    # print(x_train[:10])

    y_train = x_train['Danceability'].to_numpy().astype(np.int32)
    y_test = x_test['Danceability'].to_numpy().astype(np.int32)

    # y_train = keras.utils.to_categorical(y_train, 10)
    # y_test = keras.utils.to_categorical(y_test, 10)

    x_train = x_train.drop(['Danceability'], axis=1).to_numpy()
    x_test = x_test.drop(['Danceability'], axis=1).to_numpy()

    model = train_model(x_train, y_train, x_test, y_test, C=0.1)
    # model = load_model('best_model')
    make_prediction(model)