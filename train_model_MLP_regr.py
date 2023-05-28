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

from get_data import *
# from sklearn.linear_model.

def build_model(num_classes, input_shape, C=0.1):

    # print('input_shape', input_shape)
    C = 10

    regularizer = regularizers.l2(C)

    model = Sequential([
        keras.layers.Dense(32, activation='relu', input_shape=input_shape),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(num_classes, activation='softmax')
    ])

    model_tanh = Sequential([
        keras.layers.Dense(32, activation='relu', input_shape=input_shape, kernel_regularizer=regularizer),
        keras.layers.BatchNormalization(),
        keras.layers.Dense(64, activation='relu', kernel_regularizer=regularizer),
        keras.layers.BatchNormalization(),
        keras.layers.Dense(32, activation='relu', kernel_regularizer=regularizer),
        keras.layers.BatchNormalization(),
        keras.layers.Dense(1, activation='linear')
    ])

    model_tanh.compile(
        optimizer=keras.optimizers.legacy.Adam(learning_rate=0.00024), 
        # optimizer='adam',
        # loss=keras.losses.sparse_categorical_crossentropy, 
        loss=keras.losses.mae,
        metrics=['mae'],
    )

    model.compile(
        optimizer=keras.optimizers.legacy.Adam(learning_rate=0.0001),
        loss=keras.losses.categorical_crossentropy,
        # loss=keras.losses.SparseCategoricalCrossentropy,
        metrics=['mse', 'mae'],
    )

    return model_tanh

    # '''
    # '''

    # model.summary()

    return model_tanh


def train_model(x_train, y_train, x_test, y_test, batch_size, epochs, num_classes, input_shape):

    model = build_model(num_classes, input_shape)

    '''
    regressor = KerasRegressor(
        build_fn=build_model, 
        num_classes=num_classes,
        input_shape=input_shape,
        batch_size=batch_size, 
        epochs=epochs, 
    )
    '''

    # kfold = KFold(n_splits=10, shuffle=True)
    # results = cross_val_predict(regressor, x_train, y_train, cv=kfold)
    # print('results', results)

    # exit()

    early_stopping = EarlyStopping(patience=2, restore_best_weights=True)

    # print(x_train)
    # regressor.fit(x_train, y_train)
    # regressor.fit(x_train, y_train, callbacks=[early_stopping], validation_split=0.1)
    # model = regressor.best_estimator_.model

    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1, callbacks=[early_stopping])
    # results = model.evaluate(x_train, y_train, batch_size=batch_size)
    results = model.evaluate(x_test, y_test, batch_size=batch_size)

    # print(history.history.keys())
    # print(results)

    # print(np.argmax(y_test[:10], axis=1))
    # predict_x = model.predict(x_test[:10])



    train_predictions = model.predict(x_train)

    predictions = model.predict(x_test)
    predictions = np.round(predictions).astype(np.int32)
    print('QAQ integers', mae(y_test, predictions))
    print(predictions[:10])

    # regressor = XGBRegressor(objective='reg:absoluteerror', n_estimators=100, learning_rate=0.1, max_depth=3)
    # regressor.fit(x_train, train_predictions, early_stopping_rounds=2, eval_set=[(x_test, y_test)], verbose=False)

    # predictions = regressor.predict(x_test)
    # predictions = np.round(predictions).astype(np.int32)
    # print('QAQ XGB', mae(y_test, predictions))

    return model 

    exit()

    # print(mae(y_test, predictions))
    # print(x)

    return model

    exit()


    print(np.argmax(y_test, axis=1).shape, np.array(x).shape)
    print(mae(np.argmax(y_train, axis=1), x))

    print()

    exit()

    # print(model.predict(x_train[:10]))

    pyplot.title('Loss')
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='test')
    pyplot.legend()
    # plot mse during training
    pyplot.subplot(212)
    pyplot.title('Mean Absolute Error')
    pyplot.plot(history.history['mae'], label='train')
    pyplot.plot(history.history['val_mae'], label='test')
    pyplot.legend()
    pyplot.show()

def cross_validation(x, y, C, batch_size, epochs, num_classes, input_shape):

    fold_number = 10
    kfold = KFold(n_splits=fold_number, shuffle=True)

    score_per_fold = []
    mae_per_fold = []

    for train, test in kfold.split(x, y):

        model = build_model(num_classes, input_shape, C)
        early_stopping = EarlyStopping(patience=2, restore_best_weights=True)

        history = model.fit(x[train], y[train], batch_size=batch_size, epochs=epochs, validation_split=0.1, callbacks=[early_stopping], verbose=0)
        results = model.evaluate(x[test], y[test], batch_size=batch_size, verbose=0)

        score_per_fold.append(results[0])
        mae_per_fold.append(results[1])

    print('-' * 100)
    print('C: ', C)
    print('Mean score', np.mean(score_per_fold))
    print('Mean mae', np.mean(mae_per_fold))
    print('-' * 100)




def make_prediction(model):
    fix_missing_data_test('test')

    x = pd.read_csv('evil_imputed_test.csv').to_numpy()

    base_id = 17170
    predictions = model.predict(x)
    predictions = np.round(predictions).astype(np.int32).reshape(-1, 1)
    predictions = np.concatenate((np.array([np.arange(base_id, base_id + len(predictions))]).T, predictions), axis=1)

    df = pd.DataFrame(predictions, columns=['id', 'Danceability'])
    df.to_csv('predictions.csv', index=False)
    print(df.head(10))
    print('successfully saved to predictions.csv')



if __name__ == '__main__':

    fix_missing_data_train('train')

    x = pd.read_csv('evil_imputed_train.csv')
    x_train, x_test = train_test_split(x, test_size=0.2, random_state=1126)
    # x_train['Stream'] = x_train['Stream'].apply(lambda x: np.log(x))

    # print(x_train[:10])

    y_train = x_train['Danceability'].to_numpy().astype(np.int32)
    y_test = x_test['Danceability'].to_numpy().astype(np.int32)

    # y_train = keras.utils.to_categorical(y_train, 10)
    # y_test = keras.utils.to_categorical(y_test, 10)

    x_train = x_train.drop(['Danceability'], axis=1).to_numpy()
    x_test = x_test.drop(['Danceability'], axis=1).to_numpy()

    # print(*list(zip(x_train[:10], y_train[:10])), sep='\n')

    # print(x_train.shape, y_train.shape)

    EPOCHS = 100
    batch_size = 32
    num_classes = 10

    # for C in [0.1, 1, 10, 100, 1000]:
    #     cross_validation(x_train, y_train, C, batch_size, EPOCHS, num_classes, x_train[0].shape)


    model = train_model(x_train, y_train, x_test, y_test, batch_size, EPOCHS, num_classes, x_train[0].shape)
    make_prediction(model)
