import numpy as np
import pandas as pd
import keras
import matplotlib.pyplot as pyplot
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error as mae

from get_data import fix_missing_data
# from sklearn.linear_model.

def build_model(num_classes, input_shape):

    print('input_shape', input_shape)

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
        keras.layers.Dense(16, activation='tanh', input_shape=input_shape),
        keras.layers.Dense(32, activation='tanh'),
        keras.layers.Dense(64, activation='tanh'),
        keras.layers.Dense(16, activation='tanh'),
        keras.layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        loss=keras.losses.categorical_crossentropy,
        # loss=keras.losses.SparseCategoricalCrossentropy,
        metrics=['mse', 'mae'],
    )

    '''
    model_tanh.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0001), 
        # loss=keras.losses.sparse_categorical_crossentropy, 
        loss=keras.losses.SparseCategoricalCrossentropy,
        metrics=['mse', 'mae'],
    )
    # '''

    # model.summary()

    return model


def train_model(x_train, y_train, x_test, y_test, batch_size, epochs, num_classes, input_shape):

    model = build_model(num_classes, input_shape)

    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)
    # results = model.evaluate(x_train, y_train, batch_size=batch_size)
    results = model.evaluate(x_test, y_test, batch_size=batch_size)

    # print(history.history.keys())
    print(results)

    # print(np.argmax(y_test[:10], axis=1))
    # predict_x = model.predict(x_test[:10])


    print(keras.losses.mae)

    predictions = model.predict(x_train)
    x = []
    for i in range(predictions.shape[0]):
        # normalized = predict_x[i] / np.sum(predict_x[i])
        # x.append(np.random.choice(range(predict_x[i].shape[0]), p=normalized.ravel()))
        x.append(np.argmax(predictions[i]))
    # print(x)


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

def make_prediction():
    pass


if __name__ == '__main__':

    fix_missing_data('train')

    x = pd.read_csv('imputed_train.csv')
    x_train, x_test = train_test_split(x, test_size=0.2, random_state=24)
    # x_train['Stream'] = x_train['Stream'].apply(lambda x: np.log(x))

    y_train = x_train['Danceability'].to_numpy().astype(np.int32)
    y_test = x_test['Danceability'].to_numpy().astype(np.int32)

    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    x_train = x_train.drop(['Danceability'], axis=1).to_numpy()
    x_test = x_test.drop(['Danceability'], axis=1).to_numpy()

    print(*list(zip(x_train[:10], y_train[:10])), sep='\n')

    print(x_train.shape, y_train.shape)

    EPOCHS = 20
    batch_size = 32
    num_classes = 10

    train_model(x_train, y_train, x_test, y_test, batch_size, EPOCHS, num_classes, x_train[0].shape)

