import numpy as np
import pandas as pd
# import keras
import matplotlib.pyplot as pyplot
from tensorflow import keras
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from scikeras.wrappers import KerasRegressor
from keras import regularizers

from sklearn.model_selection import train_test_split, cross_val_predict, KFold
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import accuracy_score

from xgboost import XGBRegressor, XGBClassifier

from get_data import *
from visual import *
# from sklearn.linear_model.

from hyperopt import STATUS_OK, Trials, fmin, hp, tpe

space = {
    'max_depth': hp.quniform("max_depth", 3, 10, 1),
    'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.2)),
    'gamma': hp.uniform ('gamma', 1, 100),
    'reg_alpha' : hp.uniform('reg_alpha', 0, 5), # L1 regularization term on weights. Increasing this value will make model more conservative.
    'reg_lambda' : hp.uniform('reg_lambda', 0, 24),
    'min_child_weight' : hp.quniform('min_child_weight', 0, 10, 1),
    'subsample' : hp.uniform('subsample', 0.5, 1),
    'n_estimators': 1024,
    'seed': 24
}

def objective(space, x_train, y_train, x_test, y_test):
    clf = XGBRegressor(
        max_depth = int(space['max_depth']), 
        learning_rate = space['learning_rate'], 
        gamma = space['gamma'], 
        reg_lambda = space['reg_lambda'],
        min_child_weight=int(space['min_child_weight']), 
        subsample = space['subsample'],
        n_estimators = 1024,
        seed = 24,
    )
    
    evaluation = [(x_train, y_train), (x_test, y_test)]
    
    clf.set_params(early_stopping_rounds=10)
    clf.fit(x_train, y_train, eval_set=evaluation, verbose=False)

    pred = clf.predict(x_test)
    accuracy = mae(y_test, pred)
    print ("SCORE:", accuracy)
    return {'loss': accuracy, 'status': STATUS_OK }

def optimize(x_train, y_train, x_test, y_test):
    trials = Trials()
    best = fmin(fn=lambda x: objective(x, x_train, y_train, x_test, y_test), space=space, algo=tpe.suggest, max_evals=100, trials=trials)
    print(best)
    best['max_depth'] = int(best['max_depth'])
    best['min_child_weight'] = int(best['min_child_weight'])

    return best

    # {'gamma': 3.0528760271700226, 'learning_rate': 0.07070344017996913, 'max_depth': 3.0, 'min_child_weight': 4.0, 'reg_lambda': 0.24530937244505072, 'subsample': 0.59134764560718}
    # new
    # {'gamma': 1.1138574104861145, 'learning_rate': 0.045342649547356235, 'max_depth': 5.0, 'min_child_weight': 7.0, 'reg_alpha': 99.82174257328865, 'reg_lambda': 97.13413024528857, 'subsample': 0.5770902244439459}

def train_model(x_train, y_train, x_test, y_test, best):

    # model = XGBClassifier(
    #     n_estimators=1000, 
    #     max_depth=5, 
    #     learning_rate=0.05, 
    #     n_jobs=20, 
    #     random_state=24
    # )
    # model.set_params(**{'gamma': 3.0528760271700226, 'learning_rate': 0.07070344017996913, 'max_depth': 3, 'min_child_weight': 4, 'reg_alpha': 12, 'reg_lambda': 0.24530937244505072, 'subsample': 0.59134764560718})
    # model.set_params(**{'gamma': 1.1138574104861145, 'learning_rate': 0.045342649547356235, 'max_depth': 10, 'min_child_weight': 10, 'reg_alpha': 5, 'reg_lambda': 100, 'subsample': 0.5770902244439459})
    model = XGBRegressor(n_estimators=1024, seed=24)
    model.set_params(**best)
    # model.set_params(**{'gamma': 10.755973858543882, 'learning_rate': 0.02044733867378702, 'max_depth': 10, 'min_child_weight': 4, 'reg_alpha': 1.5343631118150503, 'reg_lambda': 20.588154090655394, 'subsample': 0.7273078350958033})
    # model.set_params(**{'gamma': 6.117732364829907, 'learning_rate': 0.01789984525725737, 'max_depth': 8, 'min_child_weight': 9, 'reg_alpha': 1.8294593789764522, 'reg_lambda': 19.366843721924326, 'subsample': 0.901881061476004})
    # model.set_params(**{'gamma': 18.178963328098988, 'learning_rate': 0.15745997640920106, 'max_depth': 9, 'min_child_weight': 10, 'reg_alpha': 3.960353276398714, 'reg_lambda': 19.74099787992459, 'subsample': 1})
    # model.set_params(**{'gamma': 8.740580409491402, 'learning_rate': 0.012669524576354197, 'max_depth': 7, 'min_child_weight': 5, 'reg_alpha': 1.8027576981991906, 'reg_lambda': 1.6532145371944296, 'subsample': 0.7926234052142963})
    print(x_train.shape, y_train.shape)
    model.fit(
        x_train, 
        y_train, 
        eval_set=[(x_train, y_train), (x_test, y_test)], 
        early_stopping_rounds=10, 
        # verbose=2
    )

    temp = pd.read_csv('new_test.csv')
    print('feature importances')
    print(list(temp.keys()))
    print(list(model.feature_importances_))
    p = []
    for a, b in zip(list(temp.keys()), list(model.feature_importances_)):
        print(f'feature: {a}\t\t importance: {b}')
        if b > 0.01:
            p.append(a)

    print(p)

    train_predictions = model.predict(x_train)

    predictions = model.predict(x_test)
    predictions = np.round(predictions).astype(np.int32)
    print(predictions[:10], y_test[:10])

    print('train', mae(y_train, train_predictions))
    print('test', mae(y_test, predictions))

    return model

def make_prediction(model):

    x = pd.read_csv('new_test.csv').to_numpy()

    base_id = 17170
    predictions = model.predict(x)
    predictions = np.round(predictions).astype(np.int32).reshape(-1, 1)

    predictions = np.concatenate((np.array([np.arange(base_id, base_id + len(predictions))]).T, predictions), axis=1)


    df = pd.DataFrame(predictions, columns=['id', 'Danceability'])
    df.to_csv('predictions.csv', index=False)
    print(df.head(10))
    print('successfully saved to predictions.csv')

    evaluate_prediction()

def evaluate_prediction():

    predictions = pd.read_csv('predictions.csv')
    test_partial_answer = pd.read_csv('test_partial_answer.csv')
    answer = dict(zip(test_partial_answer['id'], test_partial_answer['Danceability']))

    a = []

    for idx, danceability in answer.items():
        # print(int(predictions.loc[predictions['id'] == idx, 'id']), idx)
        # print(int(predictions.loc[predictions['id'] == idx, 'Danceability']))

        # print('predict, answer', int(predictions.loc[predictions['id'] == idx, 'Danceability']), danceability)
        a.append(abs(int(predictions.loc[predictions['id'] == idx, 'Danceability']) - danceability))

    print(len(a))

    print('mae', np.mean(a))


if __name__ == '__main__':

    x = pd.read_csv('new_train.csv')
    x_train, x_test = train_test_split(x, test_size=0.2, random_state=24)

    y_train = x_train['Danceability'].to_numpy().astype(np.float32)
    y_test = x_test['Danceability'].to_numpy().astype(np.float32)

    x_train = x_train.drop(['Danceability'], axis=1).to_numpy()
    x_test = x_test.drop(['Danceability'], axis=1).to_numpy()

    print(x_train.shape, y_train.shape)
    print(x_test.shape, y_test.shape)

    # best = optimize(x_train, y_train, x_test, y_test)
    best = {'gamma': 1.4981199094390032, 'learning_rate': 0.02826125457226122, 'max_depth': 8, 'min_child_weight': 4, 'reg_alpha': 0.9703688691732006, 'reg_lambda': 13.207608840048582, 'subsample': 0.9413801560715024}
    # print(y_train[:10])

    model = train_model(x_train, y_train, x_test, y_test, best)
    make_prediction(model)
    print_prediction_stats()
