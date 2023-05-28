import numpy as np
import random
from liblinear.liblinearutil import *
from itertools import combinations_with_replacement as cwr

def read_dataset():
    x = []
    y = []
    with open('hw4_train.dat', 'r') as f:
        lines = f.readlines()
        for line in lines:
            data = line.strip('\r\n').split('\t')
            x.append(list(map(float, data[:-1])))
            y.append(int(float(data[-1])))
        x = np.array(x)
        y = np.array(y)

    test_x = []
    test_y = []
    with open('hw4_test.dat', 'r') as f:
        lines = f.readlines()
        for line in lines:
            data = line.strip('\r\n').split('\t')
            test_x.append(list(map(float, data[:-1])))
            test_y.append(int(float(data[-1])))
        test_x = np.array(test_x)
        test_y = np.array(test_y)

    return x, y, test_x, test_y

def transform(x, Q):
    res = []
    for xx in x:
        resx = [1]
        for i in range(1, Q + 1):
            for comb in cwr(xx, i):
                tmp = 1
                for j in comb:
                    tmp *= j
                resx.append(tmp)
        res.append(resx)
    res = np.array(res)
    return res

def zero_one_loss(y_predict, y_true):
    return sum(y_predict != y_true) / len(y_true)

def p12_13_20():
    x, y, test_x, test_y = read_dataset()
    x = transform(x, 4)
    test_x = transform(test_x, 4)
    in_ans = 1e9
    out_ans = 1e9
    in_log_lambda = -1e9
    out_log_lambda = -1e9
    model = None

    for log_lambda in [-6, -3, 0, 3, 6]:
        lamb = 10 ** log_lambda
        C = 1 / (2 * lamb)
        m = train(y, x, f'-s 0 -c {C} -e 0.000001 -q')
        y_train_predict, _, _ = predict(y, x, m, '-q')
        y_test_predict, _, _ = predict(test_y, test_x, m, '-q')
        E_in = zero_one_loss(y_train_predict, y)
        E_out = zero_one_loss(y_test_predict, test_y)

        if E_in == in_ans:
            if log_lambda > in_log_lambda:
                in_log_lambda = log_lambda
        elif E_in < in_ans:
            in_ans = E_in
            in_log_lambda = log_lambda

        if E_out == out_ans:
            if log_lambda > out_log_lambda:
                out_log_lambda = log_lambda
                model = m
        elif E_out < out_ans:
            out_ans = E_out
            out_log_lambda = log_lambda
            model = m

    cnt = sum(abs(val) < 1e-6 for val in model.get_decfun()[0])
    
    print('p12: E_out log_10 lambda*', out_log_lambda)
    print('p13: E_in log_10 lambda*', in_log_lambda)
    print('p20: # of non zero w', cnt)


def p14_15_16():
    x, y, test_x, test_y = read_dataset()
    x = transform(x, 4)
    test_x = transform(test_x, 4)
    dic = {-6: 0, -3: 0, 0: 0, 3: 0, 6: 0}
    tot_E_out = 0
    full_tot_E_out = 0

    for i in range(256):
        val_log_lambda = -1e9
        val_ans = 1e9
        idx = random.sample(range(200), 120)
        x_train = np.array([x[i] for i in idx])
        y_train = np.array([y[i] for i in idx])
        x_val = np.array([x[i] for i in range(200) if i not in idx])
        y_val = np.array([y[i] for i in range(200) if i not in idx])

        for log_lambda in [-6, -3, 0, 3, 6]:
            lamb = 10 ** log_lambda
            C = 1 / (2 * lamb)
            m = train(y_train, x_train, f'-s 0 -c {C} -e 0.000001 -q')
            y_valid_predict, _, _ = predict(y_val, x_val, m, '-q')
            E_val = zero_one_loss(y_valid_predict, y_val)

            if E_val == val_ans:
                if log_lambda > val_log_lambda:
                    val_log_lambda = log_lambda
            elif E_val < val_ans:
                val_ans = E_val
                val_log_lambda = log_lambda

        dic[val_log_lambda] += 1
        C = 1 / (2 * (10 ** val_log_lambda))

        # This part is for problem 15
        m = train(y_train, x_train, f'-s 0 -c {C} -e 0.000001 -q')
        y_test_predict, _, _ = predict(test_y, test_x, m, '-q')
        E_out = zero_one_loss(y_test_predict, test_y)
        tot_E_out += E_out
        # This part ends here

        # This part is for problem 16
        m = train(y, x, f'-s 0 -c {C} -e 0.000001 -q')
        y_test_predict, _, _ = predict(test_y, test_x, m, '-q')
        E_out = zero_one_loss(y_test_predict, test_y)
        full_tot_E_out += E_out
        # This part ends here

    
    print('p14: E_val log_10 lambda*', max(dic, key=dic.get))
    print('p15: E_out', tot_E_out / 256)
    print('p16: E_out', full_tot_E_out / 256)

def p17():
    x, y, test_x, test_y = read_dataset()
    x = transform(x, 4)
    test_x = transform(test_x, 4)
    idx = np.array([i for i in range(200)])
    folds = 5
    tot_E_cv = 0

    for i in range(256):
        random.seed(i)
        random.shuffle(idx)
        x = [x[i] for i in idx]
        y = [y[i] for i in idx]
        ans_cv = 1e9
        ans_log_lambda = -1e9
        for log_lambda in [-6, -3, 0, 3, 6]:
            E_cv = 0
            for v in range(folds):
                x_train = np.array(x[:40 * v] + x[40 * (v + 1):])
                y_train = np.array(y[:40 * v] + y[40 * (v + 1):])
                x_val = np.array(x[40 * v:40 * (v + 1)])
                y_val = np.array(y[40 * v:40 * (v + 1)])
                lamb = 10 ** log_lambda
                C = 1 / (2 * lamb)
                m = train(y_train, x_train, f'-s 0 -c {C} -e 0.000001 -q')
                y_val_predict, _, _ = predict(y_val, x_val, m, '-q')
                E_cv += zero_one_loss(y_val_predict, y_val)
            E_cv /= folds
            ans_cv = min(ans_cv, E_cv)
        tot_E_cv += ans_cv
    print('p17: E_cv Avg lambda*', tot_E_cv / 256)

def p18_19():
    x, y, test_x, test_y = read_dataset()
    x = transform(x, 4)
    test_x = transform(test_x, 4)
    in_ans = 1e9
    out_ans = 1e9
    in_log_lambda = -1e9
    out_log_lambda = -1e9
    model = None

    for log_lambda in [-6, -3, 0, 3, 6]:
        lamb = 10 ** log_lambda
        C = 1 / (2 * lamb)
        m = train(y, x, f'-s 6 -c {C} -e 0.000001 -q')
        y_train_predict, _, _ = predict(y, x, m, '-q')
        y_test_predict, _, _ = predict(test_y, test_x, m, '-q')
        E_out = zero_one_loss(y_test_predict, test_y)

        if E_out == out_ans:
            if log_lambda > out_log_lambda:
                out_log_lambda = log_lambda
                model = m
        elif E_out < out_ans:
            out_ans = E_out
            out_log_lambda = log_lambda
            model = m
    
    cnt = sum(val < 1e-6 for val in model.get_decfun()[0])
    
    print('p18: E_out log_10 lambda*', out_log_lambda)
    print('p19: # of non-zero w', cnt)



if __name__ == '__main__':
    p12_13_20()
    p14_15_16()
    p17()
    p18_19()
