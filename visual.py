import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from get_data import *


def plot_histogram(filename='imputed_train.csv'):
    df = pd.read_csv(filename)

    for key in df.keys():
        plt.figure(figsize=(8, 8))
        plt.hist(df[key], bins=50)
        plt.xlabel(key)
        plt.ylabel('Count')
        plt.title(f'{key} Distribution')
        plt.savefig(f'plots/hist/{filename[:-4]}_{key}_hist.png')

def plot_data_one_to_one():
    df = pd.read_csv('imputed_train.csv')

    for key in df.keys():
        for key2 in df.keys():

            if key == key2:
                continue

            plt.figure(figsize=(8, 8))
            plt.scatter(df[key], df[key2], cmap='viridis')
            plt.xlabel(key)
            plt.ylabel(key2)
            plt.title(f'{key} vs {key2}')
            plt.savefig(f'plots/{key}_vs_{key2}.png')

def print_data_stats(filename):
    df = pd.read_csv(filename)

    print('-------------------')
    print(filename)
    for key in df.keys():
        print(key, np.min(df[key]), np.max(df[key]))
        print('mean', np.mean(df[key]))
        print('var', np.var(df[key]))
    print('-------------------')

def print_prediction_stats():
    df = pd.read_csv('predictions.csv')

    # print each value's count
    for i in range(10):
        print(df['Danceability'][df['Danceability'] == i].count())

if __name__ == '__main__':
    # print_data_stats('imputed_train.csv')
    # print_data_stats('imputed_test.csv')
    # print_prediction_stats()

    df = pd.read_csv('train.csv')


    for count in df['Artist'].unique():
        print(count, df['Artist'][df['Artist'] == count].count())

    print(df['Artist'].value_counts())
    df = pd.read_csv('test.csv')
    print(df['Artist'].value_counts())

    exit()

    a = list(df['Album_type'].to_numpy())

    b = {}
    for i in a:
        b[i] = a.count(i)

    print(b)
    print(len(b))


    for key in b.keys():


        plt.figure(figsize=(8, 8))
        plt.hist(df.loc[df['Album_type'] == key, 'Danceability'], bins=50)
        plt.xlabel(key)
        plt.ylabel('Danceability')
        plt.title(f'{key} Danceability Distribution')
        plt.savefig(f'plots/hist/composer_{key}_hist.png')


    # df = pd.read_csv('imputed_train.csv')
    # print(df['Danceability'])
    # print(df['Danceability'].to_numpy())

    # print(df.to_numpy().shape)


    # y = df['Danceability'].to_numpy().astype(np.int32)
    # df.drop(['Danceability'])
