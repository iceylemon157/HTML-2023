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
        plt.savefig(f'plots/hist/{key}_hist.png')

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

if __name__ == '__main__':
    df = pd.read_csv('predictions.csv')
    # print each value's count
    for i in range(10):
        print(df['Danceability'][df['Danceability'] == i].count())
