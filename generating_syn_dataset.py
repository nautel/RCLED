import numpy as np
import pandas as pd
from tqdm import tqdm



def generate_random_time_series(ts_length, noise_level):
    time = np.arange(0, ts_length, 1)
    binomial_parameter = np.random.binomial(size=3, n=1, p=0.5)
    epsilon = np.random.normal(0, 1, len(time))
    t0 = binomial_parameter[0] * 50 + (1 - binomial_parameter[0]) * 100
    w = binomial_parameter[1] * 40 + (1 - binomial_parameter[1]) * 50
    S1 = np.sin((time - t0) / w) + noise_level * epsilon / 100
    S2 = np.cos((time - t0) / w) + noise_level * epsilon / 100
    return binomial_parameter[2] * S1 + (1 - binomial_parameter[2]) * S2


def generate_time_series_dataset(num_var, ts_length, noise_level):
    # return matrix N x L
    dataset = []
    for i in tqdm(range(num_var)):
        series = generate_random_time_series(ts_length, noise_level)
        dataset.append(series)
    return np.array(dataset)


def adding_anomaly(dataset, information):
    for i in range(len(information)):
        anomaly = information.iloc[i]
        volume = anomaly['volume']
        length = int(anomaly['length'])
        index = int(anomaly['begin_id'])
        root = int(anomaly['root'])
        dataset[root, index: index + length] += volume
    return dataset


