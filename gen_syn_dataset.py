import argparse
import os.path
from os import listdir
import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser("Generating synthtic datase")
    parser.add_argument("--output_dir", default='./data/data_ts/synthetic/', type=str,
                        help='Output directory for synthetic dataset')
    parser.add_argument("--noise_level", default=20, type=int,
                        help='Selecting noise level (%) for synthetic dataset')
    parser.add_argument("--ts_length", default=20000, type=int)

    parser.add_argument("--num_vars", default=30, type=int,
                        help='Selecting number of time series variables for synthetic dataset')


    parser.add_argument("--random_seed", default=1)

    return parser.parse_args()


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
    for i in range(num_var):
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


def main():
    args = parse_args()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    information = pd.read_csv(args.output_dir + 'labeled_anomalies.csv')
    s = generate_time_series_dataset(args.num_vars, args.ts_length, args.noise_level)
    s = adding_anomaly(s, information)
    np.save(args.output_dir + f'/{args.noise_level}', s)


if __name__ == '__main__':
    main()
