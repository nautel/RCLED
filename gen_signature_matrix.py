import numpy as np
import argparse


def normalization(ts):
    max_value = np.max(ts, axis=1)
    min_value = np.min(ts, axis=1)
    ts = (np.transpose(ts) - min_value) / (max_value - min_value + 1e-8)
    return np.transpose(ts)


def ts2matrix(data, window, time_step):
    n_features = data.shape[0]
    data_length = data.shape[1]
    matrix = []
    for t in range(0, data_length, time_step):
        matrix_t = np.zeros((n_features, n_features))
        if t > window:
            for i in range(n_features):
                for j in range(i, n_features):
                    x_i = data[i, t - window: t]
                    x_j = data[j, t - window: t]
                    matrix_t[i][j] = np.inner(x_i, x_j) / window
                    matrix_t[j][i] = matrix_t[i][j]
        matrix.append(matrix_t)
    return matrix


def parse_args():
    parser = argparse.ArgumentParser("Convert time series to matrix")
    parser.add_argument("--input_dir", default='./data/data_ts/X.npy', type=str)
    parser.add_argument("--output_dir", default='./data/data_matrix/', type=str)
    parser.add_argument("--time_step", default=10)
    parser.add_argument("--window", default=10)

    return parser.parse_args()


def main():
    args = parse_args()
    data = np.load(args.input_dir).T
    data_normalized = normalization(data)
    matrix = ts2matrix(data_normalized, args.window)
    np.save(args.output_dir + str(args.window), matrix)


if __name__ == "__main__":
    main()
