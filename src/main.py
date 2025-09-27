import numpy as np
from nn import General
from csv_read import load_csv_data

if __name__ == "__main__":
    X, Y, _, _ = load_csv_data("data/score_test.csv")
    nsamples = X.shape[0]

    genel = General([4, 16, 12, 2])
    for i in range(50):
        total_loss = 0
        idx = np.random.permutation(nsamples)
        Xs, Ys = X[idx], Y[idx]
        for x, y in zip(Xs, Ys):
            total_loss += genel.backprop(x, y, 0.0001)
        print(total_loss / nsamples)