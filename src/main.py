import numpy as np
from nn import General
from csv_read import load_csv_data

if __name__ == "__main__":
    X, Y, xmin, xmax = load_csv_data("data/score_train.csv")
    nsamples = X.shape[0]
    """"
    genel = General([4, 16, 12, 2], ["leakyrelu", "leakyrelu", "leakyrelu"])
    for i in range(10):
        total_loss = 0
        idx = np.random.permutation(nsamples)
        Xs, Ys = X[idx], Y[idx]
        for x, y in zip(Xs, Ys):
            total_loss += genel.backprop(x, y, 0.001)
        print(total_loss / nsamples)
    genel.save("models/model.weights")
    """
    genel = General.load("models\model.weights")
    while True:
        print("\n--- Test için değer gir ---")
        gelir = float(input("Gelir: "))
        yas = float(input("Yaş: "))
        kredi = float(input("Kredi Puani: "))
        calisma = float(input("Çalışma Yılı: "))

        test_x = np.array([gelir, yas, kredi, calisma], dtype=float)
        test_x_norm = (test_x - xmin) / (xmax - xmin)

        pred = genel.forward(test_x_norm)

        skor = (gelir / 80000) * 30 + (kredi / 850) * 40 + (yas / 85) * 20 + (calisma / 60) * 10
        gos = 60 - (gelir / 80000) * 20 - (kredi / 850) * 25 + (yas / 85) * 10
        gos = np.clip(gos, 6, 60)

        print("\nTest input (normalize edilmiş):", test_x_norm)
        print(f"Tahmin: {pred[0] / 100:.4f} | {pred[1]:.4f}")
        print(f"Normal: {skor / 100:.4f} | {gos:.4f}")