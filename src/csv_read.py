import pandas as pd

def load_csv_data(path):
    df = pd.read_csv(path)

    input_cols = ["Gelir", "Yas", "KrediPuani", "CalismaYili"]
    output_cols = ["Skor", "GeriOdemeSuresi"]
    X = df[input_cols].values.astype(float)
    Y = df[output_cols].values.astype(float)

    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    X_norm = (X - X_min) / (X_max - X_min)
    return X_norm, Y, X_min, X_max