import pandas as pd


COLS_TO_DROP = ["RowNumber", "CustomerId", "Surname"]  # ID-like cols


def load_dataset(path: str):
    """
    Loads the Bank Churners CSV and returns (X, y) without splitting.
    """
    df = pd.read_csv(path)
    y = df["Exited"]
    X = df.drop(columns=COLS_TO_DROP + ["Exited"])
    return X, y
