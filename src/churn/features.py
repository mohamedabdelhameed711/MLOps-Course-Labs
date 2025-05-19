from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(numeric_cols, categorical_cols):
    """
    Returns a ColumnTransformer that:
      • scales numeric columns
      • one-hot encodes categorical columns (drop='first' to avoid k-dummies trap)
    """
    numeric = StandardScaler()
    categor = OneHotEncoder(drop="first", handle_unknown="ignore")

    return ColumnTransformer(
        transformers=[
            ("num", numeric, numeric_cols),
            ("cat", categor, categorical_cols),
        ]
    )
