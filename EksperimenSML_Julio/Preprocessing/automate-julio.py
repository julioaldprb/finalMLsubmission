import os
import pandas as pd
from sklearn.preprocessing import StandardScaler


def load_data(path_raw):
    """
    Load dataset from CSV file.
    """
    return pd.read_csv(path_raw)


def handle_missing(df):
    """
    Impute missing values for numeric columns with median.
    """
    for col in df.select_dtypes(include=['number']).columns:
        df[col] = df[col].fillna(df[col].median())
    return df


def remove_duplicates(df):
    """
    Remove duplicate rows.
    """
    return df.drop_duplicates()


def drop_non_numeric(df):
    """
    Drop non-numeric columns that are not needed (e.g., 'name').
    """
    return df.drop(columns=['name'], errors='ignore')


def encode_categorical(df):
    """
    One-hot encode the 'origin' column.
    """
    if 'origin' in df.columns:
        df = pd.get_dummies(df, columns=['origin'], drop_first=True)
    return df


def remove_outliers_iqr(df, column):
    """
    Remove outliers in a numeric column based on the IQR method.
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    return df[df[column].between(Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)]


def scale_numeric(df):
    """
    Standardize numeric features.
    """
    num_cols = df.select_dtypes(include=['number']).columns.tolist()
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])
    return df


def preprocess(path_in, path_out):
    """
    Full preprocessing pipeline: load, clean, and save.
    """
    # Load raw data
    df = load_data(path_in)

    # Cleaning steps
    df = handle_missing(df)
    df = remove_duplicates(df)
    df = drop_non_numeric(df)
    df = encode_categorical(df)

    # Optional: remove outliers for 'mpg' and 'horsepower'
    if 'mpg' in df.columns:
        df = remove_outliers_iqr(df, 'mpg')
    if 'horsepower' in df.columns:
        df = remove_outliers_iqr(df, 'horsepower')

    # Scale numeric features
    df = scale_numeric(df)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(path_out), exist_ok=True)
    # Save cleaned data
    df.to_csv(path_out, index=False)
    print(f"Preprocessed data saved to {path_out}")


if __name__ == '__main__':
    # Define default input and output paths
    script_dir = os.path.dirname(__file__)
    raw_path = os.path.join(os.getcwd(), "Automobile_clean.csv")
    out_path = os.path.join(script_dir, 'namadataset_preprocessing', 'Automobile_clean.csv')

    preprocess(raw_path, out_path)
