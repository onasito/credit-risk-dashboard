import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

train_df = pd.read_csv('./data/application_train.csv', nrows=1000)
print("Initial DataFrame shape:", train_df.shape)

# Clean dataframe functions
def drop_high_missing_columns(df, threshold=50):
    df = df.copy()
    missing_pct = df.isnull().mean() * 100
    cols_to_drop = missing_pct[missing_pct > threshold].index
    return df.drop(columns=cols_to_drop), cols_to_drop

def fix_days_employed_anomaly(df):
    df = df.copy()
    df['DAYS_EMPLOYED_ANOM'] = (df['DAYS_EMPLOYED'] == 365243).astype(int)
    df['DAYS_EMPLOYED'] = df['DAYS_EMPLOYED'].replace(365243, np.nan)
    return df

def impute_missing_values(df, fill_values=None, numeric_fill='median', categorical_fill='Unknown'):
    df = df.copy()
    fill_values = fill_values or {}

    numer_cols = df.select_dtypes(include=['float64', 'int64']).columns
    for col in numer_cols:
        if col not in fill_values:
            fill_values[col] = df[col].median() if numeric_fill == 'median' else df[col].mean()
        df[col] = df[col].fillna(fill_values[col])

    obj_cols = df.select_dtypes(include=['str']).columns
    for col in obj_cols:
        if col not in fill_values:
            fill_values[col] = 'Unknown' if categorical_fill == 'Unknown' else df[col].mode()[0]
        df[col] = df[col].fillna(fill_values[col])

    return df, fill_values

def encode_categorical_columns(df, encoders=None):
    df = df.copy()
    categorical_cols = df.select_dtypes(include=['str']).columns
    encoders = encoders or {}

    for col in categorical_cols:
        if col not in encoders:
            encoders[col] = LabelEncoder()
            df[col] = encoders[col].fit_transform(df[col])
        else:
            df[col] = encoders[col].transform(df[col])

    return df, encoders

def preprocess_data(df, threshold=50, numeric_fill='median', categorical_fill='Unknown', fill_values=None, encoders=None):
    df, dropped_cols = drop_high_missing_columns(df, threshold=threshold)
    df = fix_days_employed_anomaly(df)
    df, fill_values = impute_missing_values(df,fill_values=fill_values, numeric_fill=numeric_fill, categorical_fill=categorical_fill)
    df, encoders = encode_categorical_columns(df, encoders=encoders)
    return df, dropped_cols, fill_values, encoders

print("Original shape:", train_df.shape)

df_clean, dropped_cols, fill_values, encoders = preprocess_data(train_df)

print("Dropped columns:", list(dropped_cols))
print("Shape after dropping high-missing columns:", train_df.shape[1] - len(dropped_cols))
print("Final cleaned shape:", df_clean.shape)
print("Remaining missing values:", df_clean.isnull().sum().sum())
print("Data types after encoding:\n", df_clean.dtypes.value_counts())
print(df_clean.head())