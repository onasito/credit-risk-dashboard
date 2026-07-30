import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split

EXT_SOURCE_COLS = ('EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3')

# Clean dataframe functions
def drop_high_missing_columns(df, threshold=50, cols_to_drop=None, protected_cols=EXT_SOURCE_COLS):
    df = df.copy()
    if cols_to_drop is None:
        missing_pct = df.isnull().mean() * 100
        cols_to_drop = missing_pct[missing_pct > threshold].index.difference(protected_cols)
    return df.drop(columns=cols_to_drop, errors='ignore'), cols_to_drop

def fix_days_employed_anomaly(df):
    df = df.copy()
    df['DAYS_EMPLOYED_ANOM'] = (df['DAYS_EMPLOYED'] == 365243).astype(int)
    df['DAYS_EMPLOYED'] = df['DAYS_EMPLOYED'].replace(365243, np.nan)
    return df

def flag_missing_ext_sources(df, ext_source_cols=EXT_SOURCE_COLS):
    df = df.copy()
    for col in ext_source_cols:
        if col in df.columns:
            df[f'{col}_ISNULL'] = df[col].isnull().astype(int)
    return df

def impute_missing_values(df, fill_values=None, numeric_fill='median', categorical_fill='Unknown'):
    df = df.copy()
    fill_values = fill_values or {}

    numer_cols = df.select_dtypes(include=['float64', 'int64']).columns
    for col in numer_cols:
        if col not in fill_values:
            fill_values[col] = df[col].median() if numeric_fill == 'median' else df[col].mean()
        df[col] = df[col].fillna(fill_values[col])

    obj_cols = df.select_dtypes(include=['object','str']).columns
    for col in obj_cols:
        if col not in fill_values:
            fill_values[col] = 'Unknown' if categorical_fill == 'Unknown' else df[col].mode()[0]
        df[col] = df[col].fillna(fill_values[col])

    return df, fill_values

def encode_categorical_columns(df, encoders=None):
    df = df.copy()
    categorical_cols = df.select_dtypes(include=['object', 'str']).columns
    encoders = encoders or {}

    for col in categorical_cols:
        if col not in encoders:
            encoders[col] = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
            df[col] = encoders[col].fit_transform(df[[col]])
        else:
            df[col] = encoders[col].transform(df[[col]])

    return df, encoders

def preprocess_data(df, threshold=50, numeric_fill='median', categorical_fill='Unknown', fill_values=None, encoders=None, cols_to_drop=None, protected_cols=EXT_SOURCE_COLS):
    df, dropped_cols = drop_high_missing_columns(df, threshold=threshold, cols_to_drop=cols_to_drop, protected_cols=protected_cols)
    df = fix_days_employed_anomaly(df)
    df = flag_missing_ext_sources(df)
    df, fill_values = impute_missing_values(df,fill_values=fill_values, numeric_fill=numeric_fill, categorical_fill=categorical_fill)
    df, encoders = encode_categorical_columns(df, encoders=encoders)
    return df, dropped_cols, fill_values, encoders