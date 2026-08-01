import os

import joblib
import pandas as pd

from preprocessing import preprocess_data
from feature_engineering import (
    build_bureau_features,
    build_bureau_balance_features,
    build_previous_applications_features,
    build_pos_cash_features,
    build_credit_card_features,
    build_installments_features,
)

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'model.pkl')
TEST_FEATURES_CACHE_PATH = os.path.join(DATA_DIR, 'test_features_cache.parquet')

_bundle = None
_test_features = None


def _load_bundle():
    global _bundle
    if _bundle is None:
        _bundle = joblib.load(MODEL_PATH)
    return _bundle


def _load_test_features():
    """Build the same merged feature table used for training, but for
    application_test.csv. Cached in memory for the life of the process, and
    on disk as parquet so restarting the process doesn't re-join the raw
    CSVs (the supplementary tables are multi-million rows and take ~5min)."""
    global _test_features
    if _test_features is not None:
        return _test_features

    if os.path.exists(TEST_FEATURES_CACHE_PATH):
        _test_features = pd.read_parquet(TEST_FEATURES_CACHE_PATH)
        return _test_features

    test_df = pd.read_csv(os.path.join(DATA_DIR, 'application_test.csv'))

    bureau = pd.read_csv(os.path.join(DATA_DIR, 'bureau.csv'))
    test_df = test_df.merge(build_bureau_features(bureau), on='SK_ID_CURR', how='left')

    bureau_balance = pd.read_csv(os.path.join(DATA_DIR, 'bureau_balance.csv'))
    test_df = test_df.merge(
        build_bureau_balance_features(bureau_balance, bureau), on='SK_ID_CURR', how='left'
    )

    previous_applications = pd.read_csv(os.path.join(DATA_DIR, 'previous_application.csv'))
    test_df = test_df.merge(
        build_previous_applications_features(previous_applications), on='SK_ID_CURR', how='left'
    )

    pos_cash = pd.read_csv(os.path.join(DATA_DIR, 'POS_CASH_balance.csv'))
    test_df = test_df.merge(build_pos_cash_features(pos_cash), on='SK_ID_CURR', how='left')

    credit_card = pd.read_csv(os.path.join(DATA_DIR, 'credit_card_balance.csv'))
    test_df = test_df.merge(build_credit_card_features(credit_card), on='SK_ID_CURR', how='left')

    installments = pd.read_csv(os.path.join(DATA_DIR, 'installments_payments.csv'))
    test_df = test_df.merge(build_installments_features(installments), on='SK_ID_CURR', how='left')

    _test_features = test_df.set_index('SK_ID_CURR')
    _test_features.to_parquet(TEST_FEATURES_CACHE_PATH)
    return _test_features


def predict_applicant(sk_id_curr):
    """Run the trained model on a single applicant from application_test.csv,
    reproducing the exact preprocessing used at training time."""
    bundle = _load_bundle()
    test_features = _load_test_features()

    if sk_id_curr not in test_features.index:
        raise KeyError(f'SK_ID_CURR {sk_id_curr} not found in application_test.csv')

    applicant_row = test_features.loc[[sk_id_curr]]

    x_clean, _, _, _ = preprocess_data(
        applicant_row,
        fill_values=bundle['fill_values'],
        encoders=bundle['encoders'],
        cols_to_drop=bundle['dropped_cols'],
    )
    x_clean = x_clean.reindex(columns=bundle['feature_names'])

    proba = bundle['model'].predict_proba(x_clean)[:, 1][0]
    return {
        'SK_ID_CURR': int(sk_id_curr),
        'default_probability': float(proba),
        'prediction': int(proba >= 0.5),
    }


if __name__ == '__main__':
    sample_id = _load_test_features().index[0]
    print(predict_applicant(sample_id))
