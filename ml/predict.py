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


def list_applicant_ids(limit=50, offset=0):
    """Return a page of SK_ID_CURR values available to browse/predict on."""
    test_features = _load_test_features()
    return test_features.index[offset:offset + limit].tolist()


def _clean(value):
    """Convert pandas/numpy scalars to plain Python values, NaN -> None."""
    if value is None or pd.isna(value):
        return None
    if hasattr(value, 'item'):
        value = value.item()
    if isinstance(value, (pd.Timestamp,)):
        return str(value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return int(value) if float(value).is_integer() else float(value)
    return value


def get_applicant_profile(sk_id_curr):
    """Pull the underlying application/bureau features for one applicant,
    for display alongside the model's prediction (not fed back into the model)."""
    test_features = _load_test_features()

    if sk_id_curr not in test_features.index:
        raise KeyError(f'SK_ID_CURR {sk_id_curr} not found in application_test.csv')

    row = test_features.loc[sk_id_curr]

    days_birth = _clean(row.get('DAYS_BIRTH'))
    age_years = int(-days_birth // 365) if days_birth is not None else None

    days_employed = _clean(row.get('DAYS_EMPLOYED'))
    employment_years = (
        round(-days_employed / 365, 1) if days_employed is not None and days_employed != 365243 else None
    )

    ext_sources = [
        v for v in (
            _clean(row.get('EXT_SOURCE_1')),
            _clean(row.get('EXT_SOURCE_2')),
            _clean(row.get('EXT_SOURCE_3')),
        ) if v is not None
    ]
    ext_source_avg = round(sum(ext_sources) / len(ext_sources), 3) if ext_sources else None

    return {
        'SK_ID_CURR': int(sk_id_curr),
        'age_years': age_years,
        'gender': _clean(row.get('CODE_GENDER')),
        'income_total': _clean(row.get('AMT_INCOME_TOTAL')),
        'employment_years': employment_years,
        'education': _clean(row.get('NAME_EDUCATION_TYPE')),
        'family_status': _clean(row.get('NAME_FAMILY_STATUS')),
        'occupation': _clean(row.get('OCCUPATION_TYPE')),
        'housing_type': _clean(row.get('NAME_HOUSING_TYPE')),
        'own_car': row.get('FLAG_OWN_CAR') == 'Y',
        'own_realty': row.get('FLAG_OWN_REALTY') == 'Y',
        'children': _clean(row.get('CNT_CHILDREN')),
        'credit_amount': _clean(row.get('AMT_CREDIT')),
        'annuity': _clean(row.get('AMT_ANNUITY')),
        'goods_price': _clean(row.get('AMT_GOODS_PRICE')),
        'ext_source_avg': ext_source_avg,
        'bureau_active_loans': _clean(row.get('BUREAU_ACTIVE_LOAN_COUNT')),
        'bureau_total_debt': _clean(row.get('BUREAU_TOTAL_DEBT')),
        'bureau_max_days_overdue': _clean(row.get('BUREAU_MAX_DAYS_OVERDUE')),
        'prev_app_refused_count': _clean(row.get('PREV_APP_REFUSED_COUNT')),
        'prev_app_approved_count': _clean(row.get('PREV_APP_APPROVED_COUNT')),
    }


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
