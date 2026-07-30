import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix

from preprocessing import preprocess_data
from feature_engineering import (
    build_bureau_features,
    build_bureau_balance_features,
    build_previous_applications_features,
    build_pos_cash_features,
    build_credit_card_features,
    build_installments_features,
)

if __name__ == "__main__":
    train_df = pd.read_csv('./data/application_train.csv')

    bureau = pd.read_csv('./data/bureau.csv')
    bureau_agg = build_bureau_features(bureau)
    train_df = train_df.merge(bureau_agg, on='SK_ID_CURR', how='left')

    bureau_balance = pd.read_csv('./data/bureau_balance.csv')
    bureau_balance_agg = build_bureau_balance_features(bureau_balance, bureau)
    train_df = train_df.merge(bureau_balance_agg, on='SK_ID_CURR', how='left')

    previous_applications = pd.read_csv('./data/previous_application.csv')
    previous_applications_agg = build_previous_applications_features(previous_applications)
    train_df = train_df.merge(previous_applications_agg, on='SK_ID_CURR', how='left')

    pos_cash = pd.read_csv('./data/POS_CASH_balance.csv')
    pos_cash_agg = build_pos_cash_features(pos_cash)
    train_df = train_df.merge(pos_cash_agg, on='SK_ID_CURR', how='left')

    credit_card = pd.read_csv('./data/credit_card_balance.csv')
    credit_card_agg = build_credit_card_features(credit_card)
    train_df = train_df.merge(credit_card_agg, on='SK_ID_CURR', how='left')

    installments = pd.read_csv('./data/installments_payments.csv')
    installments_agg = build_installments_features(installments)
    train_df = train_df.merge(installments_agg, on='SK_ID_CURR', how='left')

    train_df.drop(columns=['SK_ID_CURR'], inplace=True)
    y = train_df['TARGET']
    x = train_df.drop(columns=['TARGET'])

    x_train, x_val, y_train, y_val = train_test_split(
        x, y, test_size=0.2, stratify=y, random_state=42
    )

    # fit preprocessing on train only
    x_train_clean, dropped_cols, fill_values, encoders = preprocess_data(x_train)

    x_val_clean, _, _, _ = preprocess_data(
        x_val, fill_values=fill_values, encoders=encoders, cols_to_drop=dropped_cols
    )

    # default rate is ~8%, so weight positives to counter the imbalance
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    model = XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        eval_metric='auc',
        random_state=42,
    )
    model.fit(x_train_clean, y_train, eval_set=[(x_val_clean, y_val)])

    # --- Evaluate the final trained model on the held-out validation set ---
    y_val_proba = model.predict_proba(x_val_clean)[:, 1]
    y_val_pred = (y_val_proba >= 0.5).astype(int)

    print("\nValidation ROC-AUC:", roc_auc_score(y_val, y_val_proba))
    print("\nClassification report:\n", classification_report(y_val, y_val_pred))
    print("\nConfusion matrix:\n", confusion_matrix(y_val, y_val_pred))
