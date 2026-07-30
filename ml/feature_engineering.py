import pandas as pd
import numpy as np


def build_bureau_features(bureau):
    """Collapse bureau.csv (one row per previous loan) into one row per SK_ID_CURR."""
    return bureau.groupby('SK_ID_CURR').agg(
        BUREAU_LOAN_COUNT=('SK_ID_BUREAU', 'count'),
        BUREAU_ACTIVE_LOAN_COUNT=('CREDIT_ACTIVE', lambda s: (s == 'Active').sum()),
        BUREAU_AVG_CREDIT_SUM=('AMT_CREDIT_SUM', 'mean'),
        BUREAU_MAX_CREDIT_SUM=('AMT_CREDIT_SUM', 'max'),
        BUREAU_TOTAL_DEBT=('AMT_CREDIT_SUM_DEBT', 'sum'),
        BUREAU_MAX_DAYS_OVERDUE=('CREDIT_DAY_OVERDUE', 'max'),
        BUREAU_AVG_DAYS_CREDIT=('DAYS_CREDIT', 'mean'),
    ).reset_index()

def build_previous_applications_features(previous_applications):
    return previous_applications.groupby('SK_ID_CURR').agg(
        PREV_APP_COUNT=('SK_ID_PREV', 'count'),
        PREV_APP_APPROVED_COUNT=('NAME_CONTRACT_STATUS', lambda s: (s == 'Approved').sum()),
        PREV_APP_REFUSED_COUNT=('NAME_CONTRACT_STATUS', lambda s: (s == 'Refused').sum()),
        PREV_APP_AVG_CREDIT=('AMT_CREDIT', 'mean'),
        PREV_APP_AVG_ANNUITY=('AMT_ANNUITY', 'mean'),
        PREV_APP_AVG_APPLICATION=('AMT_APPLICATION', 'mean'),
        PREV_APP_LAST_DECISION_DAYS=('DAYS_DECISION', 'max'),
    ).reset_index()


def build_pos_cash_features(pos_cash):
    """POS_CASH_balance.csv: monthly history of previous POS/cash loans."""
    return pos_cash.groupby('SK_ID_CURR').agg(
        POS_CASH_LOAN_COUNT=('SK_ID_PREV', 'nunique'),
        POS_CASH_MAX_DPD=('SK_DPD', 'max'),
        POS_CASH_AVG_DPD=('SK_DPD', 'mean'),
        POS_CASH_MAX_DPD_DEF=('SK_DPD_DEF', 'max'),
        POS_CASH_COMPLETED_COUNT=('NAME_CONTRACT_STATUS', lambda s: (s == 'Completed').sum()),
    ).reset_index()


def build_credit_card_features(credit_card):
    """credit_card_balance.csv: monthly history of previous credit card loans."""
    return credit_card.groupby('SK_ID_CURR').agg(
        CC_LOAN_COUNT=('SK_ID_PREV', 'nunique'),
        CC_AVG_BALANCE=('AMT_BALANCE', 'mean'),
        CC_MAX_BALANCE=('AMT_BALANCE', 'max'),
        CC_AVG_CREDIT_LIMIT=('AMT_CREDIT_LIMIT_ACTUAL', 'mean'),
        CC_MAX_DPD=('SK_DPD', 'max'),
        CC_TOTAL_DRAWINGS=('AMT_DRAWINGS_CURRENT', 'sum'),
    ).reset_index()


def build_installments_features(installments):
    """installments_payments.csv: actual repayment history of previous loans."""
    installments = installments.copy()
    installments['PAYMENT_DELAY_DAYS'] = (
        installments['DAYS_ENTRY_PAYMENT'] - installments['DAYS_INSTALMENT']
    )
    payment_ratio = installments['AMT_PAYMENT'] / installments['AMT_INSTALMENT']
    installments['PAYMENT_RATIO'] = payment_ratio.replace([np.inf, -np.inf], np.nan)

    return installments.groupby('SK_ID_CURR').agg(
        INSTALL_PAYMENT_COUNT=('SK_ID_PREV', 'count'),
        INSTALL_AVG_PAYMENT_DELAY=('PAYMENT_DELAY_DAYS', 'mean'),
        INSTALL_MAX_PAYMENT_DELAY=('PAYMENT_DELAY_DAYS', 'max'),
        INSTALL_AVG_PAYMENT_RATIO=('PAYMENT_RATIO', 'mean'),
    ).reset_index()


def build_bureau_balance_features(bureau_balance, bureau):
    """bureau_balance.csv only has SK_ID_BUREAU, so first collapse it to one row
    per previous bureau loan, then use bureau.csv to trace each loan back to SK_ID_CURR."""
    per_loan = bureau_balance.groupby('SK_ID_BUREAU').agg(
        BB_MONTHS_ON_FILE=('MONTHS_BALANCE', 'count'),
        BB_OVERDUE_MONTHS=('STATUS', lambda s: s.isin(['2', '3', '4', '5']).sum()),
    ).reset_index()

    per_loan = per_loan.merge(bureau[['SK_ID_BUREAU', 'SK_ID_CURR']], on='SK_ID_BUREAU', how='left')

    return per_loan.groupby('SK_ID_CURR').agg(
        BB_AVG_MONTHS_ON_FILE=('BB_MONTHS_ON_FILE', 'mean'),
        BB_TOTAL_OVERDUE_MONTHS=('BB_OVERDUE_MONTHS', 'sum'),
    ).reset_index()