import numpy as np
import pandas as pd

"""
Model evaluation metrics.
Handles edge cases: zero actuals, near-zero actuals, perfect predictions.
"""

def compute_mae(y_true, y_pred) -> float:
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))

def compute_rmse(y_true, y_pred) -> float:
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return float(np.sqrt(np.mean(np.square(y_true - y_pred))))

def compute_mape(y_true, y_pred, epsilon=1.0) -> float:
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true >= epsilon
    if not np.any(mask):
        return float('nan')
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)

def compute_smape(y_true, y_pred) -> float:
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    numerator = np.abs(y_true - y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    mask = denominator != 0
    if not np.any(mask):
        return 0.0
    return float(np.mean(numerator[mask] / denominator[mask]) * 100)

def compute_wape(y_true, y_pred) -> float:
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    sum_abs_err = np.sum(np.abs(y_true - y_pred))
    sum_actual = np.sum(np.abs(y_true))
    if sum_actual == 0:
        return float('nan')
    return float((sum_abs_err / sum_actual) * 100)

def compute_r2(y_true, y_pred) -> float:
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    ss_res = np.sum(np.square(y_true - y_pred))
    ss_tot = np.sum(np.square(y_true - np.mean(y_true)))
    if ss_tot == 0:
        return float('nan')
    return float(1 - (ss_res / ss_tot))

def compute_all_metrics(y_true, y_pred) -> dict:
    return {
        'mae': compute_mae(y_true, y_pred),
        'rmse': compute_rmse(y_true, y_pred),
        'mape': compute_mape(y_true, y_pred),
        'smape': compute_smape(y_true, y_pred),
        'wape': compute_wape(y_true, y_pred),
        'r2': compute_r2(y_true, y_pred)
    }

def evaluate_per_horizon(results_df: pd.DataFrame, horizons=[1, 7, 30]) -> pd.DataFrame:
    metrics_list = []
    for h in horizons:
        df_h = results_df[results_df['horizon'] == h]
        if df_h.empty:
            continue
        
        m = compute_all_metrics(df_h['actual'], df_h['predicted'])
        m['horizon'] = h
        m['sample_size'] = len(df_h)
        metrics_list.append(m)
        
    if not metrics_list:
        return pd.DataFrame()
        
    return pd.DataFrame(metrics_list)

def evaluate_per_segment(results_df: pd.DataFrame, group_cols: list) -> pd.DataFrame:
    results = []
    for name, group in results_df.groupby(group_cols):
        m = compute_all_metrics(group['actual'], group['predicted'])
        if isinstance(name, tuple):
            for col, val in zip(group_cols, name):
                m[col] = val
        else:
            m[group_cols[0]] = name
        m['sample_size'] = len(group)
        results.append(m)
        
    return pd.DataFrame(results)

def create_evaluation_report(results_df: pd.DataFrame) -> dict:
    overall = compute_all_metrics(results_df['actual'], results_df['predicted'])
    
    report = {
        'overall_metrics': overall,
        'total_samples': len(results_df)
    }
    
    if 'horizon' in results_df.columns:
        horizon_df = evaluate_per_horizon(results_df, horizons=sorted(results_df['horizon'].unique()))
        report['horizon_metrics'] = horizon_df.to_dict(orient='records')
        
    if 'commodity' in results_df.columns:
        commodity_df = evaluate_per_segment(results_df, ['commodity'])
        report['commodity_metrics'] = commodity_df.to_dict(orient='records')
        
    if 'market_id' in results_df.columns:
        market_df = evaluate_per_segment(results_df, ['market_id'])
        report['market_metrics'] = market_df.to_dict(orient='records')
        
    return report
