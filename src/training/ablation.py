import pandas as pd
import numpy as np
import logging

"""
Ablation study: measures contribution of each feature group.
Experiments:
  A: Price only
  B: Price + Arrivals
  C: Price + Arrivals + Weather
  D: Price + Arrivals + Weather + Temporal
  E: Price + Arrivals + Weather + Temporal + Spatial
  F: Price + Arrivals + Weather + Temporal + Spatial + GraphSAGE
"""

class AblationStudy:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def define_experiments(self) -> dict:
        return {
            'A': ['price'],
            'B': ['price', 'arrivals'],
            'C': ['price', 'arrivals', 'weather'],
            'D': ['price', 'arrivals', 'weather', 'temporal'],
            'E': ['price', 'arrivals', 'weather', 'temporal', 'spatial'],
            'F': ['price', 'arrivals', 'weather', 'temporal', 'spatial', 'graphsage']
        }

    def _filter_features(self, X, feature_group_prefixes):
        # Helper to filter columns based on group prefixes
        cols = []
        for prefix in feature_group_prefixes:
            cols.extend([c for c in X.columns if c.startswith(prefix)])
        # Also include target lags if 'price' is in prefixes
        if 'price' in feature_group_prefixes:
            cols.extend([c for c in X.columns if 'price_lag' in c or 'target_lag' in c])
        return X[list(set(cols))]

    def run_experiment(self, experiment_name: str, feature_subset: list, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series, model) -> dict:
        self.logger.info(f"Running experiment {experiment_name}...")
        
        X_train_sub = self._filter_features(X_train, feature_subset)
        X_val_sub = self._filter_features(X_val, feature_subset)
        
        if X_train_sub.empty:
            self.logger.warning(f"Experiment {experiment_name} has no features. Skipping.")
            return {}

        model.fit(X_train_sub, y_train)
        y_pred = model.predict(X_val_sub)
        
        from .evaluation import compute_all_metrics
        metrics = compute_all_metrics(y_val, y_pred)
        metrics['experiment'] = experiment_name
        metrics['features'] = ", ".join(feature_subset)
        return metrics

    def run_all_experiments(self, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series, model) -> pd.DataFrame:
        experiments = self.define_experiments()
        results = []
        for exp_name, features in experiments.items():
            metrics = self.run_experiment(exp_name, features, X_train, y_train, X_val, y_val, model)
            if metrics:
                results.append(metrics)
        return pd.DataFrame(results)

    def run_spatial_ablation(self, X_train, y_train, X_val, y_val, model) -> pd.DataFrame:
        self.logger.info("Running spatial ablation...")
        experiments = {
            'No_Spatial': ['price', 'arrivals', 'weather', 'temporal'],
            'Distance_Only': ['price', 'arrivals', 'weather', 'temporal', 'spatial_dist'],
            'Neighbors_Only': ['price', 'arrivals', 'weather', 'temporal', 'spatial_neighbor'],
            'GraphSAGE': ['price', 'arrivals', 'weather', 'temporal', 'graphsage']
        }
        
        results = []
        for exp_name, features in experiments.items():
            metrics = self.run_experiment(exp_name, features, X_train, y_train, X_val, y_val, model)
            if metrics:
                results.append(metrics)
        return pd.DataFrame(results)

    def generate_ablation_report(self, results: pd.DataFrame) -> dict:
        if results.empty:
            return {}
            
        report = {
            'experiments': results.to_dict(orient='records')
        }
        
        # Calculate relative improvement if 'rmse' is present
        if 'rmse' in results.columns and 'A' in results['experiment'].values:
            base_rmse = results[results['experiment'] == 'A']['rmse'].iloc[0]
            improvements = {}
            for _, row in results.iterrows():
                imp = ((base_rmse - row['rmse']) / base_rmse) * 100
                improvements[row['experiment']] = imp
            report['relative_improvement_pct'] = improvements
            
        return report
