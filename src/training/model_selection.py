import os
import json
import logging
import pandas as pd

class ModelSelector:
    def __init__(self, primary_metric='rmse', secondary_metric='mae'):
        self.primary_metric = primary_metric
        self.secondary_metric = secondary_metric
        self.logger = logging.getLogger(__name__)

    def select_best_model(self, model_metrics: dict) -> str:
        """
        model_metrics format: { 'ModelA': {'rmse': 10.5, 'mae': 8.2}, ... }
        Uses validation performance only.
        """
        if not model_metrics:
            raise ValueError("No model metrics provided for selection.")

        best_model = None
        best_primary = float('inf')
        best_secondary = float('inf')

        for model_name, metrics in model_metrics.items():
            primary_val = metrics.get(self.primary_metric)
            secondary_val = metrics.get(self.secondary_metric)

            if primary_val is None:
                continue

            if primary_val < best_primary:
                best_model = model_name
                best_primary = primary_val
                best_secondary = secondary_val if secondary_val is not None else float('inf')
            elif primary_val == best_primary:
                if secondary_val is not None and secondary_val < best_secondary:
                    best_model = model_name
                    best_secondary = secondary_val

        if best_model is None:
            self.logger.warning("Could not determine best model. Returning arbitrarily if possible.")
            best_model = list(model_metrics.keys())[0]
            
        return best_model

    def create_comparison_table(self, model_metrics: dict) -> pd.DataFrame:
        """
        Creates a formatted comparison table from a dictionary of model metrics.
        """
        rows = []
        for model_name, metrics in model_metrics.items():
            row = {'Model': model_name}
            row.update(metrics)
            rows.append(row)
            
        df = pd.DataFrame(rows)
        # Sort by primary metric if it exists
        if self.primary_metric in df.columns:
            df = df.sort_values(by=self.primary_metric)
        return df

    def save_selection_result(self, result: dict, path: str):
        """
        Saves selection result to a JSON file.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(result, f, indent=4)
        self.logger.info(f"Model selection result saved to {path}")
