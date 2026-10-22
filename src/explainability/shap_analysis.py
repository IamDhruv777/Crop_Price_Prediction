"""
SHAP-based explainability for model predictions.
Uses correct language: 'contributed to model prediction' (not causality).
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Optional
try:
    import shap
except ImportError:
    shap = None

class SHAPExplainer:
    def __init__(self, model: Any, model_type: str = 'xgboost'):
        self.model = model
        self.model_type = model_type
        if shap is None:
            self.explainer = None
        elif model_type in ['xgboost', 'random_forest', 'lightgbm']:
            self.explainer = shap.TreeExplainer(model)
        elif model_type in ['pytorch', 'keras', 'tensorflow']:
            # Placeholder for DeepExplainer
            self.explainer = None
        else:
            self.explainer = shap.Explainer(model)

    def compute_shap_values(self, X: pd.DataFrame, feature_names: Optional[List[str]] = None) -> Any:
        """
        Computes SHAP values.
        """
        if self.explainer is None:
            raise ImportError("SHAP library is not installed or explainer not initialized.")
        if feature_names:
            X = X[feature_names]
        shap_values = self.explainer(X)
        return shap_values

    def get_feature_importance(self, shap_values: Any, feature_names: List[str]) -> pd.Series:
        """
        Returns mean(|SHAP|) per feature, sorted descending.
        """
        vals = np.abs(shap_values.values).mean(0)
        importance = pd.Series(vals, index=feature_names).sort_values(ascending=False)
        return importance

    def plot_summary(self, shap_values: Any, X: pd.DataFrame, feature_names: List[str], save_path: Optional[str] = None):
        """
        SHAP summary plot (beeswarm).
        """
        plt.figure()
        shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        plt.close()

    def plot_dependence(self, shap_values: Any, X: pd.DataFrame, feature_name: str, save_path: Optional[str] = None):
        """
        SHAP dependence plot.
        """
        plt.figure()
        shap.dependence_plot(feature_name, shap_values.values, X, show=False)
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        plt.close()

    def explain_single_prediction(self, X_instance: pd.DataFrame, feature_names: List[str]) -> Dict[str, Any]:
        """
        Returns local explanation for one prediction.
        """
        shap_values = self.compute_shap_values(X_instance, feature_names)
        vals = shap_values.values[0]
        base_value = shap_values.base_values[0] if hasattr(shap_values, 'base_values') else 0
        
        contributions = []
        for name, val in zip(feature_names, vals):
            direction = '+' if val > 0 else '-'
            contributions.append({
                "feature": name,
                "contribution": float(val),
                "direction": direction,
                "importance": abs(float(val))
            })
            
        contributions.sort(key=lambda x: x["importance"], reverse=True)
        
        return {
            "base_value": float(base_value),
            "top_contributions": contributions[:5],
            "message": "These features contributed most to the model prediction."
        }

    def generate_explanation_text(self, local_explanation: Dict[str, Any]) -> str:
        """
        Generates human-readable text explanation.
        """
        text = "The model predicted the price starting from a base value. "
        text += "The top features that contributed to the model prediction were:\n"
        for item in local_explanation["top_contributions"]:
            effect = "increased" if item["direction"] == '+' else "decreased"
            text += f"- {item['feature']} {effect} the predicted value by {abs(item['contribution']):.2f}.\n"
        
        text += "\nNote: The model assigned weight to these features to generate the prediction, this does not imply a causal relationship."
        return text

    def to_dataframe(self, shap_values: Any, feature_names: List[str]) -> pd.DataFrame:
        """
        Converts SHAP values to DataFrame.
        """
        return pd.DataFrame(shap_values.values, columns=feature_names)
