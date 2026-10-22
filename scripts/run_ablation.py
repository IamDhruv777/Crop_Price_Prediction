"""
Script to execute the ablation study experiments.
"""
import argparse
import sys
import pandas as pd
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.experiments.ablation import AblationStudy
from src.experiments.configs import get_experiment_config
from src.utils.logger import get_logger
from src.utils.config import load_config
from src.data.io import load_features, load_graph

logger = get_logger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Run ablation study experiments")
    parser.add_argument("--config", type=str, default="config/default.yaml", help="Path to config file")
    parser.add_argument("--output", type=str, default="reports/ablation_results.csv", help="Output CSV path")
    return parser.parse_args()

def main():
    args = parse_args()
    config = load_config(args.config)
    
    logger.info("Starting ablation study pipeline...")
    
    features = load_features(config['paths']['features_data'])
    graph = load_graph(config['paths']['graph_data'])
    
    study = AblationStudy(features=features, graph=graph)
    
    # Define Experiments A through F
    experiments = {
        'Exp_A': get_experiment_config('price_only'),
        'Exp_B': get_experiment_config('price_arrivals'),
        'Exp_C': get_experiment_config('price_arrivals_weather'),
        'Exp_D': get_experiment_config('temporal_added'),
        'Exp_E': get_experiment_config('spatial_gnn_no_weather'),
        'Exp_F': get_experiment_config('full_graphsage')
    }
    
    results = []
    
    for exp_name, exp_config in experiments.items():
        logger.info(f"Running experiment: {exp_name}")
        metrics = study.run_experiment(exp_name, exp_config)
        metrics['experiment'] = exp_name
        metrics['description'] = exp_config['description']
        results.append(metrics)
        logger.info(f"Results for {exp_name}: {metrics}")
        
    # Generate final report
    logger.info("Generating final ablation report CSV...")
    results_df = pd.DataFrame(results)
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)
    
    logger.info(f"Ablation study completed. Results saved to {output_path}")

if __name__ == "__main__":
    main()
