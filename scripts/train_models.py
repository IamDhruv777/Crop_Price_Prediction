"""
Script to train baseline, classical ML, and deep learning models.
"""
import argparse
import sys
import random
import numpy as np
import torch
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.models.trainer import ModelTrainer
from src.models.validation import WalkForwardValidation
from src.models.registry import ModelRegistry
from src.models.baselines import MovingAverageModel
from src.models.classical import XGBoostModel
from src.models.deep import GraphSAGEModel
from src.utils.logger import get_logger
from src.utils.config import load_config
from src.data.io import load_features, load_graph

logger = get_logger(__name__)

def set_seed(seed: int = 42):
    """Sets random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def parse_args():
    parser = argparse.ArgumentParser(description="Train prediction models")
    parser.add_argument("--config", type=str, default="config/default.yaml", help="Path to config file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--models", type=str, default="all", help="Comma-separated list of models to train or 'all'")
    return parser.parse_args()

def main():
    args = parse_args()
    config = load_config(args.config)
    set_seed(args.seed)
    
    logger.info(f"Starting model training pipeline with seed {args.seed}...")
    
    # Load features and graph
    logger.info("Loading features and graph data...")
    features = load_features(config['paths']['features_data'])
    graph = load_graph(config['paths']['graph_data'])
    
    # Configure validation strategy
    val_strategy = WalkForwardValidation(
        train_window=config['validation']['train_window'],
        test_window=config['validation']['test_window']
    )
    
    models_to_train = []
    model_names = [m.strip().lower() for m in args.models.split(',')]
    
    if 'all' in model_names or 'baseline' in model_names:
        models_to_train.append(MovingAverageModel(config['model_params']['baseline']))
    if 'all' in model_names or 'xgboost' in model_names:
        models_to_train.append(XGBoostModel(config['model_params']['xgboost']))
    if 'all' in model_names or 'graphsage' in model_names:
        models_to_train.append(GraphSAGEModel(config['model_params']['graphsage'], graph=graph))
        
    registry = ModelRegistry(config['paths']['model_registry'])
    
    # Train models
    for model in models_to_train:
        logger.info(f"Training {model.__class__.__name__}...")
        trainer = ModelTrainer(model=model, validation_strategy=val_strategy)
        results = trainer.train(features)
        
        logger.info(f"Training completed for {model.__class__.__name__}. Metrics: {results['metrics']}")
        
        # Save model
        registry.save_model(model, results['metrics'])
        
    logger.info("Model training pipeline completed successfully.")

if __name__ == "__main__":
    main()
