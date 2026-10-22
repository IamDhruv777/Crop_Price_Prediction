"""
Script to engineer temporal, price, arrival, and weather features.
"""
import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.features.pipeline import FeaturePipeline
from src.features.temporal import TemporalFeatures
from src.features.price import PriceFeatures
from src.features.weather import WeatherFeatures
from src.utils.logger import get_logger
from src.utils.config import load_config
from src.data.io import load_processed_data, save_features

logger = get_logger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Build features from processed data")
    parser.add_argument("--config", type=str, default="config/default.yaml", help="Path to config file")
    return parser.parse_args()

def main():
    args = parse_args()
    config = load_config(args.config)
    
    logger.info("Starting feature engineering pipeline...")
    
    # Load processed data
    logger.info("Loading processed data...")
    processed_data = load_processed_data(config['paths']['processed_data'])
    
    # Initialize pipeline
    pipeline = FeaturePipeline(
        steps=[
            ('temporal', TemporalFeatures()),
            ('price', PriceFeatures(lags=config['features']['price_lags'])),
            ('weather', WeatherFeatures())
        ]
    )
    
    # Fit and transform
    logger.info("Engineering features...")
    features = pipeline.fit_transform(processed_data)
    
    # Save features
    logger.info("Saving feature datasets to data/features/...")
    save_features(features, config['paths']['features_data'])
    
    logger.info("Feature engineering completed successfully.")

if __name__ == "__main__":
    main()
