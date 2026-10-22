"""
Script to validate, clean, and preprocess ingested raw data.
"""
import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data.preprocessing.cleaner import DataCleaner
from src.data.preprocessing.validator import DataValidator
from src.data.preprocessing.imputer import MissingValueImputer
from src.data.preprocessing.outliers import OutlierDetector
from src.utils.logger import get_logger
from src.utils.config import load_config
from src.data.io import load_raw_data, save_processed_data

logger = get_logger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess raw ingested data")
    parser.add_argument("--config", type=str, default="config/default.yaml", help="Path to config file")
    return parser.parse_args()

def main():
    args = parse_args()
    config = load_config(args.config)
    
    logger.info("Starting data preprocessing pipeline...")
    
    # Load raw data
    logger.info("Loading raw data...")
    raw_data = load_raw_data(config['paths']['raw_data'])
    
    # Validate
    logger.info("Validating raw data...")
    validator = DataValidator(config)
    is_valid, validation_report = validator.validate(raw_data)
    if not is_valid:
        logger.warning(f"Data validation issues found: {validation_report}")
        
    # Clean
    logger.info("Cleaning data...")
    cleaner = DataCleaner(config)
    cleaned_data = cleaner.clean(raw_data)
    
    # Impute missing values
    logger.info("Imputing missing values...")
    imputer = MissingValueImputer(config)
    imputed_data = imputer.fit_transform(cleaned_data)
    
    # Outlier detection and treatment
    logger.info("Treating outliers...")
    outlier_detector = OutlierDetector(config)
    processed_data = outlier_detector.fit_transform(imputed_data)
    
    # Save processed data
    logger.info("Saving interim/processed data...")
    save_processed_data(processed_data, config['paths']['processed_data'])
    
    logger.info("Data preprocessing completed successfully.")

if __name__ == "__main__":
    main()
