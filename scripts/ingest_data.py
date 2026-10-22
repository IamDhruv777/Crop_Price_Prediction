"""
Script to run data ingestion for mandi prices, weather, arrivals, and production.
"""
import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data.ingestion.mandi import MandiIngestor
from src.data.ingestion.weather import WeatherIngestor
from src.data.ingestion.arrivals import ArrivalIngestor
from src.data.ingestion.production import ProductionIngestor
from src.utils.logger import get_logger
from src.utils.config import load_config

logger = get_logger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Ingest data for crop price prediction")
    parser.add_argument("--state", type=str, default="Maharashtra", help="Target state for ingestion")
    parser.add_argument("--crop", type=str, default="Onion,Tomato,Soybean", help="Comma-separated list of crops")
    parser.add_argument("--start-date", type=str, default="2018-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, default="2023-12-31", help="End date (YYYY-MM-DD)")
    parser.add_argument("--config", type=str, default="config/default.yaml", help="Path to config file")
    return parser.parse_args()

def main():
    args = parse_args()
    config = load_config(args.config)
    
    crops = [c.strip() for c in args.crop.split(",")]
    logger.info(f"Starting data ingestion pipeline...")
    logger.info(f"Parameters: State={args.state}, Crops={crops}, Range={args.start_date} to {args.end_date}")
    
    mandi_ingestor = MandiIngestor(config)
    weather_ingestor = WeatherIngestor(config)
    arrival_ingestor = ArrivalIngestor(config)
    prod_ingestor = ProductionIngestor(config)
    
    for crop in crops:
        logger.info(f"Ingesting data for crop: {crop}")
        mandi_ingestor.ingest(state=args.state, crop=crop, start_date=args.start_date, end_date=args.end_date)
        arrival_ingestor.ingest(state=args.state, crop=crop, start_date=args.start_date, end_date=args.end_date)
        prod_ingestor.ingest(state=args.state, crop=crop)
        
    logger.info(f"Ingesting weather data for state: {args.state}")
    weather_ingestor.ingest(state=args.state, start_date=args.start_date, end_date=args.end_date)
    
    logger.info("Data ingestion completed successfully.")

if __name__ == "__main__":
    main()
