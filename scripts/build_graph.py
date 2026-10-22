"""
Script to build the spatial market graph.
"""
import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.models.graph.builder import SpatialGraphBuilder
from src.models.graph.distance import DistanceCalculator
from src.utils.logger import get_logger
from src.utils.config import load_config
from src.data.io import load_market_metadata, save_graph

logger = get_logger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Build spatial market graph")
    parser.add_argument("--config", type=str, default="config/default.yaml", help="Path to config file")
    parser.add_argument("--k-neighbors", type=int, default=5, help="Number of neighbors for KNN graph")
    return parser.parse_args()

def main():
    args = parse_args()
    config = load_config(args.config)
    
    logger.info("Starting spatial market graph construction...")
    
    # Load market metadata (lat/lon etc.)
    logger.info("Loading market metadata...")
    market_data = load_market_metadata(config['paths']['market_metadata'])
    
    # Calculate pairwise distances
    logger.info("Calculating pairwise distances...")
    dist_calc = DistanceCalculator(metric='haversine')
    distance_matrix = dist_calc.calculate(market_data)
    
    # Build KNN graph
    logger.info(f"Computing KNN graph with K={args.k_neighbors}...")
    graph_builder = SpatialGraphBuilder(k=args.k_neighbors)
    graph = graph_builder.build_from_distances(distance_matrix, market_data)
    
    # Save spatial graph
    logger.info("Saving spatial graph edges...")
    save_graph(graph, config['paths']['graph_data'])
    
    logger.info("Graph construction completed successfully.")

if __name__ == "__main__":
    main()
