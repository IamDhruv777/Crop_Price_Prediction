"""
Main Pipeline Orchestrator Script.
Executes the full end-to-end workflow:
1. Ingestion: Download & cache mandi prices and NASA weather
2. Preprocessing: Cleaning, validation, outlier flagging, gap interpolation
3. Feature Engineering: Lags, rolling windows, agro-seasons, benchmark arbitrage
4. Spatial Graph: Haversine distance matrix, k-NN edges, GraphSAGE embeddings
5. Training: Multi-model training, walk-forward validation, and Super Ensemble
"""
import argparse
import sys
import subprocess
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.utils.logging import get_logger

logger = get_logger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Orchestrate the full crop price prediction pipeline")
    parser.add_argument("--state", type=str, default="Maharashtra", help="Target state")
    parser.add_argument("--crop", type=str, default="Wheat,Rice,Soyabean,Moong", help="Target crops")
    parser.add_argument("--skip-ingestion", action="store_true", help="Skip data ingestion step")
    parser.add_argument("--skip-training", action="store_true", help="Skip model training step")
    parser.add_argument("--benchmark", action="store_true", help="Run full multi-model benchmark suite")
    return parser.parse_args()

def run_step(script_name: str, args_list: list):
    """Helper to run a pipeline step as a subprocess."""
    script_path = Path(__file__).parent / script_name
    if not script_path.exists():
        logger.warning(f"Script {script_name} not found at {script_path}. Skipping.")
        return
        
    cmd = [sys.executable, str(script_path)] + args_list
    logger.info(f"Executing: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info(f"{script_name} completed successfully.\n{result.stdout}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error executing {script_name}.\nExit code: {e.returncode}\nStderr: {e.stderr}\nStdout: {e.stdout}")

def main():
    args = parse_args()
    logger.info("==========================================================================")
    logger.info("=== STARTING END-TO-END CROP PRICE PREDICTION PIPELINE ===")
    logger.info("==========================================================================")
    
    # Step 1: Ingestion
    if not args.skip_ingestion:
        logger.info("--- STEP 1: INGESTION ---")
        run_step("ingest_data.py", ["--state", args.state, "--crop", args.crop])
    else:
        logger.info("--- STEP 1: INGESTION (SKIPPED) ---")
        
    # Step 2: Preprocessing
    logger.info("--- STEP 2: PREPROCESSING ---")
    run_step("preprocess_data.py", [])
    
    # Step 3: Feature Engineering
    logger.info("--- STEP 3: FEATURE ENGINEERING ---")
    run_step("build_features.py", [])
    
    # Step 4: Graph Construction
    logger.info("--- STEP 4: SPATIAL GRAPH CONSTRUCTION ---")
    run_step("build_graph.py", [])
    
    # Step 5: Model Training & Super Ensemble
    if not args.skip_training:
        logger.info("--- STEP 5: MODEL TRAINING & ENSEMBLE ---")
        if args.benchmark:
            run_step("run_full_system_benchmark.py", [])
            run_step("train_spatial_temporal_ensemble.py", [])
            run_step("train_per_crop_champions.py", [])
        else:
            run_step("train_spatial_temporal_ensemble.py", [])
    else:
        logger.info("--- STEP 5: MODEL TRAINING (SKIPPED) ---")
        
    logger.info("==========================================================================")
    logger.info("=== PIPELINE EXECUTION COMPLETED SUCCESSFULLY! ===")
    logger.info("==========================================================================")

if __name__ == "__main__":
    main()
