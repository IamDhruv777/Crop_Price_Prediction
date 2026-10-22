"""
Main pipeline orchestrator script.
"""
import argparse
import sys
import subprocess
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.utils.logger import get_logger

logger = get_logger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Orchestrate the full crop price prediction pipeline")
    parser.add_argument("--state", type=str, default="Maharashtra", help="Target state")
    parser.add_argument("--crop", type=str, default="Onion,Tomato,Soybean", help="Target crops")
    parser.add_argument("--skip-ingestion", action="store_true", help="Skip data ingestion step")
    parser.add_argument("--skip-training", action="store_true", help="Skip model training step")
    return parser.parse_args()

def run_step(script_name: str, args_list: list):
    """Helper to run a pipeline step as a subprocess."""
    script_path = Path(__file__).parent / script_name
    cmd = [sys.executable, str(script_path)] + args_list
    
    logger.info(f"Executing: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info(f"{script_name} completed successfully.\n{result.stdout}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error executing {script_name}.\nExit code: {e.returncode}\nStderr: {e.stderr}\nStdout: {e.stdout}")
        sys.exit(1)

def main():
    args = parse_args()
    logger.info("Starting end-to-end crop price prediction pipeline...")
    
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
    logger.info("--- STEP 4: GRAPH CONSTRUCTION ---")
    run_step("build_graph.py", [])
    
    # Step 5: Model Training & Selection
    if not args.skip_training:
        logger.info("--- STEP 5: MODEL TRAINING ---")
        run_step("train_models.py", [])
    else:
        logger.info("--- STEP 5: MODEL TRAINING (SKIPPED) ---")
        
    logger.info("End-to-end pipeline execution completed successfully!")

if __name__ == "__main__":
    main()
