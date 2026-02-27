"""
run_m1.py: Script to run M1 training/loading using the new factory structure.
"""
import argparse
from pathlib import Path
import sys

# Ensure package is in path if running as script
sys.path.append(str(Path(__file__).resolve().parent.parent))

from motor_controller_model.config_schema import MotorControllerConfig
from motor_controller_model.m1_factory import get_trained_m1

def main():
    parser = argparse.ArgumentParser(description="Run M1 Network (Spike Input Mode)")
    parser.add_argument("--force-retrain", action="store_true", help="Force retraining even if cache exists")
    args = parser.parse_args()

    # 1. Define Configuration
    config = MotorControllerConfig()
    
    # 2. Define Training Data (Spike Input Case)
    # Paths to spike files (Planner -> M1)
    base_data_dir = Path(__file__).resolve().parent / "dataset_motor_training" / "input_ouput_data"
    
    if not base_data_dir.exists():
        print(f"Error: Dataset directory not found at {base_data_dir}")
        return

    # Example: 90->20 trajectory
    traj1 = {
        'input': (str(base_data_dir / 'N200_9020_planner_p.dat'), 
                  str(base_data_dir / 'N200_9020_planner_n.dat')),
        'output': (str(base_data_dir / 'N200_9020_mc_m1_p.dat'), 
                   str(base_data_dir / 'N200_9020_mc_m1_n.dat'))
    }
    
    # Example: 90->140 trajectory
    traj2 = {
        'input': (str(base_data_dir / 'N200_90140_planner_p.dat'), 
                  str(base_data_dir / 'N200_90140_planner_n.dat')),
        'output': (str(base_data_dir / 'N200_90140_mc_m1_p.dat'), 
                   str(base_data_dir / 'N200_90140_mc_m1_n.dat'))
    }
    
    training_data = [traj1, traj2]

    # 3. Use Factory to get Model
    print(f"Requesting M1 model (Artifacts: sim_results/m1_artifacts)...")
    network = get_trained_m1(config, training_data, args.force_retrain)
    print("M1 Model ready.")

if __name__ == "__main__":
    main()