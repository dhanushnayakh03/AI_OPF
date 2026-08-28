import os
import sys

def main():
    print("================================================================")
    print("  AI-Based Optimal Reactive Power Dispatch (IEEE-14 Bus System) ")
    print("================================================================\n")
    
    # Step 1: Data Generation
    print(">>> STEP 1: Running Data Generation (data_generation.py)...")
    exit_code = os.system("python3 data_generation.py")
    if exit_code != 0:
        print("[ERROR] Data generation failed!")
        sys.exit(1)
        
    # Step 2: Model Training
    print("\n>>> STEP 2: Training Neural Network (train_model.py)...")
    exit_code = os.system("python3 train_model.py")
    if exit_code != 0:
        print("[ERROR] Model training failed!")
        sys.exit(1)
        
    # Step 3: Evaluation & Visualization
    print("\n>>> STEP 3: Evaluating Surrogate & Generating Figures (evaluate_model.py)...")
    exit_code = os.system("python3 evaluate_model.py")
    if exit_code != 0:
        print("[ERROR] Evaluation failed!")
        sys.exit(1)
        
    print("\n================================================================")
    print(" Pipeline completed successfully!")
    print(" - Dataset saved: dataset.csv")
    print(" - Trained model: best_model.pth")
    print(" - Loss curve:    training_loss.png")
    print(" - Figures saved: figures/fig1_topology.png to fig6_computation_time.png")
    print("================================================================")

if __name__ == "__main__":
    main()
