import os
import sys

def main():
    print("================================================================")
    print("     TESTING TRAINED AI SURROGATE ACROSS MULTIPLE ENVIRONMENTS  ")
    print("================================================================\n")
    print("1. ENV 1: Peak Demand & Heavy Load Stress Test (120% - 150%)")
    print("2. ENV 2: High Reactive Load & Low Power Factor Stress Test")
    print("3. ENV 3: N-1 Contingency (Line 4-5 Outage Stress Test)\n")
    
    # Run ENV 1
    print(">>> EXECUTING ENV 1...")
    os.system("cd env1 && python3 evaluate_env.py && cd ..")
    
    # Run ENV 2
    print("\n>>> EXECUTING ENV 2...")
    os.system("cd env2 && python3 evaluate_env.py && cd ..")
    
    # Run ENV 3
    print("\n>>> EXECUTING ENV 3...")
    os.system("cd env3 && python3 evaluate_env.py && cd ..")
    
    print("\n================================================================")
    print(" All environment evaluations completed successfully!")
    print(" - ENV1 Results & Figures: env1/figures/")
    print(" - ENV2 Results & Figures: env2/figures/")
    print(" - ENV3 Results & Figures: env3/figures/")
    print("================================================================")

if __name__ == "__main__":
    main()
