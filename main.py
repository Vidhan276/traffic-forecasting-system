"""
Traffic Forecasting using Graph Neural Networks
================================================
Main entry point for the project.

This script orchestrates the full pipeline:
  1. Generate realistic synthetic traffic data
  2. Train the T-GCN model
  3. Evaluate on test data
  4. Generate traffic prediction maps

Usage:
    python main.py --all          # Run everything
    python main.py --generate     # Only generate data
    python main.py --train        # Only train model
    python main.py --evaluate     # Only evaluate model
    python main.py --predict      # Only generate prediction maps
    python main.py --dashboard    # Launch Streamlit dashboard

Authors: EDI Project -- Second Year Engineering
"""

import argparse
import subprocess
import sys
import os
import io

from colorama import Fore, Style, init

# Enable colored output on Windows
init(autoreset=True)

# Fix encoding for Windows console
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")


def print_header(text):
    """Print a formatted section header."""
    print()
    print(Fore.CYAN + "=" * 60)
    print(Fore.CYAN + f"  {text}")
    print(Fore.CYAN + "=" * 60)
    print()


def print_success(text):
    """Print a success message in green."""
    print(Fore.GREEN + f"  [OK] {text}")


def print_error(text):
    """Print an error message in red."""
    print(Fore.RED + f"  [FAIL] {text}")


def run_script(script_path, description):
    """Run a Python script and handle errors."""
    print_header(description)

    if not os.path.exists(script_path):
        print_error(f"File not found: {script_path}")
        return False

    try:
        result = subprocess.run(
            [sys.executable, script_path],
            check=True
        )
        print()
        print_success(f"{description} completed successfully!")
        return True

    except subprocess.CalledProcessError as e:
        print_error(f"{description} failed with error code {e.returncode}")
        return False


def main():
    # ---- Parse command-line arguments ----
    parser = argparse.ArgumentParser(
        description="Traffic Forecasting using Graph Neural Networks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --all          Run the full pipeline
  python main.py --train        Train the model only
  python main.py --predict      Generate prediction maps
  python main.py --dashboard    Launch the Streamlit dashboard
        """
    )

    parser.add_argument("--generate", action="store_true",
                        help="Generate synthetic traffic data")
    parser.add_argument("--train", action="store_true",
                        help="Train the T-GCN model")
    parser.add_argument("--evaluate", action="store_true",
                        help="Evaluate model on test data")
    parser.add_argument("--predict", action="store_true",
                        help="Generate traffic prediction maps")
    parser.add_argument("--dashboard", action="store_true",
                        help="Launch the Streamlit dashboard")
    parser.add_argument("--all", action="store_true",
                        help="Run the full pipeline (generate + train + evaluate + predict)")

    args = parser.parse_args()

    # If no arguments given, show help
    if not any(vars(args).values()):
        parser.print_help()
        print()
        print(Fore.YELLOW + "  Tip: Use --all to run the full pipeline")
        return

    # ---- Print project banner ----
    print()
    print(Fore.CYAN + Style.BRIGHT + "  +----------------------------------------------+")
    print(Fore.CYAN + Style.BRIGHT + "  |   Traffic Forecasting using GNN              |")
    print(Fore.CYAN + Style.BRIGHT + "  |   Kothrud, Pune Road Network                 |")
    print(Fore.CYAN + Style.BRIGHT + "  +----------------------------------------------+")
    print()

    # ---- Run requested steps ----

    if args.all or args.generate:
        success = run_script("data/generate_data.py", "Step 1: Generating Traffic Data")
        if not success and args.all:
            print_error("Pipeline stopped due to data generation failure.")
            return

    if args.all or args.train:
        success = run_script("ml/train_model.py", "Step 2: Training T-GCN Model")
        if not success and args.all:
            print_error("Pipeline stopped due to training failure.")
            return

    if args.all or args.evaluate:
        success = run_script("ml/evaluate.py", "Step 3: Evaluating Model")
        if not success and args.all:
            print_error("Pipeline stopped due to evaluation failure.")
            return

    if args.all or args.predict:
        success = run_script("ml/predict.py", "Step 4: Generating Prediction Maps")
        if not success and args.all:
            print_error("Pipeline stopped due to prediction failure.")
            return

    if args.all:
        print_header("Pipeline Complete!")
        print_success("All steps finished successfully.")
        print()
        print(f"  Run the dashboard:  {Fore.YELLOW}streamlit run app.py")
        print(f"  Open current map:   {Fore.YELLOW}kothrud_current_traffic.html")
        print(f"  Open forecast map:  {Fore.YELLOW}kothrud_predicted_traffic.html")
        print()

    if args.dashboard:
        print_header("Launching Streamlit Dashboard")
        print("  Starting dashboard at http://localhost:8501")
        print("  Press Ctrl+C to stop")
        print()
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])


if __name__ == "__main__":
    main()
