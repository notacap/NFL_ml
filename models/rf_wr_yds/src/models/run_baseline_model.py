"""
Execution script for the RF baseline model.

This script runs the complete training pipeline for the Random Forest
baseline model for WR receiving yards prediction.

Usage:
    python run_baseline_model.py

Output:
    - Trained model saved to models/rf_baseline_v1.joblib
    - Model config saved to models/model_config.yaml
    - Metrics report saved to outputs/model_evaluation/metrics_report.txt
    - Visualizations saved to outputs/model_evaluation/
"""

import sys
import os
from pathlib import Path

# Add the parent directory to path for imports
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent
sys.path.insert(0, str(project_root))

from src.models.train_rf_baseline import main


def run():
    """Execute the baseline model training pipeline."""
    print("\n" + "=" * 70)
    print("EXECUTING RF BASELINE MODEL TRAINING")
    print("=" * 70 + "\n")

    # Run training pipeline
    results = main()

    print("\n" + "=" * 70)
    print("EXECUTION SUMMARY")
    print("=" * 70)
    print(f"\nModel Performance (Test Set - 2024 Season):")
    print(f"  - MAE:  {results['metrics']['MAE']:.2f} yards")
    print(f"  - RMSE: {results['metrics']['RMSE']:.2f} yards")
    print(f"  - R2:   {results['metrics']['R2']:.4f}")
    print(f"  - MAPE: {results['metrics']['MAPE']:.2f}%")

    print(f"\nArtifacts saved:")
    for name, path in results['artifact_paths'].items():
        print(f"  - {name}: {path}")

    print(f"\nVisualization files:")
    output_dir = os.path.join(project_root, "outputs", "model_evaluation")
    print(f"  - {os.path.join(output_dir, 'feature_importance.png')}")
    print(f"  - {os.path.join(output_dir, 'pred_vs_actual.png')}")
    print(f"  - {os.path.join(output_dir, 'residuals.png')}")

    print("\n" + "=" * 70)
    print("Top 5 Most Important Features:")
    print("=" * 70)
    for idx, row in results['feature_importance'].head(5).iterrows():
        print(f"  {row['feature']:30s}: {row['importance']:.4f}")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70 + "\n")

    return results


if __name__ == "__main__":
    run()
