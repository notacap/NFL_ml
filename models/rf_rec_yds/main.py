"""
Main entry point for RF Receiving Yards prediction model.

This CLI provides a unified interface for all model operations:
- Feature engineering
- Model training
- Prediction generation
- Model evaluation
"""

import argparse
import sys
from pathlib import Path


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="RF Receiving Yards Prediction Model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Build features
  python main.py build-features --config configs/feature_config.yaml

  # Train model
  python main.py train --config configs/model_config.yaml

  # Generate predictions
  python main.py predict --week 1 --year 2025

  # Evaluate model
  python main.py evaluate --model models/saved/production/latest.pkl
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Build features command
    build_parser = subparsers.add_parser("build-features", help="Build feature sets")
    build_parser.add_argument(
        "--config",
        type=str,
        default="configs/feature_config.yaml",
        help="Path to feature configuration file"
    )
    build_parser.add_argument(
        "--version",
        type=str,
        help="Feature set version (e.g., v1, v2)"
    )

    # Train model command
    train_parser = subparsers.add_parser("train", help="Train model")
    train_parser.add_argument(
        "--config",
        type=str,
        default="configs/model_config.yaml",
        help="Path to model configuration file"
    )
    train_parser.add_argument(
        "--experiment",
        type=str,
        help="Experiment name for tracking"
    )

    # Predict command
    predict_parser = subparsers.add_parser("predict", help="Generate predictions")
    predict_parser.add_argument(
        "--week",
        type=int,
        required=True,
        help="Week number to predict"
    )
    predict_parser.add_argument(
        "--year",
        type=int,
        default=2025,
        help="Year to predict"
    )
    predict_parser.add_argument(
        "--model",
        type=str,
        default="models/saved/production/latest.pkl",
        help="Path to saved model"
    )

    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate model")
    eval_parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to model to evaluate"
    )
    eval_parser.add_argument(
        "--test-data",
        type=str,
        help="Path to test data (optional)"
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    # Command routing
    if args.command == "build-features":
        print(f"Building features with config: {args.config}")
        print("TODO: Implement feature building pipeline")
        # from features.pipelines.build_features import main as build_features
        # build_features(args.config, args.version)

    elif args.command == "train":
        print(f"Training model with config: {args.config}")
        print("TODO: Implement model training pipeline")
        # from models.training.train_model import main as train_model
        # train_model(args.config, args.experiment)

    elif args.command == "predict":
        print(f"Generating predictions for Week {args.week}, {args.year}")
        print(f"Using model: {args.model}")
        print("TODO: Implement prediction pipeline")
        # from models.inference.predict import main as predict
        # predict(args.week, args.year, args.model)

    elif args.command == "evaluate":
        print(f"Evaluating model: {args.model}")
        print("TODO: Implement evaluation pipeline")
        # from evaluation.testing.evaluate_model import main as evaluate
        # evaluate(args.model, args.test_data)


if __name__ == "__main__":
    main()
