"""
DGS (Deep Genomic Sequence Analysis Toolkit) - Main Module

This module serves as the main entry point for the DGS command line interface. It provides a comprehensive
set of tools for genomic sequence analysis using deep learning approaches.

Key Features:
    - Model Training: Train deep learning models on genomic sequences
    - Model Evaluation: Assess model performance with various metrics
    - Model Explanation: Generate interpretable explanations for model predictions
    - Sequence Prediction: Make predictions on new genomic sequences
    - Pipeline Execution: Run complete analysis pipelines with multiple modes

Command Line Interface:
    train    - Train a new model or continue training an existing one
    evaluate - Evaluate model performance on test data
    explain  - Generate model explanations and visualizations
    predict  - Make predictions on new sequences
    run      - Execute a complete analysis pipeline
    config   - Generate or manage configuration files

For detailed usage examples, use the --help flag with any command.
"""

import sys
import logging
import argparse
from pathlib import Path
from typing import Optional, Dict, Any

from . import setup_environment, __version__
from .Config import ConfigManager, DgsConfig, ConfigError
from .Cli import DgsCLI

def create_parser() -> argparse.ArgumentParser:
    """
    Create and configure the main argument parser for the DGS command line interface.
    
    Returns:
        argparse.ArgumentParser: Configured argument parser with the following features:
            - Global arguments for verbosity, GPU selection, and random seed
            - Subcommands for different operational modes (train, evaluate, etc.)
            - Configuration file management
            - Example generation capabilities
    
    Example Usage:
        parser = create_parser()
        args = parser.parse_args()
    """
    parser = argparse.ArgumentParser(
        description="DGS: Deep Learning Toolkit for Genomics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Examples:
            # generate example config file
            dgs config --example minimal --output config.json

            # Run with configuration file
            dgs run --config config.json
            
            # Train a model
            dgs train --config config.json
            
            # Evaluate a model
            dgs evaluate --config config.json
            
            # Generate model explanations
            dgs explain --config config.json
            
            # Make predictions
            dgs predict --config config.json
        """
    )
    
    # Essential global arguments
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--verbose", type=int, default=1, choices=[0,1,2],
                      help="Verbosity level (0=warning, 1=info, 2=debug)")
    parser.add_argument("--gpu", type=int, default=0,
                      help="GPU device ID (-1 for CPU)")
    parser.add_argument("--seed", type=int, default=42,
                      help="Random seed")
    parser.add_argument("--benchmark", action="store_true", default=True,
                      help="Enable PyTorch benchmark mode")
    
    # Create minimal subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Add config subparser
    config_parser = subparsers.add_parser("config", help="Configuration file utilities")
    config_parser.add_argument("--example", type=str, choices=["minimal", "full"], default="minimal",
                             help="Generate example configuration file")
    config_parser.add_argument("--output", type=str, default="config.json",
                             help="Output path for example config (default: config.json)")
    
    # Add subparsers with only config argument
    for cmd in ["run", "train", "evaluate", "explain", "predict"]:
        sub_parser = subparsers.add_parser(cmd, help=f"{cmd.capitalize()} mode")
        sub_parser.add_argument("--config", type=str, required=True,
                             help="Path to configuration file")
    
    return parser

def main():
    """
    Main entry point for the DGS command line interface.
    
    This function:
    1. Parses command line arguments
    2. Handles configuration file generation and loading
    3. Sets up the execution environment (GPU/CPU, logging, random seed)
    4. Initializes and executes the requested command
    5. Provides error handling and user feedback
    
    The function will exit with status code 1 if any errors occur during execution,
    such as configuration errors or runtime exceptions.
    """
    # Parse arguments
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.command == "config":
        config_manager = ConfigManager()
        config_manager.generate_example_config(args.example, args.output)
        sys.exit(0)
    
    try:
        # Load configuration
        config_manager = ConfigManager()
        config = config_manager.load_config(args.config)
        
        # Add command line arguments to config
        config["verbose"] = args.verbose
        config["device"] = "cuda" if args.gpu >= 0 else "cpu"
        config["gpu"] = args.gpu
        config["seed"] = args.seed
        config["benchmark"] = args.benchmark
        
        # Setup environment
        device, logger = setup_environment(
            config.get("output_dir", "outputs"),
            config["verbose"],
            config["seed"],
            config.get("benchmark", True),
            config["gpu"]
        )
        
        logger.info(f"Config: {config}")
        
        # Initialize and execute CLI
        try:
            cli = DgsCLI(config=config, device=device)
            cli.execute()
            logger.info(f"Successfully completed {args.command} command!")
            
        except Exception as e:
            logger.error(f"Execution error: {str(e)}")
            raise
            
    except ConfigError as e:
        print(f"Configuration error: {str(e)}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()