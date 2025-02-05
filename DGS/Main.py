"""
This module provides Command line interface for DGS (Deep Genomic Sequence Analysis Toolkit)

The main entry point for the DGS command line interface. It provides the following commands:
- train: Train a model
- evaluate: Evaluate a trained model
- explain: Generate model explanations
- predict: Make predictions on new data
- run: Run a complete pipeline with multiple modes
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
    """Create the main argument parser with only essential arguments."""
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
    """Main entry point for DGS CLI."""
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