"""CLI entrypoint for DGS.

This module builds the command parser, loads and normalizes configuration, and
dispatches execution to the high-level CLI orchestrator.

Supported commands:
    - `config`: Generate example config files.
    - `run`: Execute configured modes in sequence.
    - `train`: Run training only.
    - `evaluate`: Run evaluation only.
    - `explain`: Run motif/explanation workflow only.
    - `predict`: Run variant effect prediction only.
"""

import sys
import logging
import argparse
from pathlib import Path
from typing import Optional, Dict, Any

from . import setup_environment, __version__
from .Config import ConfigManager, DgsConfig, ConfigError, normalize_config
from .Cli import DgsCLI

def create_parser() -> argparse.ArgumentParser:
    """Create the top-level command parser.

    Returns:
        Configured `argparse.ArgumentParser` with global runtime flags and
        subcommands for each DGS execution mode.
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
    benchmark_group = parser.add_mutually_exclusive_group()
    benchmark_group.add_argument(
        "--benchmark",
        dest="benchmark",
        action="store_true",
        help="Enable PyTorch benchmark mode"
    )
    benchmark_group.add_argument(
        "--no-benchmark",
        dest="benchmark",
        action="store_false",
        help="Disable PyTorch benchmark mode"
    )
    parser.set_defaults(benchmark=True)
    
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
    """Run the DGS command-line program.

    Execution flow:
        1. Parse command-line arguments.
        2. Optionally generate example config files.
        3. Load configuration and apply compatibility normalization.
        4. Merge runtime overrides from command-line flags.
        5. Set up environment and dispatch to `DgsCLI`.

    Raises:
        SystemExit: Exits with non-zero status on configuration/runtime errors.

    Side effects:
        Creates output/log directories, writes logs, and may trigger model
        training/evaluation/explanation/prediction workflows depending on mode.
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
        config, inline_compat_notes = normalize_config(config)
        
        # Add command line arguments to config
        config["verbose"] = args.verbose
        config["device"] = "cuda" if args.gpu >= 0 else "cpu"
        config["gpu"] = args.gpu
        config["seed"] = args.seed
        config["benchmark"] = args.benchmark
        if args.command in {"train", "evaluate", "explain", "predict"}:
            # Single-mode commands should override config modes.
            config["modes"] = [args.command]
        
        # Setup environment
        device, logger = setup_environment(
            config.get("output_dir", "outputs"),
            config["verbose"],
            config["seed"],
            config.get("benchmark", True),
            config["gpu"]
        )

        compat_notes = []
        compat_notes.extend(config_manager.get_compat_notes())
        compat_notes.extend(inline_compat_notes)
        for note in compat_notes:
            logger.info(note)
        
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
