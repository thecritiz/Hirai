#!/usr/bin/env python3
"""
100B Jobs - AI-powered Hiring System
Main entry point for the application
"""

import os
import sys
import logging
import argparse
from pathlib import Path

from hiring_system.pipeline.orchestrator import HiringOrchestrator, HiringCrewOrchestrator
from hiring_system.config.settings import DEFAULT_DATA_PATH, DEFAULT_OUTPUT_PATH, SHORTLIST_SIZE

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("hiring.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="100B Jobs - AI-powered Hiring System")
    
    # Data options
    parser.add_argument("--data", type=str, default=str(DEFAULT_DATA_PATH), 
                        help=f"Path to candidate data file (default: {DEFAULT_DATA_PATH})")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT_PATH),
                        help=f"Path to output file (default: {DEFAULT_OUTPUT_PATH})")
    
    # Team selection options
    parser.add_argument("--team-size", type=int, default=3,
                        help="Size of the team to select (default: 3)")
    parser.add_argument("--shortlist", type=int, default=SHORTLIST_SIZE,
                        help=f"Size of the candidate shortlist (default: {SHORTLIST_SIZE})")
    
    # AWS options
    parser.add_argument("--aws-region", type=str, default="us-east-1",
                        help="AWS region for Bedrock (default: us-east-1)")
    parser.add_argument("--aws-profile", type=str, default=None,
                        help="AWS profile to use (default: None)")
    
    # Execution options
    parser.add_argument("--use-crew", action="store_true",
                        help="Use CrewAI for orchestration")
    parser.add_argument("--use-tool-use", action="store_true",
                        help="Use Claude's tool use for final selection")
    parser.add_argument("--save-only", action="store_true",
                        help="Save results without displaying")
    parser.add_argument("--top-candidates", action="store_true",
                        help="Get top 3 candidates for each role")
    parser.add_argument("--top-n", type=int, default=3,
                        help="Number of candidates to select per role (default: 3)")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose output")
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Check if data file exists
    if not Path(args.data).exists():
        logger.error(f"Data file not found: {args.data}")
        return 1
    
    try:
        # Print banner
        print("\n" + "=" * 80)
        print("100B JOBS: AI-POWERED HIRING SYSTEM".center(80))
        print("=" * 80 + "\n")
        
        print(f"Loading data from: {args.data}")
        print(f"Team size: {args.team_size}")
        print(f"Shortlist size: {args.shortlist}")
        
        # Create orchestrator
        if args.use_crew:
            print("\nUsing CrewAI orchestration")
            orchestrator = HiringCrewOrchestrator(
                aws_region=args.aws_region,
                aws_profile=args.aws_profile,
                team_size=args.team_size,
                shortlist_size=args.shortlist,
                data_path=args.data,
                output_path=args.output
            )
        else:
            print("\nUsing standard orchestration")
            orchestrator = HiringOrchestrator(
                aws_region=args.aws_region,
                aws_profile=args.aws_profile,
                team_size=args.team_size,
                shortlist_size=args.shortlist,
                data_path=args.data,
                output_path=args.output
            )
        
        # Run the pipeline
        if args.top_candidates:
            print("\nGetting top candidates for each role")
            top_candidates = orchestrator._load_and_preprocess_data()
            orchestrator._generate_embeddings()
            orchestrator._build_graph_and_rank()
            orchestrator.get_top_candidates_by_role(count=args.top_n)
            
            # Save results
            orchestrator.save_results(args.output)
            
            # Display results
            if not args.save_only:
                orchestrator.display_team()
                
            print(f"\nResults saved to {args.output}")
            
        elif args.use_tool_use:
            print("\nRunning with Claude's tool use for final selection")
            results = orchestrator.run_with_tool_use(shortlist_size=args.shortlist)
            
            # Display Claude's response
            print("\n" + "=" * 80)
            print("CLAUDE'S TEAM SELECTION".center(80))
            print("=" * 80 + "\n")
            print(results.get("claude_response", "No response from Claude"))
            
            # Save results
            with open(args.output, 'w') as f:
                f.write(results.get("claude_response", "No response from Claude"))
                
            print(f"\nResults saved to {args.output}")
            
        else:
            print("\nRunning standard pipeline")
            team = orchestrator.run()
            
            # Save results
            orchestrator.save_results(args.output)
            
            # Display results
            if not args.save_only:
                orchestrator.display_team()
                
            print(f"\nResults saved to {args.output}")
        
        return 0
        
    except Exception as e:
        logger.exception(f"Error running hiring system: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
