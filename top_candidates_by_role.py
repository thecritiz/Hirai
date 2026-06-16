#!/usr/bin/env python3
"""
Script to select the top 3 candidates for each role from the hiring system
"""

import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Any

from hiring_system.pipeline.orchestrator import HiringOrchestrator
from hiring_system.config.settings import ROLE_DEFINITIONS, DEFAULT_DATA_PATH, DEFAULT_OUTPUT_PATH

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
    parser = argparse.ArgumentParser(description="Get top candidates by role")
    
    # Data options
    parser.add_argument("--data", type=str, default=str(DEFAULT_DATA_PATH), 
                        help=f"Path to candidate data file (default: {DEFAULT_DATA_PATH})")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT_PATH),
                        help=f"Path to output file (default: {DEFAULT_OUTPUT_PATH})")
    
    # Options
    parser.add_argument("--top-n", type=int, default=3,
                        help="Number of candidates to select per role (default: 3)")
    parser.add_argument("--aws-region", type=str, default="us-east-1",
                        help="AWS region for Bedrock (default: us-east-1)")
    parser.add_argument("--aws-profile", type=str, default=None,
                        help="AWS profile to use (default: None)")
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
        print("TOP CANDIDATES BY ROLE".center(80))
        print("=" * 80 + "\n")
        
        print(f"Loading data from: {args.data}")
        print(f"Selecting top {args.top_n} candidates per role")
        
        # Create orchestrator
        orchestrator = HiringOrchestrator(
            aws_region=args.aws_region,
            aws_profile=args.aws_profile,
            data_path=args.data,
            output_path=args.output
        )
        
        # Run pipeline preprocessing steps
        orchestrator._load_and_preprocess_data()
        orchestrator._generate_embeddings()
        orchestrator._build_graph_and_rank()
        
        # Get top candidates for each role
        top_candidates_by_role = get_top_candidates(orchestrator, args.top_n)
        
        # Save results
        save_results(top_candidates_by_role, args.output)
        
        # Display results
        display_results(top_candidates_by_role)
        
        return 0
        
    except Exception as e:
        logger.exception(f"Error running hiring system: {e}")
        return 1


def get_top_candidates(orchestrator: HiringOrchestrator, top_n: int) -> Dict[str, List[Dict[str, Any]]]:
    """
    Get top candidates for each role
    
    Args:
        orchestrator: Hiring orchestrator instance
        top_n: Number of candidates to select per role
    
    Returns:
        Dictionary of role -> list of candidates
    """
    logger.info(f"Getting top {top_n} candidates for each role")
    
    top_candidates = {}
    
    for role in ROLE_DEFINITIONS.keys():
        logger.info(f"Processing role: {role}")
        
        # Get ranked candidates for this role
        candidates = orchestrator.graph_agent.rerank_candidates(
            orchestrator.candidates,
            role,
            top_k=top_n
        )
        
        # Process top candidates
        role_candidates = []
        for candidate in candidates[:top_n]:
            # Generate justification
            justification = orchestrator.justifier_agent.justify_candidate(candidate, [])
            
            # Create candidate info
            candidate_info = {
                "name": candidate.metadata.name,
                "role": role,
                "score": candidate.role_matches.get(role, 0.0),
                "justification": justification,
                "location": candidate.metadata.location,
                "skills": candidate.metadata.skills,
                "education": {
                    "highest_level": candidate.metadata.education.highest_level,
                    "schools": candidate.metadata.education.schools
                },
                "experience": {
                    "years": candidate.metadata.experience.years,
                    "roles": candidate.metadata.experience.roles[:3] if candidate.metadata.experience.roles else []
                }
            }
            
            role_candidates.append(candidate_info)
        
        top_candidates[role] = role_candidates
    
    return top_candidates


def save_results(results: Dict[str, List[Dict[str, Any]]], output_path: str) -> None:
    """
    Save results to file
    
    Args:
        results: Dictionary of role -> list of candidates
        output_path: Path to output file
    """
    logger.info(f"Saving results to {output_path}")
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {output_path}")


def display_results(results: Dict[str, List[Dict[str, Any]]]) -> None:
    """
    Display results
    
    Args:
        results: Dictionary of role -> list of candidates
    """
    print("\nTOP CANDIDATES BY ROLE:\n")
    
    for role, candidates in results.items():
        print(f"\n{role.upper()}")
        print("-" * 80)
        
        for i, candidate in enumerate(candidates, 1):
            name = candidate["name"]
            score = candidate["score"]
            location = candidate["location"]
            skills = ", ".join(candidate["skills"]) if candidate["skills"] else "No listed skills"
            justification = candidate["justification"]
            
            print(f"{i}. {name} (Score: {score:.4f})")
            print(f"   Location: {location}")
            print(f"   Skills: {skills}")
            print(f"   Justification: {justification}")
            print()
    
    print("-" * 80)
    print(f"\nTotal candidates selected: {sum(len(candidates) for candidates in results.values())}")


if __name__ == "__main__":
    main()
