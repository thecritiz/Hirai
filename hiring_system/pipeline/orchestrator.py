"""
Orchestrator for the hiring system using LangChain and LangGraph
"""

import os
import json
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set
from crewai import Agent, Task, Crew
from langchain.tools import Tool
from langchain.agents import AgentExecutor
import boto3

from hiring_system.utils.models import Candidate, TeamMember, Team
from hiring_system.agents.preprocessing_agent import PreprocessingAgent
from hiring_system.agents.embedding_agent import EmbeddingAgent
from hiring_system.agents.graph_reranker_agent import GraphRerankerAgent
from hiring_system.agents.diversity_agent import DiversityAgent
from hiring_system.agents.justifier_agent import JustifierAgent
from hiring_system.utils.bedrock_client import BedrockClient
from hiring_system.utils.tool_specs import get_tool_specs
from hiring_system.utils.tool_handlers import ToolHandlers

from hiring_system.config.settings import (
    ROLE_DEFINITIONS,
    TEAM_SIZE,
    SHORTLIST_SIZE,
    DEFAULT_DATA_PATH,
    DEFAULT_OUTPUT_PATH
)

logger = logging.getLogger(__name__)


class HiringOrchestrator:
    """
    Orchestrator for the hiring system
    
    Coordinates the workflow between all agents using LangChain and LangGraph
    """
    
    def __init__(self, 
                aws_region: str = "us-east-1",
                aws_profile: Optional[str] = None,
                team_size: int = TEAM_SIZE,
                shortlist_size: int = SHORTLIST_SIZE,
                data_path: str = DEFAULT_DATA_PATH,
                output_path: str = DEFAULT_OUTPUT_PATH):
        """
        Initialize the hiring orchestrator
        
        Args:
            aws_region: AWS region for Bedrock
            aws_profile: Optional AWS profile to use
            team_size: Size of the team to select
            shortlist_size: Size of the candidate shortlist
            data_path: Path to the data file
            output_path: Path to the output file
        """
        logger.info("Initializing HiringOrchestrator")
        
        # Store configuration
        self.aws_region = aws_region
        self.aws_profile = aws_profile
        self.team_size = team_size
        self.shortlist_size = shortlist_size
        self.data_path = data_path
        self.output_path = output_path
        
        # Initialize agents
        self.preprocessing_agent = PreprocessingAgent()
        self.embedding_agent = EmbeddingAgent()
        self.graph_agent = GraphRerankerAgent()
        self.diversity_agent = DiversityAgent()
        self.justifier_agent = JustifierAgent(aws_region=aws_region, aws_profile=aws_profile)
        
        # Initialize Claude Haiku client
        self.bedrock_client = BedrockClient(region_name=aws_region, profile_name=aws_profile)
        
        # Storage for intermediate results
        self.candidates = []
        self.candidates_map = {}
        self.shortlisted_candidates = []
        self.selected_team = []
        self.final_team = None
        self.top_role_candidates = {}
        
    def run(self) -> Team:
        """
        Run the full hiring pipeline
        
        Returns:
            Final selected team
        """
        logger.info("Running hiring pipeline")
        
        # Step 1: Load and preprocess data
        self._load_and_preprocess_data()
        
        # Step 2: Generate embeddings
        self._generate_embeddings()
        
        # Step 3: Build graph and rank candidates
        self._build_graph_and_rank()
        
        # Step 4: Select diverse team
        self._select_team()
        
        # Step 5: Generate justifications
        self._generate_justifications()
        
        return self.final_team
        
    def _load_and_preprocess_data(self) -> None:
        """
        Load and preprocess the data
        """
        logger.info(f"Loading and preprocessing data from {self.data_path}")
        
        # Determine file type
        file_extension = Path(self.data_path).suffix.lower()
        
        # Process based on file type
        if file_extension == '.json':
            self.candidates = self.preprocessing_agent.process_json_file(self.data_path)
        elif file_extension == '.csv':
            self.candidates = self.preprocessing_agent.process_csv_file(self.data_path)
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")
            
        # Enrich candidates
        self.candidates = self.preprocessing_agent.enrich_candidates(self.candidates)
        
        # Create candidate map
        self.candidates_map = {c.id: c for c in self.candidates}
        
        logger.info(f"Processed {len(self.candidates)} candidates")
        
    def _generate_embeddings(self) -> None:
        """
        Generate embeddings for candidates
        """
        logger.info("Generating embeddings")
        
        # Generate embeddings
        self.candidates = self.embedding_agent.generate_embeddings(self.candidates)
        
        # Build FAISS index
        self.embedding_agent.build_faiss_index(self.candidates)
        
        logger.info("Embeddings generated")
        
    def _build_graph_and_rank(self) -> None:
        """
        Build graph and rank candidates
        """
        logger.info("Building graph and ranking candidates")
        
        # Build graph
        self.graph_agent.build_graph(self.candidates)
        
        # Rank candidates for each role
        self.shortlisted_candidates = []
        role_candidates = {}
        
        for role in ROLE_DEFINITIONS.keys():
            # Get reranked candidates for this role
            reranked = self.graph_agent.rerank_candidates(
                self.candidates, 
                role, 
                top_k=self.shortlist_size
            )
            
            role_candidates[role] = reranked[:self.shortlist_size]
            self.shortlisted_candidates.extend(reranked[:self.shortlist_size])
        
        # Remove duplicates
        seen_ids = set()
        unique_shortlist = []
        
        for candidate in self.shortlisted_candidates:
            if candidate.id not in seen_ids:
                seen_ids.add(candidate.id)
                unique_shortlist.append(candidate)
        
        self.shortlisted_candidates = unique_shortlist[:self.shortlist_size]
        
        logger.info(f"Shortlisted {len(self.shortlisted_candidates)} candidates")
        
    def _select_team(self) -> None:
        """
        Select the final team
        """
        logger.info(f"Selecting team of {self.team_size}")
        
        # Select team
        self.selected_team = self.diversity_agent.select_team(
            self.shortlisted_candidates,
            self.team_size,
            role_preference=True
        )
        
        logger.info(f"Selected {len(self.selected_team)} team members")
        
    def _generate_justifications(self) -> None:
        """
        Generate justifications for selected team
        """
        logger.info("Generating justifications")
        
        # Create Team object
        team = self.diversity_agent.create_team_object(self.selected_team)
        
        # Generate justifications
        team = self.justifier_agent.justify_team(team)
        
        # Store final team
        self.final_team = team
        
        logger.info("Justifications generated")
        
    def save_results(self, output_path: Optional[str] = None) -> None:
        """
        Save results to file
        
        Args:
            output_path: Optional path to output file
        """
        path = output_path or self.output_path
        logger.info(f"Saving results to {path}")
        
        # Create results dictionary
        if hasattr(self, 'final_team') and self.final_team is not None:
            results = {
                "team_size": len(self.final_team.members),
                "diversity_metrics": self.final_team.diversity_metrics,
                "total_score": self.final_team.total_score,
                "members": []
            }
            
            # Add team members
            for member in self.final_team.members:
                results["members"].append({
                    "name": member.candidate.metadata.name,
                    "role": member.role,
                    "score": member.score,
                    "justification": member.justification,
                    "location": member.candidate.metadata.location,
                    "skills": member.candidate.metadata.skills,
                    "education": {
                        "highest_level": member.candidate.metadata.education.highest_level,
                        "schools": member.candidate.metadata.education.schools
                    },
                    "experience": {
                        "years": member.candidate.metadata.experience.years,
                        "roles": member.candidate.metadata.experience.roles[:3]
                    }
                })
        elif self.top_role_candidates:
            results = self.top_role_candidates
        else:
            results = {"error": "No results available"}
        
        # Save as JSON
        with open(path, 'w') as f:
            json.dump(results, f, indent=2)
            
        logger.info(f"Results saved to {path}")
        
    def get_top_candidates_by_role(self, count: int = 3) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get top candidates for each role
        
        Args:
            count: Number of candidates to select per role
            
        Returns:
            Dictionary of role -> list of candidates
        """
        logger.info(f"Getting top {count} candidates for each role")
        
        # Ensure we have candidates with role matches
        if not self.candidates:
            logger.warning("No candidates available")
            return {}
            
        if not any('role_matches' in vars(c) for c in self.candidates):
            self._build_graph_and_rank()
        
        top_candidates = {}
        
        # Get top candidates for each role
        for role in ROLE_DEFINITIONS.keys():
            logger.info(f"Processing role: {role}")
            role_candidates = self.justifier_agent.get_role_candidates(
                self.candidates,
                role,
                count=count
            )
            top_candidates[role] = role_candidates
        
        # Store results
        self.top_role_candidates = top_candidates
        
        return top_candidates
        
    def display_team(self) -> None:
        """
        Display the selected team
        """
        if self.top_role_candidates:
            # Display top candidates by role
            self._display_top_candidates()
            return
        
        if not self.final_team:
            logger.warning("No team selected yet")
            return
            
        # Display regular team
        print("\n" + "=" * 80)
        print("SELECTED TEAM".center(80))
        print("=" * 80 + "\n")
        
        for i, member in enumerate(self.final_team.members, 1):
            candidate = member.candidate
            name = candidate.metadata.name
            role = member.role
            location = candidate.metadata.location
            education = candidate.metadata.education
            schools = ", ".join(education.schools) if education.schools else "Unknown"
            skills = ", ".join(candidate.metadata.skills) if candidate.metadata.skills else "No listed skills"
            
            print(f"{i}. {name} - {role} (Score: {member.score:.3f})")
            print(f"   Location: {location}")
            print(f"   Education: {education.highest_level} from {schools}")
            print(f"   Skills: {skills}")
            print(f"   Justification: {member.justification}")
            print("-" * 80)
            
        print("\nTeam Diversity Metrics:")
        for metric, value in self.final_team.diversity_metrics.items():
            print(f"- {metric}: {value:.2f}" if isinstance(value, float) else f"- {metric}: {value}")
            
    def _display_top_candidates(self) -> None:
        """
        Display top candidates for each role
        """
        print("\n" + "=" * 80)
        print("TOP CANDIDATES BY ROLE".center(80))
        print("=" * 80 + "\n")
        
        total_candidates = 0
        
        for role, candidates in self.top_role_candidates.items():
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
            
            total_candidates += len(candidates)
        
        print("-" * 80)
        print(f"\nTotal candidates selected: {total_candidates}")
            
    def run_with_tool_use(self, shortlist_size: int = 10) -> Dict[str, Any]:
        """
        Run hiring with Claude's tool use
        
        Args:
            shortlist_size: Number of candidates to include in the shortlist
            
        Returns:
            Final results dictionary
        """
        logger.info("Running hiring with Claude's tool use")
        
        # First run the pipeline to get candidate data and graph prepared
        if not self.candidates:
            self._load_and_preprocess_data()
            
        if not any(c.embedding for c in self.candidates):
            self._generate_embeddings()
            
        if not self.graph_agent.graph:
            self.graph_agent.build_graph(self.candidates)
        
        # Get shortlisted candidates for each role
        role_shortlists = {}
        all_shortlisted = set()
        
        for role in ROLE_DEFINITIONS.keys():
            # Get reranked candidates for this role
            reranked = self.graph_agent.rerank_candidates(
                self.candidates, 
                role, 
                top_k=shortlist_size
            )
            
            role_shortlists[role] = reranked[:shortlist_size]
            all_shortlisted.update(c.id for c in reranked[:shortlist_size])
        
        # Create candidate info for Claude
        candidates_info = []
        for candidate in self.candidates:
            if candidate.id in all_shortlisted:
                metadata = candidate.metadata
                education = metadata.education
                experience = metadata.experience
                
                candidate_info = {
                    "id": candidate.id,
                    "name": metadata.name,
                    "location": metadata.location,
                    "education": {
                        "level": education.highest_level,
                        "schools": education.schools,
                        "is_top_school": education.is_top25 or education.is_top50
                    },
                    "experience": {
                        "years": experience.years,
                        "roles": experience.roles[:3]
                    },
                    "skills": metadata.skills,
                    "role_scores": {role: candidate.role_matches.get(role, 0.0) for role in ROLE_DEFINITIONS.keys()}
                }
                
                candidates_info.append(candidate_info)
        
        # Create tool handlers
        tool_handlers = ToolHandlers(
            graph_agent=self.graph_agent,
            diversity_agent=self.diversity_agent,
            candidates_map=self.candidates_map,
            current_team=[]
        )
        
        # Create system prompt for Claude
        system_prompt = """
        You are a hiring expert for a tech startup that has just raised $100M in funding.
        Your task is to select the best team of 5 candidates for the following roles:
        1. Engineering Lead
        2. Product Manager
        3. Senior Developer
        4. Marketing/Growth Lead
        5. Finance/Operations
        
        You have access to tools that will help you evaluate candidates on various dimensions:
        - Graph_Score_Tool: Returns a candidate's score based on PageRank in a skills-role graph
        - Skill_Gap_Tool: Shows what unique skills a candidate adds to the team
        - Diversity_Tool: Evaluates how much diversity a candidate adds to the team
        - Education_Score_Tool: Returns a candidate's education quality score
        - Experience_Score_Tool: Returns a candidate's experience quality score
        - Candidate_Info_Tool: Gets detailed information about a candidate
        
        Follow these steps:
        1. For each role, identify the top 3-5 candidates using Graph_Score_Tool
        2. For each candidate, use additional tools to evaluate them more deeply
        3. Select the best candidate for each role, considering:
           - Role fit (50%)
           - Skill complementarity (30%)
           - Diversity (20%)
        4. Justify each selection with 1-2 sentences
        
        Your final output should be a JSON with:
        - Selected candidates for each role with their scores
        - Justification for each selection
        - Overall team assessment
        """
        
        # Create prompt for Claude
        prompt = f"""
        Select the best team of 5 candidates for our startup from this pool of {len(candidates_info)} candidates.
        
        Remember to consider:
        1. Role-specific fit
        2. Team skill complementarity
        3. Diversity (geographic, educational background)
        
        I'll need one person for each of these roles:
        - Engineering Lead
        - Product Manager
        - Senior Developer
        - Marketing/Growth Lead
        - Finance/Operations
        
        Please use the tools available to evaluate candidates thoroughly before making your final selections.
        """
        
        # Run Claude with tool use
        try:
            # Prepare for converse API
            session = boto3.Session(profile_name=self.aws_profile) if self.aws_profile else boto3.Session()
            client = session.client('bedrock-runtime', region_name=self.aws_region)
            
            # Set up tool configs
            tool_config = {
                "tools": get_tool_specs()
            }
            
            # Initial message
            messages = [{
                "role": "user",
                "content": [{"text": prompt}]
            }]
            
            # Call converse API
            response = client.converse(
                modelId="anthropic.claude-3-haiku-20240307-v1:0",
                messages=messages,
                system=[{"text": system_prompt}],
                toolConfig=tool_config,
                inferenceConfig={
                    "maxTokens": 4000,
                    "temperature": 0.0,
                    "topP": 0.9
                }
            )
            
            # Handle tool use
            stop_reason = response.get("stopReason")
            claude_response = response.get("output", {}).get("message", {})
            
            # Message history
            message_history = [{"role": "user", "content": [{"text": prompt}]}]
            
            max_turns = 20  # Safety limit
            turn_count = 0
            
            while stop_reason == "tool_use" and turn_count < max_turns:
                turn_count += 1
                logger.info(f"Claude requested tool use (turn {turn_count})")
                
                # Add Claude's message to history
                message_history.append(claude_response)
                
                # Process tool use requests
                tool_results = []
                for content_block in claude_response.get("content", []):
                    if "text" in content_block:
                        logger.info(f"Claude: {content_block['text'][:100]}...")
                        
                    if "toolUse" in content_block:
                        tool_use = content_block["toolUse"]
                        tool_result = tool_handlers.handle_tool_use(tool_use)
                        tool_results.append({"toolResult": tool_result})
                        logger.info(f"Processed tool use: {tool_use['name']}")
                
                # Create user message with tool results
                tool_message = {
                    "role": "user", 
                    "content": tool_results
                }
                
                message_history.append(tool_message)
                
                # Call Claude again
                response = client.converse(
                    modelId="anthropic.claude-3-haiku-20240307-v1:0",
                    messages=message_history,
                    system=[{"text": system_prompt}],
                    toolConfig=tool_config,
                    inferenceConfig={
                        "maxTokens": 4000,
                        "temperature": 0.0,
                        "topP": 0.9
                    }
                )
                
                # Update for next iteration
                stop_reason = response.get("stopReason")
                claude_response = response.get("output", {}).get("message", {})
            
            # Final response
            final_response = ""
            for content_block in claude_response.get("content", []):
                if "text" in content_block:
                    final_response += content_block["text"]
            
            # Parse results
            results = {
                "claude_response": final_response,
                "message_history": message_history
            }
            
            logger.info("Claude tool use completed")
            return results
            
        except Exception as e:
            logger.error(f"Error in Claude tool use: {e}")
            raise


class HiringCrewOrchestrator:
    """
    CrewAI-based orchestrator for the hiring system
    
    Uses CrewAI to coordinate the agents
    """
    
    def __init__(self, 
                aws_region: str = "us-east-1",
                aws_profile: Optional[str] = None,
                team_size: int = TEAM_SIZE,
                shortlist_size: int = SHORTLIST_SIZE,
                data_path: str = DEFAULT_DATA_PATH,
                output_path: str = DEFAULT_OUTPUT_PATH):
        """
        Initialize the hiring orchestrator
        
        Args:
            aws_region: AWS region for Bedrock
            aws_profile: Optional AWS profile to use
            team_size: Size of the team to select
            shortlist_size: Size of the candidate shortlist
            data_path: Path to the data file
            output_path: Path to the output file
        """
        logger.info("Initializing HiringCrewOrchestrator")
        
        # Store configuration
        self.aws_region = aws_region
        self.aws_profile = aws_profile
        self.team_size = team_size
        self.shortlist_size = shortlist_size
        self.data_path = data_path
        self.output_path = output_path
        
        # Initialize agents
        self.preprocessing_agent = PreprocessingAgent()
        self.embedding_agent = EmbeddingAgent()
        self.graph_agent = GraphRerankerAgent()
        self.diversity_agent = DiversityAgent()
        self.justifier_agent = JustifierAgent(aws_region=aws_region, aws_profile=aws_profile)
        
        # Storage for results
        self.candidates = []
        self.enriched_candidates = []
        self.ranked_candidates = []
        self.selected_team = []
        
        # Create CrewAI agents
        self.data_agent = self._create_data_agent()
        self.embedding_crew_agent = self._create_embedding_agent()
        self.ranking_agent = self._create_ranking_agent()
        self.team_selection_agent = self._create_team_selection_agent()
        self.justification_agent = self._create_justification_agent()
        
    def _create_data_agent(self) -> Agent:
        """Create the data preprocessing agent"""
        return Agent(
            role="Data Preprocessing Specialist",
            goal="Extract and structure candidate data from raw JSON/CSV files",
            backstory="You are an expert in data extraction and preprocessing, specializing in candidate data.",
            verbose=True,
            allow_delegation=False,
            tools=[
                Tool.from_function(
                    func=self.preprocessing_agent.process_json_file,
                    name="process_json_file",
                    description="Process a JSON file containing candidate data"
                ),
                Tool.from_function(
                    func=self.preprocessing_agent.process_csv_file,
                    name="process_csv_file",
                    description="Process a CSV file containing candidate data"
                ),
                Tool.from_function(
                    func=self.preprocessing_agent.enrich_candidates,
                    name="enrich_candidates",
                    description="Enrich candidate profiles with summaries and keywords"
                )
            ]
        )
    
    def _create_embedding_agent(self) -> Agent:
        """Create the embedding agent"""
        return Agent(
            role="Embedding and Vector Search Specialist",
            goal="Generate high-quality embeddings and build efficient vector search indexes",
            backstory="You are an expert in NLP and vector embeddings, specializing in semantic search.",
            verbose=True,
            allow_delegation=False,
            tools=[
                Tool.from_function(
                    func=self.embedding_agent.generate_embeddings,
                    name="generate_embeddings",
                    description="Generate embeddings for a list of candidates"
                ),
                Tool.from_function(
                    func=self.embedding_agent.build_faiss_index,
                    name="build_faiss_index",
                    description="Build a FAISS index from candidate embeddings"
                )
            ]
        )
    
    def _create_ranking_agent(self) -> Agent:
        """Create the ranking agent"""
        return Agent(
            role="Graph Algorithm Specialist",
            goal="Build a heterogeneous graph and rank candidates using PageRank",
            backstory="You are an expert in graph algorithms and candidate ranking.",
            verbose=True,
            allow_delegation=False,
            tools=[
                Tool.from_function(
                    func=self.graph_agent.build_graph,
                    name="build_graph",
                    description="Build a heterogeneous graph from candidates"
                ),
                Tool.from_function(
                    func=self.graph_agent.rerank_candidates,
                    name="rerank_candidates",
                    description="Rerank candidates based on PageRank scores for a role"
                )
            ]
        )
    
    def _create_team_selection_agent(self) -> Agent:
        """Create the team selection agent"""
        return Agent(
            role="Team Composition Specialist",
            goal="Select a diverse and complementary team of candidates",
            backstory="You are an expert in team building and diversity optimization.",
            verbose=True,
            allow_delegation=False,
            tools=[
                Tool.from_function(
                    func=self.diversity_agent.select_team,
                    name="select_team",
                    description="Select a diverse team using greedy marginal gain algorithm"
                ),
                Tool.from_function(
                    func=self.diversity_agent.create_team_object,
                    name="create_team_object",
                    description="Create a Team object from selected candidates"
                )
            ]
        )
    
    def _create_justification_agent(self) -> Agent:
        """Create the justification agent"""
        return Agent(
            role="Hiring Justification Specialist",
            goal="Generate compelling justifications for candidate selection",
            backstory="You are an expert in HR and recruitment, specializing in candidate evaluation.",
            verbose=True,
            allow_delegation=False,
            tools=[
                Tool.from_function(
                    func=self.justifier_agent.justify_team,
                    name="justify_team",
                    description="Generate justifications for all team members"
                ),
                Tool.from_function(
                    func=self.justifier_agent.get_final_team_assessment,
                    name="get_final_team_assessment",
                    description="Generate overall assessment of the selected team"
                )
            ]
        )
    
    def run(self) -> Team:
        """
        Run the hiring pipeline using CrewAI
        
        Returns:
            Final selected team
        """
        logger.info("Running CrewAI hiring pipeline")
        
        # Create tasks
        data_task = Task(
            description="Extract and structure candidate data from the raw file",
            expected_output="List of structured candidate objects",
            agent=self.data_agent
        )
        
        embedding_task = Task(
            description="Generate embeddings for all candidates and build a search index",
            expected_output="Candidates with embeddings and a search index",
            agent=self.embedding_crew_agent
        )
        
        ranking_task = Task(
            description="Build a graph and rank candidates for each role",
            expected_output="Ranked candidates for each role",
            agent=self.ranking_agent
        )
        
        team_selection_task = Task(
            description="Select a diverse team of candidates",
            expected_output="Selected team members",
            agent=self.team_selection_agent
        )
        
        justification_task = Task(
            description="Generate justifications for the selected team",
            expected_output="Team with justifications",
            agent=self.justification_agent
        )
        
        # Create crew
        crew = Crew(
            agents=[
                self.data_agent,
                self.embedding_crew_agent,
                self.ranking_agent,
                self.team_selection_agent,
                self.justification_agent
            ],
            tasks=[
                data_task,
                embedding_task,
                ranking_task,
                team_selection_task,
                justification_task
            ],
            verbose=True
        )
        
        # Run crew
        result = crew.kickoff()
        
        # Process result
        # Note: In a real implementation, we would parse the result and extract the team
        
        return self.selected_team
