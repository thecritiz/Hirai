"""
Tool handlers for Claude Haiku tool use
"""

import json
import logging
from typing import Dict, List, Any

from hiring_system.utils.models import Candidate
from hiring_system.agents.graph_reranker_agent import GraphRerankerAgent
from hiring_system.agents.diversity_agent import DiversityAgent
from hiring_system.utils.tool_specs import TOOL_SPECS

logger = logging.getLogger(__name__)


class ToolHandlers:
    """
    Handlers for Claude tool use calls
    """
    
    def __init__(self, graph_agent: GraphRerankerAgent, diversity_agent: DiversityAgent, 
                candidates_map: Dict[str, Candidate], current_team: List[Candidate]):
        """
        Initialize the tool handlers
        
        Args:
            graph_agent: GraphRerankerAgent instance
            diversity_agent: DiversityAgent instance
            candidates_map: Dictionary mapping candidate IDs to candidates
            current_team: Current team members
        """
        self.graph_agent = graph_agent
        self.diversity_agent = diversity_agent
        self.candidates_map = candidates_map
        self.current_team = current_team
        
    def get_graph_score(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle Graph_Score_Tool calls
        
        Args:
            input_data: Tool input with candidate_id and role
            
        Returns:
            Graph score results
        """
        candidate_id = input_data.get("candidate_id")
        role = input_data.get("role")
        
        logger.info(f"Graph_Score_Tool called for {candidate_id} and role {role}")
        
        if not candidate_id or not role:
            return {
                "error": True,
                "message": "Missing required input: candidate_id and role are required"
            }
        
        try:
            score = self.graph_agent.get_graph_score(candidate_id, role)
            
            # Get candidate details
            candidate = self.candidates_map.get(candidate_id)
            
            if not candidate:
                return {
                    "error": True,
                    "message": f"Candidate with ID {candidate_id} not found"
                }
                
            # Get top matched skills for role
            matched_skills = []
            candidate_skills = set(candidate.metadata.skills)
            
            # Look up role skills in settings
            from hiring_system.config.settings import ROLE_DEFINITIONS
            if role in ROLE_DEFINITIONS:
                role_skills = set(ROLE_DEFINITIONS[role].get("required_skills", []))
                matched_skills = list(candidate_skills.intersection(role_skills))
            
            result = {
                "candidate_id": candidate_id,
                "role": role,
                "graph_score": score,
                "score_normalized": min(score * 100, 100),  # Convert to 0-100 scale
                "matched_skills": matched_skills,
                "skill_match_percentage": len(matched_skills) / max(1, len(ROLE_DEFINITIONS.get(role, {}).get("required_skills", []))) * 100 if role in ROLE_DEFINITIONS else 0
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error in graph score tool: {e}")
            return {
                "error": True,
                "message": f"Error calculating graph score: {str(e)}"
            }
    
    def get_skill_gap(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle Skill_Gap_Tool calls
        
        Args:
            input_data: Tool input with candidate_id and team_skills
            
        Returns:
            Skill gap results
        """
        candidate_id = input_data.get("candidate_id")
        team_skills = input_data.get("team_skills", [])
        
        logger.info(f"Skill_Gap_Tool called for {candidate_id}")
        
        if not candidate_id:
            return {
                "error": True,
                "message": "Missing required input: candidate_id is required"
            }
        
        try:
            candidate = self.candidates_map.get(candidate_id)
            
            if not candidate:
                return {
                    "error": True,
                    "message": f"Candidate with ID {candidate_id} not found"
                }
                
            candidate_skills = candidate.metadata.skills
            
            return self.diversity_agent.calculate_skill_gap(candidate_id, candidate_skills, team_skills)
            
        except Exception as e:
            logger.error(f"Error in skill gap tool: {e}")
            return {
                "error": True,
                "message": f"Error calculating skill gap: {str(e)}"
            }
    
    def get_diversity_contribution(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle Diversity_Tool calls
        
        Args:
            input_data: Tool input with candidate_id, team_locations, team_education_backgrounds
            
        Returns:
            Diversity contribution results
        """
        candidate_id = input_data.get("candidate_id")
        team_locations = input_data.get("team_locations", [])
        team_education_backgrounds = input_data.get("team_education_backgrounds", [])
        
        logger.info(f"Diversity_Tool called for {candidate_id}")
        
        if not candidate_id:
            return {
                "error": True,
                "message": "Missing required input: candidate_id is required"
            }
        
        try:
            candidate = self.candidates_map.get(candidate_id)
            
            if not candidate:
                return {
                    "error": True,
                    "message": f"Candidate with ID {candidate_id} not found"
                }
                
            candidate_location = candidate.metadata.location
            candidate_education = [s for s in candidate.metadata.education.subjects if s]
            
            return self.diversity_agent.calculate_diversity_contribution(
                candidate_id, candidate_location, candidate_education, 
                team_locations, team_education_backgrounds
            )
            
        except Exception as e:
            logger.error(f"Error in diversity tool: {e}")
            return {
                "error": True,
                "message": f"Error calculating diversity contribution: {str(e)}"
            }
    
    def get_education_score(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle Education_Score_Tool calls
        
        Args:
            input_data: Tool input with candidate_id
            
        Returns:
            Education score results
        """
        candidate_id = input_data.get("candidate_id")
        
        logger.info(f"Education_Score_Tool called for {candidate_id}")
        
        if not candidate_id:
            return {
                "error": True,
                "message": "Missing required input: candidate_id is required"
            }
        
        try:
            candidate = self.candidates_map.get(candidate_id)
            
            if not candidate:
                return {
                    "error": True,
                    "message": f"Candidate with ID {candidate_id} not found"
                }
                
            education = candidate.metadata.education
            score = education.quality_score
            
            result = {
                "candidate_id": candidate_id,
                "education_score": score,
                "score_normalized": min(score * 20, 100),  # Convert to 0-100 scale
                "highest_level": education.highest_level,
                "is_top_school": education.is_top25 or education.is_top50,
                "is_top25": education.is_top25,
                "is_top50": education.is_top50,
                "schools": education.schools,
                "subjects": education.subjects
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error in education score tool: {e}")
            return {
                "error": True,
                "message": f"Error calculating education score: {str(e)}"
            }
    
    def get_experience_score(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle Experience_Score_Tool calls
        
        Args:
            input_data: Tool input with candidate_id and role
            
        Returns:
            Experience score results
        """
        candidate_id = input_data.get("candidate_id")
        role = input_data.get("role")
        
        logger.info(f"Experience_Score_Tool called for {candidate_id} and role {role}")
        
        if not candidate_id or not role:
            return {
                "error": True,
                "message": "Missing required input: candidate_id and role are required"
            }
        
        try:
            candidate = self.candidates_map.get(candidate_id)
            
            if not candidate:
                return {
                    "error": True,
                    "message": f"Candidate with ID {candidate_id} not found"
                }
                
            experience = candidate.metadata.experience
            
            # Calculate relevance of experience to role
            role_relevance = 0.0
            
            # Check for role-specific experience
            role_categories = experience.role_categories
            
            from hiring_system.config.settings import ROLE_DEFINITIONS
            role_def = ROLE_DEFINITIONS.get(role, {})
            
            if role == "Engineering Lead" or role == "Senior Developer":
                role_relevance += min(role_categories.get('engineering', 0) * 0.3, 3.0)
                role_relevance += min(role_categories.get('management', 0) * 0.2, 2.0) if role == "Engineering Lead" else 0
            elif role == "Product Manager":
                role_relevance += min(role_categories.get('product', 0) * 0.3, 3.0)
                role_relevance += min(role_categories.get('management', 0) * 0.2, 2.0)
            elif role == "Marketing/Growth Lead":
                role_relevance += min(role_categories.get('marketing', 0) * 0.3, 3.0)
                role_relevance += min(role_categories.get('management', 0) * 0.2, 2.0)
            elif role == "Finance/Operations":
                role_relevance += min(role_categories.get('finance', 0) * 0.3, 3.0)
                role_relevance += min(role_categories.get('operations', 0) * 0.2, 2.0)
            
            # Years of experience score (max 10 years = 5 points)
            years_score = min(experience.years / 2, 5.0)
            
            # Combined score
            total_score = role_relevance + years_score
            
            result = {
                "candidate_id": candidate_id,
                "role": role,
                "experience_score": total_score,
                "score_normalized": min(total_score * 10, 100),  # Convert to 0-100 scale
                "years_experience": experience.years,
                "role_relevance": role_relevance,
                "relevant_roles": [r for r in experience.roles if any(term.lower() in r.lower() for term in role_def.get("required_skills", []))]
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error in experience score tool: {e}")
            return {
                "error": True,
                "message": f"Error calculating experience score: {str(e)}"
            }
    
    def get_candidate_info(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle Candidate_Info_Tool calls
        
        Args:
            input_data: Tool input with candidate_id and include_full_profile
            
        Returns:
            Candidate information
        """
        candidate_id = input_data.get("candidate_id")
        include_full_profile = input_data.get("include_full_profile", False)
        
        logger.info(f"Candidate_Info_Tool called for {candidate_id}")
        
        if not candidate_id:
            return {
                "error": True,
                "message": "Missing required input: candidate_id is required"
            }
        
        try:
            candidate = self.candidates_map.get(candidate_id)
            
            if not candidate:
                return {
                    "error": True,
                    "message": f"Candidate with ID {candidate_id} not found"
                }
                
            metadata = candidate.metadata
            
            # Basic info
            result = {
                "candidate_id": candidate_id,
                "name": metadata.name,
                "location": metadata.location,
                "summary": candidate.summary,
                "best_role": candidate.best_role,
                "best_role_score": candidate.best_role_score,
                "skills": metadata.skills
            }
            
            # Include full profile if requested
            if include_full_profile:
                result["education"] = {
                    "highest_level": metadata.education.highest_level,
                    "schools": metadata.education.schools,
                    "subjects": metadata.education.subjects,
                    "is_top_school": metadata.education.is_top25 or metadata.education.is_top50
                }
                
                result["experience"] = {
                    "years": metadata.experience.years,
                    "roles": metadata.experience.roles,
                    "companies": metadata.experience.companies
                }
                
                result["role_matches"] = candidate.role_matches
            
            return result
            
        except Exception as e:
            logger.error(f"Error in candidate info tool: {e}")
            return {
                "error": True,
                "message": f"Error retrieving candidate info: {str(e)}"
            }
    
    def handle_tool_use(self, tool_use: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle a tool use request from Claude
        
        Args:
            tool_use: Tool use request with name, toolUseId, and input
            
        Returns:
            Tool response
        """
        tool_name = tool_use.get("name")
        tool_use_id = tool_use.get("toolUseId")
        tool_input = tool_use.get("input", {})
        
        logger.info(f"Handling tool use: {tool_name}")
        
        # Route to appropriate handler
        response = None
        
        if tool_name == "Graph_Score_Tool":
            response = self.get_graph_score(tool_input)
        elif tool_name == "Skill_Gap_Tool":
            response = self.get_skill_gap(tool_input)
        elif tool_name == "Diversity_Tool":
            response = self.get_diversity_contribution(tool_input)
        elif tool_name == "Education_Score_Tool":
            response = self.get_education_score(tool_input)
        elif tool_name == "Experience_Score_Tool":
            response = self.get_experience_score(tool_input)
        elif tool_name == "Candidate_Info_Tool":
            response = self.get_candidate_info(tool_input)
        else:
            response = {
                "error": True,
                "message": f"Unknown tool: {tool_name}"
            }
        
        return {
            "toolUseId": tool_use_id,
            "content": response
        }
