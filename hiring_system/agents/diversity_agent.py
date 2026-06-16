"""
Diversity Agent for team composition and selection
"""

import logging
from typing import Dict, List, Set, Any, Optional
from collections import Counter

from hiring_system.utils.models import Candidate, TeamMember, Team
from hiring_system.config.settings import DIVERSITY_FACTORS

logger = logging.getLogger(__name__)


class DiversityAgent:
    """
    Agent for ensuring team diversity and optimal team selection
    """
    
    def __init__(self, diversity_weights: Optional[Dict[str, float]] = None):
        """
        Initialize the diversity agent
        
        Args:
            diversity_weights: Optional weights for diversity factors
        """
        logger.info("Initializing DiversityAgent")
        self.diversity_weights = diversity_weights or DIVERSITY_FACTORS
        
    def select_team(self, candidates: List[Candidate], team_size: int, 
                   role_preference: bool = True) -> List[Candidate]:
        """
        Select a diverse team using greedy marginal gain algorithm
        
        Args:
            candidates: List of candidates to select from
            team_size: Number of candidates to select
            role_preference: If True, consider role matches in selection
            
        Returns:
            Selected team as list of candidates
        """
        logger.info(f"Selecting team of {team_size} from {len(candidates)} candidates")
        
        # Start with empty team
        selected_team = []
        selected_ids = set()
        
        # Track diversity dimensions
        team_locations = set()
        team_education = set()
        team_skills = set()
        team_roles = set()
        
        # Sort all candidates by their best role score, regardless of role
        sorted_candidates = sorted(candidates, key=lambda x: x.best_role_score, reverse=True)
        
        # First, select the top candidates based on their best role match score
        top_candidates = sorted_candidates[:team_size]
        
        # For each top candidate
        for candidate in top_candidates:
            # Add to team
            selected_team.append(candidate)
            selected_ids.add(candidate.id)
            
            # Update diversity tracking
            team_locations.add(candidate.metadata.location)
            team_education.update([s for s in candidate.metadata.education.subjects if s])
            team_skills.update(candidate.metadata.skills)
            if candidate.best_role:
                team_roles.add(candidate.best_role)
            
            logger.info(f"Selected {candidate.metadata.name} for role {candidate.best_role}")
        
        logger.info(f"Selected team of {len(selected_team)} members")
        return selected_team
    
    def _calculate_diversity_gain(self, 
                                 candidate: Candidate, 
                                 team_locations: Set[str],
                                 team_education: Set[str],
                                 team_skills: Set[str],
                                 team_roles: Set[str]) -> float:
        """
        Calculate diversity gain from adding a candidate to the team
        
        Args:
            candidate: Candidate to evaluate
            team_locations: Set of current team locations
            team_education: Set of current team education subjects
            team_skills: Set of current team skills
            team_roles: Set of current team roles
            
        Returns:
            Diversity gain score
        """
        score = 0.0
        
        # Location diversity
        location = candidate.metadata.location
        location_weight = self.diversity_weights.get("location", 0.3)
        if location and location not in team_locations:
            score += 10.0 * location_weight
        
        # Education diversity
        education_weight = self.diversity_weights.get("education", 0.25)
        education_subjects = [s for s in candidate.metadata.education.subjects if s]
        new_education_subjects = [s for s in education_subjects if s not in team_education]
        if new_education_subjects:
            score += len(new_education_subjects) * 5.0 * education_weight
        
        # Skill complementarity
        skill_weight = self.diversity_weights.get("skills", 0.2)
        candidate_skills = set(candidate.metadata.skills)
        new_skills = candidate_skills - team_skills
        if new_skills:
            score += len(new_skills) * 3.0 * skill_weight
        
        # Role diversity
        if candidate.best_role and candidate.best_role not in team_roles:
            score += 15.0  # Strong boost for uncovered role
        
        return score
    
    def calculate_skill_gap(self, candidate_id: str, candidate_skills: List[str], team_skills: List[str]) -> Dict[str, Any]:
        """
        Calculate skill gap between a candidate and the team
        
        Args:
            candidate_id: ID of the candidate
            candidate_skills: List of candidate skills
            team_skills: List of all team skills
            
        Returns:
            Dictionary with skill gap metrics
        """
        # Convert to sets for easier operations
        candidate_skill_set = set(candidate_skills)
        team_skill_set = set(team_skills)
        
        # Calculate unique skills the candidate brings
        unique_skills = candidate_skill_set - team_skill_set
        
        # Calculate overlap with team skills
        overlap_skills = candidate_skill_set.intersection(team_skill_set)
        
        result = {
            "candidate_id": candidate_id,
            "unique_skill_count": len(unique_skills),
            "unique_skills": list(unique_skills),
            "overlap_skill_count": len(overlap_skills),
            "overlap_skills": list(overlap_skills),
            "skill_gap_score": len(unique_skills) * 3.0  # Basic scoring - 3 points per unique skill
        }
        
        return result
    
    def calculate_diversity_contribution(self, candidate_id: str, candidate_location: str, 
                                       candidate_education: List[str], team_locations: List[str], 
                                       team_education_backgrounds: List[str]) -> Dict[str, Any]:
        """
        Calculate diversity contribution of a candidate to the team
        
        Args:
            candidate_id: ID of the candidate
            candidate_location: Location of the candidate
            candidate_education: Education backgrounds of the candidate
            team_locations: Locations of current team members
            team_education_backgrounds: Education backgrounds of current team members
            
        Returns:
            Dictionary with diversity metrics
        """
        # Convert to sets
        team_location_set = set(team_locations)
        team_education_set = set(team_education_backgrounds)
        candidate_education_set = set(candidate_education)
        
        # Calculate location diversity
        location_diversity = 0.0
        location_unique = False
        if candidate_location and candidate_location not in team_location_set:
            location_diversity = 10.0
            location_unique = True
        else:
            # Penalize for location overlap
            location_count = team_locations.count(candidate_location) if candidate_location else 0
            if location_count > 0:
                # Diminishing score as more team members are from the same location
                location_diversity = 5.0 / (location_count + 1)
        
        # Calculate education diversity
        education_diversity = 0.0
        unique_education = []
        for edu in candidate_education_set:
            if edu and edu not in team_education_set:
                unique_education.append(edu)
                education_diversity += 5.0
        
        # Overall diversity score with weights
        location_weight = self.diversity_weights.get("location", 0.3)
        education_weight = self.diversity_weights.get("education", 0.25)
        
        overall_diversity_score = (location_diversity * location_weight) + (education_diversity * education_weight)
        
        result = {
            "candidate_id": candidate_id,
            "location_diversity_score": location_diversity,
            "location_unique": location_unique,
            "education_diversity_score": education_diversity,
            "unique_education_backgrounds": unique_education,
            "overall_diversity_score": overall_diversity_score
        }
        
        return result
    
    def create_team_object(self, selected_candidates: List[Candidate]) -> Team:
        """
        Create a Team object from selected candidates
        
        Args:
            selected_candidates: List of selected candidates
            
        Returns:
            Team object
        """
        team_members = []
        
        # Convert candidates to team members
        for candidate in selected_candidates:
            member = TeamMember(
                candidate=candidate,
                role=candidate.best_role or "Team Member",
                score=candidate.best_role_score,
                justification=""  # Will be filled by JustifierAgent
            )
            team_members.append(member)
        
        # Calculate team diversity metrics
        locations = Counter([m.candidate.metadata.location for m in team_members])
        location_diversity = len(locations) / len(team_members) if team_members else 0
        
        education_backgrounds = set()
        for member in team_members:
            education_backgrounds.update([s for s in member.candidate.metadata.education.subjects if s])
        
        skills = set()
        for member in team_members:
            skills.update(member.candidate.metadata.skills)
        
        roles = Counter([m.role for m in team_members])
        role_diversity = len(roles) / len(team_members) if team_members else 0
        
        # Create team
        team = Team(
            members=team_members,
            diversity_metrics={
                "location_diversity": location_diversity,
                "unique_locations": len(locations),
                "education_diversity": len(education_backgrounds),
                "skill_diversity": len(skills),
                "role_diversity": role_diversity
            },
            total_score=sum(m.score for m in team_members)
        )
        
        return team
