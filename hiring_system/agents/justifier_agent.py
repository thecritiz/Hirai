"""
Justifier Agent for generating candidate selection justifications using Claude Haiku
"""

import logging
from typing import Dict, List, Any, Optional

from hiring_system.utils.models import Candidate, TeamMember, Team
from hiring_system.utils.bedrock_client import BedrockClient

logger = logging.getLogger(__name__)


class JustifierAgent:
    """
    Agent for generating justifications for selected candidates using Claude Haiku
    """
    
    def __init__(self, aws_region: str = "us-east-1", aws_profile: Optional[str] = None):
        """
        Initialize the justifier agent
        
        Args:
            aws_region: AWS region for Bedrock
            aws_profile: Optional AWS profile to use
        """
        logger.info("Initializing JustifierAgent")
        self.bedrock_client = BedrockClient(region_name=aws_region, profile_name=aws_profile)
        
    def justify_candidate(self, candidate: Candidate, team_so_far: List[Candidate]) -> str:
        """
        Generate justification for selecting a candidate
        
        Args:
            candidate: Candidate to justify
            team_so_far: Team members selected so far
            
        Returns:
            Justification text
        """
        logger.info(f"Generating justification for {candidate.metadata.name}")
        
        try:
            # Format team info
            team_description = ""
            for i, member in enumerate(team_so_far, 1):
                name = member.metadata.name
                role = member.best_role or "Team Member"
                skills = ", ".join(member.metadata.skills) if member.metadata.skills else "No listed skills"
                team_description += f"{i}. {name} - {role}, Skills: {skills}\n"
            
            # Prepare candidate info for prompt
            role = candidate.best_role or "Team Member"
            name = candidate.metadata.name
            location = candidate.metadata.location
            education = candidate.metadata.education
            schools = ", ".join(education.schools) if education.schools else "Unknown"
            education_level = education.highest_level
            experience = candidate.metadata.experience
            experience_years = experience.years
            roles = ", ".join(experience.roles[:3]) if experience.roles else "Unknown roles"
            skills = ", ".join(candidate.metadata.skills) if candidate.metadata.skills else "No listed skills"
            
            # Create prompt for Claude
            prompt = f"""
            You are a senior hiring manager at a tech startup that has just raised $100M in funding.
            Explain why the following candidate would be a good fit for the role of {role} in the team 
            we're building.
            
            Current team composition:
            {team_description if team_description else "This is the first team member being selected."}
            
            Candidate information:
            - Name: {name}
            - Role being considered: {role}
            - Location: {location}
            - Education: {education_level} from {schools}
            - Experience: {experience_years} years including roles such as {roles}
            - Skills: {skills}
            
            Write 2-3 sentences explaining why this person would be a good addition to the team. 
            Consider their skills, experience, education, and how they complement the existing team members.
            Be specific and focus on the value they bring to the role of {role}.
            """
            
            # System prompt to guide Claude's response
            system_prompt = """
            You are a hiring expert who specializes in creating concise, compelling justifications 
            for candidate selections. Your answers should be 2-3 sentences long, focused on concrete 
            qualifications, and highlight the unique value this candidate brings to the specific role and team.
            
            Focus on:
            1. Relevant skills and experience for the role
            2. How they complement the existing team
            3. Any unique strengths or diversity they bring
            
            Be direct, specific, and convincing.
            """
            
            # Generate justification
            justification = self.bedrock_client.generate_response(
                prompt=prompt,
                system_prompt=system_prompt,
                max_tokens=256
            )
            
            logger.info(f"Generated justification for {name}")
            return justification.strip()
        
        except Exception as e:
            logger.error(f"Error generating justification: {e}")
            # Fallback justification
            return f"{candidate.metadata.name} brings valuable experience to the {candidate.best_role} role."
    
    def justify_team(self, team: Team) -> Team:
        """
        Generate justifications for all team members
        
        Args:
            team: Team with members
            
        Returns:
            Team with justifications added
        """
        logger.info("Generating justifications for team")
        
        # Iterate through team members
        for i, member in enumerate(team.members):
            # Build team so far
            team_so_far = [m.candidate for m in team.members[:i]]
            
            # Generate justification
            justification = self.justify_candidate(member.candidate, team_so_far)
            
            # Update member
            member.justification = justification
        
        logger.info("Completed team justifications")
        return team
    
    def justify_selection_with_reasoning(self, candidate: Candidate, role: str, team_so_far: List[Candidate]) -> Dict[str, str]:
        """
        Generate justification with explicit reasoning steps
        
        Args:
            candidate: Candidate to justify
            role: Role being considered
            team_so_far: Team members selected so far
            
        Returns:
            Dictionary with reasoning and justification
        """
        logger.info(f"Generating reasoning for {candidate.metadata.name}")
        
        try:
            # Format team info
            team_description = ""
            team_skills = set()
            team_locations = set()
            
            for member in team_so_far:
                team_description += f"- {member.metadata.name}: {member.best_role}, {', '.join(member.metadata.skills)}\n"
                team_skills.update(member.metadata.skills)
                team_locations.add(member.metadata.location)
            
            # Get candidate info
            name = candidate.metadata.name
            location = candidate.metadata.location
            education = candidate.metadata.education
            experience = candidate.metadata.experience
            skills = candidate.metadata.skills
            
            # Check location diversity
            location_diversity = "adds geographic diversity" if location not in team_locations else "same region as existing team member(s)"
            
            # Check skill complementarity
            new_skills = set(skills) - team_skills
            skill_complement = f"adds {len(new_skills)} new skills" if new_skills else "overlapping skill set with team"
            
            # Prompt for reasoning
            prompt = f"""
            Evaluate this candidate for our startup team:
            
            CANDIDATE:
            - Name: {name}
            - Role: {role}
            - Location: {location} ({location_diversity})
            - Education: {education.highest_level} from {', '.join(education.schools)}
            - Experience: {experience.years} years in {', '.join(experience.roles[:2])}
            - Skills: {', '.join(skills)} ({skill_complement})
            - Top school: {"Yes" if education.is_top25 or education.is_top50 else "No"}
            
            CURRENT TEAM:
            {team_description if team_description else "No members selected yet."}
            
            Think through whether this candidate is a good selection, considering:
            1. Role fit - Do they have the right skills and experience?
            2. Team complementarity - Do they add new capabilities?
            3. Diversity - Do they bring new perspectives?
            
            Then write a 2-3 sentence justification for selecting them.
            """
            
            # Generate reasoning and justification
            result = self.bedrock_client.generate_with_reasoning(prompt=prompt)
            
            logger.info(f"Generated reasoning for {name}")
            return result
        
        except Exception as e:
            logger.error(f"Error generating reasoning: {e}")
            # Fallback
            return {
                "reasoning": f"Error: {str(e)}",
                "response": f"{candidate.metadata.name} brings valuable experience to the {role} role."
            }
    
    def get_final_team_assessment(self, team: Team) -> str:
        """
        Generate overall assessment of the selected team
        
        Args:
            team: Selected team
            
        Returns:
            Assessment text
        """
        logger.info("Generating team assessment")
        
        try:
            # Format team info
            team_description = ""
            all_skills = set()
            all_locations = set()
            
            for i, member in enumerate(team.members, 1):
                candidate = member.candidate
                role = member.role
                skills = candidate.metadata.skills
                location = candidate.metadata.location
                
                team_description += f"{i}. {candidate.metadata.name} - {role}\n"
                team_description += f"   Location: {location}\n"
                team_description += f"   Experience: {candidate.metadata.experience.years} years\n"
                team_description += f"   Skills: {', '.join(skills)}\n\n"
                
                all_skills.update(skills)
                all_locations.add(location)
            
            # Diversity metrics
            location_count = len(all_locations)
            location_diversity = location_count / len(team.members) if team.members else 0
            
            # Prompt for assessment
            prompt = f"""
            Assess this team of {len(team.members)} candidates selected for our startup after raising $100M:
            
            TEAM COMPOSITION:
            {team_description}
            
            TEAM DIVERSITY METRICS:
            - Geographic diversity: {location_count} unique locations
            - Skill diversity: {len(all_skills)} unique skills
            - Role coverage: {len(set(m.role for m in team.members))} different roles
            
            Provide a 3-4 sentence assessment of the team's strengths and potential gaps.
            Focus on how well-rounded the team is, the diversity of skills and backgrounds, and
            whether they form a cohesive group that can execute effectively on building a successful startup.
            """
            
            # Generate assessment
            assessment = self.bedrock_client.generate_response(prompt=prompt, max_tokens=350)
            
            logger.info("Generated team assessment")
            return assessment.strip()
            
        except Exception as e:
            logger.error(f"Error generating team assessment: {e}")
            return "The selected team provides a good balance of skills and experiences."
            
    def get_role_candidates(self, candidates: List[Candidate], role: str, count: int = 3) -> List[Dict[str, Any]]:
        """
        Find the best candidates for a specific role
        
        Args:
            candidates: List of all candidates
            role: Target role to find matches for
            count: Number of candidates to return (default 3)
            
        Returns:
            List of candidate dictionaries with justifications
        """
        logger.info(f"Finding best {count} candidates for role: {role}")
        
        # Use role_matches scores instead of filtering by best_role
        # This ensures we get the best candidates for each role regardless of their best_role
        role_candidates = [(c, c.role_matches.get(role, 0.0)) for c in candidates]
        
        # Sort by role match score (descending)
        role_candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Take top N candidates
        top_candidates = [c[0] for c in role_candidates[:count]]
        
        # Generate results with justifications
        result = []
        for candidate in top_candidates:
            # Generate justification
            justification = self.justify_candidate(candidate, [])
            
            # Build result object
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
            
            result.append(candidate_info)
        
        logger.info(f"Found {len(result)} candidates for {role}")
        return result
