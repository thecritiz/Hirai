"""
Preprocessing Agent for candidate data extraction
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd

from hiring_system.utils.models import Candidate, CandidateMetadata, Education, Experience

logger = logging.getLogger(__name__)


class PreprocessingAgent:
    """
    Preprocessing Agent for extracting structured candidate data
    from JSON or CSV files
    """
    
    def __init__(self):
        """Initialize the preprocessing agent"""
        logger.info("Initializing PreprocessingAgent")
    
    def process_json_file(self, file_path: str) -> List[Candidate]:
        """
        Process a JSON file containing candidate data
        
        Args:
            file_path: Path to the JSON file
            
        Returns:
            List of structured Candidate objects
        """
        logger.info(f"Processing JSON file: {file_path}")
        
        try:
            # Read the JSON file
            with open(file_path, 'r') as f:
                raw_data = json.load(f)
                
            # Process each candidate record
            candidates = []
            for i, candidate_data in enumerate(raw_data):
                candidate = self._process_candidate(candidate_data, f"candidate_{i}")
                candidates.append(candidate)
                
            logger.info(f"Processed {len(candidates)} candidates from JSON")
            return candidates
            
        except Exception as e:
            logger.error(f"Error processing JSON file: {e}")
            raise
    
    def process_csv_file(self, file_path: str) -> List[Candidate]:
        """
        Process a CSV file containing candidate data
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            List of structured Candidate objects
        """
        logger.info(f"Processing CSV file: {file_path}")
        
        try:
            # Read the CSV file
            df = pd.read_csv(file_path)
            
            # Process each candidate record
            candidates = []
            for i, row in df.iterrows():
                # Convert row to dictionary
                candidate_data = row.to_dict()
                
                # Process lists from string representation
                for field in ['skills', 'work_availability']:
                    if field in candidate_data and isinstance(candidate_data[field], str):
                        candidate_data[field] = [s.strip() for s in candidate_data[field].split(',') if s.strip()]
                
                # Extract education data
                education_data = {}
                if 'highest_education' in candidate_data:
                    education_data['highest_level'] = candidate_data.get('highest_education', '')
                
                schools = []
                subjects = []
                gpas = []
                
                if 'schools' in candidate_data and isinstance(candidate_data['schools'], str):
                    schools = [s.strip() for s in candidate_data['schools'].split(',') if s.strip()]
                
                if 'subjects' in candidate_data and isinstance(candidate_data['subjects'], str):
                    subjects = [s.strip() for s in candidate_data['subjects'].split(',') if s.strip()]
                    
                if 'gpas' in candidate_data and isinstance(candidate_data['gpas'], str):
                    gpas = [s.strip() for s in candidate_data['gpas'].split(',') if s.strip()]
                
                education_data['schools'] = schools
                education_data['subjects'] = subjects
                education_data['gpas'] = gpas
                
                # Extract experience data
                experience_data = {}
                if 'companies' in candidate_data and isinstance(candidate_data['companies'], str):
                    experience_data['companies'] = [s.strip() for s in candidate_data['companies'].split(',') if s.strip()]
                
                if 'roles' in candidate_data and isinstance(candidate_data['roles'], str):
                    experience_data['roles'] = [s.strip() for s in candidate_data['roles'].split(',') if s.strip()]
                
                # Update candidate data with structured data
                candidate_data['education'] = education_data
                candidate_data['experience'] = experience_data
                
                # Process candidate
                candidate = self._process_candidate(candidate_data, f"candidate_{i}")
                candidates.append(candidate)
                
            logger.info(f"Processed {len(candidates)} candidates from CSV")
            return candidates
            
        except Exception as e:
            logger.error(f"Error processing CSV file: {e}")
            raise
    
    def _process_candidate(self, raw_data: Dict[str, Any], candidate_id: str) -> Candidate:
        """
        Process a single candidate record
        
        Args:
            raw_data: Raw candidate data
            candidate_id: Unique identifier for the candidate
            
        Returns:
            Structured Candidate object
        """
        # Extract education details
        education_data = raw_data.get('education', {})
        highest_level = education_data.get('highest_level', '')
        
        degrees = education_data.get('degrees', [])
        schools = [degree.get('originalSchool', '') for degree in degrees] if degrees else education_data.get('schools', [])
        subjects = [degree.get('subject', '') for degree in degrees if degree.get('subject')] if degrees else education_data.get('subjects', [])
        
        # Extract GPAs from degrees if available
        gpas = []
        if degrees:
            gpas = [degree.get('gpa', '') for degree in degrees if degree.get('gpa')]
        else:
            gpas = education_data.get('gpas', [])
        
        # Check for prestigious schools
        is_top50 = False
        is_top25 = False
        if degrees:
            is_top50 = any(degree.get('isTop50', False) for degree in degrees)
            is_top25 = any(degree.get('isTop25', False) for degree in degrees)
        
        education = Education(
            highest_level=highest_level,
            schools=schools,
            subjects=subjects,
            gpas=gpas,
            is_top50=is_top50,
            is_top25=is_top25
        )
        
        # Extract work experience details
        experiences = raw_data.get('work_experiences', [])
        companies = []
        roles = []
        
        if experiences:
            companies = [exp.get('company', '') for exp in experiences]
            roles = [exp.get('roleName', '') for exp in experiences]
        else:
            companies = raw_data.get('companies', [])
            roles = raw_data.get('roles', [])
        
        # Categorize roles
        role_categories = self._categorize_roles(roles)
        
        experience = Experience(
            companies=companies,
            roles=roles,
            role_categories=role_categories,
            years=len(experiences) * 1.5 if experiences else len(companies) * 1.5  # Rough estimate
        )
        
        # Extract other metadata
        metadata = CandidateMetadata(
            name=raw_data.get('name', ''),
            email=raw_data.get('email', ''),
            location=raw_data.get('location', ''),
            submitted_at=raw_data.get('submitted_at', ''),
            work_availability=raw_data.get('work_availability', []),
            annual_salary_expectation=raw_data.get('annual_salary_expectation', {}).get('full-time', '') 
                if isinstance(raw_data.get('annual_salary_expectation'), dict) 
                else raw_data.get('annual_salary_expectation', ''),
            education=education,
            experience=experience,
            skills=raw_data.get('skills', [])
        )
        
        # Create and return candidate
        return Candidate(
            id=candidate_id,
            metadata=metadata
        )
    
    def _categorize_roles(self, roles: List[str]) -> Dict[str, int]:
        """
        Categorize roles into standard categories
        
        Args:
            roles: List of role titles
            
        Returns:
            Dictionary of role categories with counts
        """
        categories = {
            'engineering': 0,
            'management': 0,
            'legal': 0,
            'finance': 0,
            'product': 0,
            'design': 0,
            'marketing': 0,
            'operations': 0,
            'data_science': 0,
            'research': 0,
            'other': 0
        }
        
        for role in roles:
            role_lower = role.lower() if role else ""
            if any(term in role_lower for term in ['developer', 'engineer', 'stack', 'software', 'system administrator']):
                categories['engineering'] += 1
            elif any(term in role_lower for term in ['manager', 'director', 'lead']):
                categories['management'] += 1
            elif any(term in role_lower for term in ['legal', 'attorney', 'lawyer']):
                categories['legal'] += 1
            elif any(term in role_lower for term in ['finance', 'financial', 'accountant']):
                categories['finance'] += 1
            elif any(term in role_lower for term in ['product']):
                categories['product'] += 1
            elif any(term in role_lower for term in ['design', 'ux', 'ui']):
                categories['design'] += 1
            elif any(term in role_lower for term in ['marketing', 'growth']):
                categories['marketing'] += 1
            elif any(term in role_lower for term in ['operations', 'operator']):
                categories['operations'] += 1
            elif any(term in role_lower for term in ['data', 'scientist', 'analytics']):
                categories['data_science'] += 1
            elif any(term in role_lower for term in ['research', 'scientist']):
                categories['research'] += 1
            else:
                categories['other'] += 1
                
        return categories
    
    def generate_candidate_summary(self, candidate: Candidate) -> str:
        """
        Generate a concise summary of a candidate
        
        Args:
            candidate: Candidate object
            
        Returns:
            Summary text
        """
        metadata = candidate.metadata
        name = metadata.name
        highest_edu = metadata.education.highest_level
        exp_years = metadata.experience.years
        
        # Get top roles
        top_roles = self._get_top_roles(metadata.experience.role_categories)
        
        # Format skills
        skills = ', '.join(metadata.skills) if metadata.skills else "No listed skills"
        
        # Create summary
        summary = f"{name} has {highest_edu} education with approximately {exp_years:.1f} years of experience. "
        summary += f"Primary areas: {top_roles}. "
        summary += f"Skills include {skills}."
        
        return summary
    
    def _get_top_roles(self, role_categories: Dict[str, int], top_n: int = 2) -> str:
        """
        Get top role categories by count
        
        Args:
            role_categories: Dictionary of role categories with counts
            top_n: Number of top categories to return
            
        Returns:
            Comma-separated string of top role categories
        """
        sorted_roles = sorted(role_categories.items(), key=lambda x: x[1], reverse=True)
        top_roles = [role.replace('_', ' ') for role, count in sorted_roles[:top_n] if count > 0]
        if not top_roles:
            return "varied roles"
        return ' and '.join(top_roles)
    
    def extract_keywords(self, candidate: Candidate) -> List[str]:
        """
        Extract keywords from a candidate's profile
        
        Args:
            candidate: Candidate object
            
        Returns:
            List of keywords
        """
        keywords = set()
        
        # Add skills
        keywords.update(candidate.metadata.skills)
        
        # Add education subjects
        subjects = candidate.metadata.education.subjects
        keywords.update([s for s in subjects if s])
        
        # Add role titles (split multi-word titles)
        roles = candidate.metadata.experience.roles
        for role in roles:
            if role:
                words = role.split()
                for word in words:
                    if len(word) > 3:  # Skip short words
                        keywords.add(word)
        
        return list(keywords)
    
    def enrich_candidates(self, candidates: List[Candidate]) -> List[Candidate]:
        """
        Enrich candidate profiles with summaries and keywords
        
        Args:
            candidates: List of candidates
            
        Returns:
            Enriched candidates
        """
        enriched_candidates = []
        
        for candidate in candidates:
            # Generate summary
            summary = self.generate_candidate_summary(candidate)
            
            # Extract keywords
            keywords = self.extract_keywords(candidate)
            
            # Update candidate
            candidate.summary = summary
            candidate.keywords = keywords
            
            enriched_candidates.append(candidate)
        
        return enriched_candidates
