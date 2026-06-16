"""
Data models for the hiring system
"""

from typing import Dict, List, Optional, Set, Union, Any
from pydantic import BaseModel, Field
from datetime import datetime


class Education(BaseModel):
    """Education data for a candidate"""
    highest_level: str = Field(default="")
    schools: List[str] = Field(default_factory=list)
    subjects: List[str] = Field(default_factory=list)
    gpas: List[str] = Field(default_factory=list)
    is_top50: bool = Field(default=False)
    is_top25: bool = Field(default=False)
    
    @property
    def quality_score(self) -> float:
        """Calculate education quality score"""
        score = 0.0
        
        # Score based on highest degree
        degree_scores = {
            "Bachelor's Degree": 1.0,
            "Master's Degree": 2.0,
            "Ph.D": 3.0,
            "Juris Doctor (J.D)": 2.5,
            "MBA": 2.2
        }
        score += degree_scores.get(self.highest_level, 0.0)
        
        # Bonus for prestigious schools
        if self.is_top25:
            score += 1.0
        elif self.is_top50:
            score += 0.5
        
        # GPA bonus (simple approximation)
        has_high_gpa = any("3.5" in gpa for gpa in self.gpas)
        if has_high_gpa:
            score += 0.5
            
        return score


class Experience(BaseModel):
    """Work experience data for a candidate"""
    companies: List[str] = Field(default_factory=list)
    roles: List[str] = Field(default_factory=list)
    role_categories: Dict[str, int] = Field(default_factory=dict)
    years: float = Field(default=0.0)
    

class CandidateMetadata(BaseModel):
    """Metadata for a candidate"""
    name: str = Field(default="")
    email: str = Field(default="")
    location: str = Field(default="")
    submitted_at: str = Field(default="")
    work_availability: List[str] = Field(default_factory=list)
    annual_salary_expectation: str = Field(default="")
    education: Education = Field(default_factory=Education)
    experience: Experience = Field(default_factory=Experience)
    skills: List[str] = Field(default_factory=list)


class Candidate(BaseModel):
    """Candidate data model"""
    id: str
    metadata: CandidateMetadata = Field(default_factory=CandidateMetadata)
    summary: str = Field(default="")
    keywords: List[str] = Field(default_factory=list)
    embedding: Optional[List[float]] = Field(default=None)
    role_matches: Dict[str, float] = Field(default_factory=dict)
    best_role: str = Field(default="")
    best_role_score: float = Field(default=0.0)
    diversity_score: float = Field(default=0.0)
    final_score: float = Field(default=0.0)
    justification: str = Field(default="")


class TeamMember(BaseModel):
    """Selected team member data"""
    candidate: Candidate
    role: str
    score: float
    justification: str


class Team(BaseModel):
    """Team data model"""
    members: List[TeamMember] = Field(default_factory=list)
    diversity_metrics: Dict[str, float] = Field(default_factory=dict)
    total_score: float = Field(default=0.0)
