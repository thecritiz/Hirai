"""
Configuration settings for the hiring system
"""

import os
from pathlib import Path
from typing import Dict, List, Any

# Define roles and their requirements
ROLE_DEFINITIONS = {
    "Engineering Lead": {
        "description": (
            "Senior engineer with strong technical and leadership skills. "
            "Experience with system architecture, team management, and full-stack development. "
            "Strong problem-solving abilities and communication skills."
        ),
        "required_skills": ["Architecture", "Leadership", "Full Stack", "System Design"],
        "weights": {
            "technical": 0.7,
            "leadership": 0.8,
            "education": 0.4,
            "experience": 0.9
        }
    },
    "Product Manager": {
        "description": (
            "Strategic product thinker with experience defining product vision and roadmap. "
            "Data-driven decision maker with strong user empathy. Background in product development "
            "and cross-functional team coordination."
        ),
        "required_skills": ["Product Management", "Strategy", "Data Analysis", "UX"],
        "weights": {
            "technical": 0.4,
            "leadership": 0.7,
            "education": 0.5,
            "experience": 0.8
        }
    },
    "Senior Developer": {
        "description": (
            "Experienced software engineer with strong coding skills. Deep knowledge of "
            "software development practices, architecture patterns, and technical implementation. "
            "Experience with multiple languages and frameworks."
        ),
        "required_skills": ["Full Stack", "Microservices", "Docker", "Cloud"],
        "weights": {
            "technical": 0.9,
            "leadership": 0.3,
            "education": 0.5,
            "experience": 0.8
        }
    },
    "Marketing/Growth Lead": {
        "description": (
            "Data-driven marketer with experience growing user acquisition and retention. "
            "Strategic thinker with background in digital marketing, analytics, and growth hacking. "
            "Creative problem solver with strong communication skills."
        ),
        "required_skills": ["Marketing", "Growth Hacking", "Analytics", "SEO"],
        "weights": {
            "technical": 0.3,
            "leadership": 0.6,
            "education": 0.4,
            "experience": 0.8
        }
    },
    "Finance/Operations": {
        "description": (
            "Experienced in financial planning, accounting, and operations management. "
            "Strong analytical skills and attention to detail. Background in financial reporting, "
            "budgeting, and operational efficiency."
        ),
        "required_skills": ["Finance", "Accounting", "Operations", "Legal"],
        "weights": {
            "technical": 0.4,
            "leadership": 0.6,
            "education": 0.7,
            "experience": 0.8
        }
    }
}

# Diversity factors for team building
DIVERSITY_FACTORS = {
    "location": 0.3,       # Geographic diversity weight
    "education": 0.25,     # Educational background diversity weight
    "experience": 0.25,    # Industry experience diversity weight
    "skills": 0.2          # Complementary skill sets weight
}

# LLM settings
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
BEDROCK_CLAUDE_MODEL = "us.anthropic.claude-3-haiku-20240307-v1:0"

# Role match settings
ROLE_MATCH_WEIGHTS = {
    "similarity": 0.5,     # Semantic similarity to role description
    "skills": 0.3,         # Hard skill match
    "education": 0.1,      # Education quality/relevance
    "experience": 0.1      # Experience relevance
}

# FAISS indexing settings
FAISS_DIMENSION = 768
FAISS_METRIC_TYPE = "cosine"  # Other options: "l2", "inner_product"

# General settings
PROJECT_ROOT = Path(__file__).parent.parent.parent
DEFAULT_DATA_PATH = PROJECT_ROOT / "main_data.json"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "selected_team.json"
TEAM_SIZE = 5
SHORTLIST_SIZE = 30
