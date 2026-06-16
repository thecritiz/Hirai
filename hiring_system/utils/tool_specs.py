"""
Tool specifications for Claude Haiku's tool use capabilities
"""

import json
from typing import Dict, Any, List

# Tool specifications for candidate evaluation and team building
GRAPH_SCORE_TOOL_SPEC = {
    "name": "Graph_Score_Tool",
    "description": "Returns the PageRank-based graph score for a candidate based on the role relevance graph.",
    "inputSchema": {
        "type": "object",
        "properties": {
            "candidate_id": {
                "type": "string",
                "description": "Unique ID of the candidate."
            },
            "role": {
                "type": "string",
                "description": "The role being considered for the candidate."
            }
        },
        "required": ["candidate_id", "role"]
    }
}

SKILL_GAP_TOOL_SPEC = {
    "name": "Skill_Gap_Tool",
    "description": "Calculates the number of unique new skills a candidate adds to the team.",
    "inputSchema": {
        "type": "object",
        "properties": {
            "candidate_id": {
                "type": "string",
                "description": "Unique ID of the candidate."
            },
            "team_skills": {
                "type": "array",
                "items": {"type": "string"},
                "description": "The union of all current team members' skills."
            }
        },
        "required": ["candidate_id", "team_skills"]
    }
}

DIVERSITY_TOOL_SPEC = {
    "name": "Diversity_Tool",
    "description": "Evaluates how much diversity a candidate adds to the existing team.",
    "inputSchema": {
        "type": "object",
        "properties": {
            "candidate_id": {
                "type": "string",
                "description": "Unique ID of the candidate."
            },
            "team_locations": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Locations of current team members."
            },
            "team_education_backgrounds": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Education backgrounds of current team members."
            }
        },
        "required": ["candidate_id", "team_locations", "team_education_backgrounds"]
    }
}

EDUCATION_SCORE_TOOL_SPEC = {
    "name": "Education_Score_Tool",
    "description": "Returns the education quality score for a candidate.",
    "inputSchema": {
        "type": "object",
        "properties": {
            "candidate_id": {
                "type": "string",
                "description": "Unique ID of the candidate."
            }
        },
        "required": ["candidate_id"]
    }
}

EXPERIENCE_SCORE_TOOL_SPEC = {
    "name": "Experience_Score_Tool",
    "description": "Returns the experience quality score for a candidate.",
    "inputSchema": {
        "type": "object",
        "properties": {
            "candidate_id": {
                "type": "string",
                "description": "Unique ID of the candidate."
            },
            "role": {
                "type": "string", 
                "description": "The role being considered for the candidate."
            }
        },
        "required": ["candidate_id", "role"]
    }
}

CANDIDATE_INFO_TOOL_SPEC = {
    "name": "Candidate_Info_Tool",
    "description": "Retrieves detailed information about a candidate.",
    "inputSchema": {
        "type": "object",
        "properties": {
            "candidate_id": {
                "type": "string",
                "description": "Unique ID of the candidate."
            },
            "include_full_profile": {
                "type": "boolean",
                "description": "Whether to include the full profile details or just summary information."
            }
        },
        "required": ["candidate_id"]
    }
}

# All available tool specifications
TOOL_SPECS = {
    "graph_score": GRAPH_SCORE_TOOL_SPEC,
    "skill_gap": SKILL_GAP_TOOL_SPEC,
    "diversity": DIVERSITY_TOOL_SPEC,
    "education_score": EDUCATION_SCORE_TOOL_SPEC,
    "experience_score": EXPERIENCE_SCORE_TOOL_SPEC,
    "candidate_info": CANDIDATE_INFO_TOOL_SPEC
}

def get_tool_specs() -> List[Dict[str, Any]]:
    """
    Get all tool specifications for Claude tool use
    
    Returns:
        List of tool specifications
    """
    return list(TOOL_SPECS.values())

def get_tool_spec(tool_name: str) -> Dict[str, Any]:
    """
    Get a specific tool specification by name
    
    Args:
        tool_name: Name of the tool
        
    Returns:
        Tool specification
    """
    return TOOL_SPECS[tool_name]
