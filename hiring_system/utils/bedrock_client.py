"""
AWS Bedrock Client for Claude Haiku integration
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

class BedrockClient:
    """Client for AWS Bedrock Claude Haiku interactions"""
    
    MODEL_ID = "anthropic.claude-3-haiku-20240307-v1:0"
    
    def __init__(self, region_name: str = "us-east-1", profile_name: Optional[str] = None):
        """
        Initialize the Bedrock client
        
        Args:
            region_name: AWS region
            profile_name: AWS profile name to use (if None, uses default credentials)
        """
        # Read AWS credentials from environment if available
        aws_access_key = os.environ.get("AWS_ACCESS_KEY_ID")
        aws_secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
        aws_session_token = os.environ.get("AWS_SESSION_TOKEN")
        
        session_kwargs = {}
        if profile_name:
            session_kwargs["profile_name"] = profile_name
        
        # Use explicit credentials if provided in environment
        if aws_access_key and aws_secret_key:
            session_kwargs["aws_access_key_id"] = aws_access_key
            session_kwargs["aws_secret_access_key"] = aws_secret_key
            if aws_session_token:
                session_kwargs["aws_session_token"] = aws_session_token
            
        # Create session and client
        session = boto3.Session(**session_kwargs)
        self.client = session.client("bedrock-runtime", region_name=region_name)
        logger.info(f"Initialized Bedrock client for model {self.MODEL_ID}")
        
    def generate_response(self, 
                         prompt: str, 
                         system_prompt: Optional[str] = None,
                         temperature: float = 0.0, 
                         max_tokens: int = 1024) -> str:
        """
        Generate a response from Claude Haiku
        
        Args:
            prompt: User prompt/question
            system_prompt: Optional system instructions
            temperature: Temperature for response generation (0.0 = deterministic)
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated text response
        """
        try:
            # Construct messages
            messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
            
            # Create request body
            request_body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": messages
            }
            
            # Add system prompt if provided
            if system_prompt:
                request_body["system"] = system_prompt
                
            # Invoke model
            response = self.client.invoke_model(
                modelId=self.MODEL_ID,
                body=json.dumps(request_body)
            )
            
            # Parse response
            response_body = json.loads(response.get("body").read().decode("utf-8"))
            return response_body["content"][0]["text"]
            
        except ClientError as e:
            logger.error(f"Error calling Bedrock: {e}")
            raise
    
    def generate_with_reasoning(self, 
                              prompt: str, 
                              system_prompt: Optional[str] = None,
                              reasoning_budget: int = 2000,
                              temperature: float = 0.0, 
                              max_tokens: int = 1024) -> Dict[str, str]:
        """
        Generate a response with explicit reasoning using Claude Haiku's reasoning capability
        
        Args:
            prompt: User prompt/question
            system_prompt: Optional system instructions
            reasoning_budget: Token budget for reasoning
            temperature: Temperature for response generation
            max_tokens: Maximum tokens to generate
            
        Returns:
            Dict with 'reasoning' and 'response' keys
        """
        try:
            # Create converse request with message
            message = {
                "role": "user", 
                "content": [{"type": "text", "text": prompt}]
            }
            
            # Configure reasoning parameters
            reasoning_config = {
                "thinking": {
                    "type": "enabled",
                    "budget_tokens": reasoning_budget
                }
            }
            
            request_kwargs = {
                "modelId": self.MODEL_ID,
                "messages": [message],
                "inferenceConfig": {
                    "maxTokens": max_tokens,
                    "temperature": temperature,
                    "topP": 0.9
                },
                "additionalModelRequestFields": reasoning_config
            }
            
            # Add system prompt if provided
            if system_prompt:
                request_kwargs["system"] = [{"text": system_prompt}]
                
            # Invoke model with converse API
            response = self.client.converse(**request_kwargs)
            
            # Extract reasoning and response text
            result = {"reasoning": "", "response": ""}
            
            for content_block in response["output"]["message"]["content"]:
                if content_block.get("type") == "reasoning":
                    result["reasoning"] = content_block.get("text", "")
                elif content_block.get("type") == "text":
                    result["response"] = content_block.get("text", "")
            
            return result
            
        except ClientError as e:
            logger.error(f"Error calling Bedrock converse API: {e}")
            # Fallback response
            return {"reasoning": f"Error: {str(e)}", "response": "Sorry, I encountered an error."}

    def justify_candidate(self, candidate_info: Dict[str, Any], team_so_far: List[Dict[str, Any]]) -> str:
        """
        Generate a justification for selecting a candidate
        
        Args:
            candidate_info: Information about the candidate
            team_so_far: List of team members selected so far
            
        Returns:
            Justification text
        """
        # Format current team info
        team_description = ""
        for i, member in enumerate(team_so_far, 1):
            name = member["metadata"]["name"]
            role = member.get("role_match", "Team Member")
            skills = ", ".join(member["metadata"]["skills"]) if member["metadata"]["skills"] else "No listed skills"
            team_description += f"{i}. {name} - {role}, Skills: {skills}\n"
        
        # Format candidate info
        name = candidate_info["metadata"]["name"]
        role = candidate_info.get("role_match", "Team Member")
        education = candidate_info["metadata"]["education"]
        schools = ", ".join(education["schools"]) if education["schools"] else "Unknown"
        experience_years = candidate_info["metadata"]["experience"]["years"]
        skills = ", ".join(candidate_info["metadata"]["skills"]) if candidate_info["metadata"]["skills"] else "No listed skills"
        
        # Create prompt
        prompt = f"""
        You are a senior hiring manager at a tech startup. Explain why the following candidate would be a good fit 
        for the role of {role} in our growing team.
        
        Current team composition:
        {team_description if team_description else "This is the first team member being selected."}
        
        Candidate information:
        - Name: {name}
        - Role being considered: {role}
        - Education: {education["highest_level"]} from {schools}
        - Experience: {experience_years} years
        - Skills: {skills}
        
        Write 2-3 sentences explaining why this person would be a good addition to the team. 
        Consider their skills, experience, education, and how they complement the existing team members.
        Be specific and focus on the value they bring to the role of {role}.
        """
        
        system_prompt = """
        You are a hiring expert who specializes in creating concise, compelling justifications for candidate selections.
        Your answers should be 2-3 sentences long, focused on concrete qualifications, and highlight the unique value
        this candidate brings to the specific role and team.
        """
        
        return self.generate_response(prompt, system_prompt=system_prompt, max_tokens=256)
