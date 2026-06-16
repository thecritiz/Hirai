"""
FastAPI application for the hiring system.
This exposes the hiring system functionality through a REST API.
"""

import logging
import os
from typing import Dict, List, Any, Optional
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from hiring_system.pipeline.orchestrator import HiringOrchestrator
from hiring_system.config.settings import DEFAULT_DATA_PATH, DEFAULT_OUTPUT_PATH

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("hiring_api.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Hiring System API",
    description="API for the 100B Jobs AI-powered Hiring System",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define request and response models
class TopCandidatesRequest(BaseModel):
    data_path: str = str(DEFAULT_DATA_PATH)
    candidates_per_role: int = 3
    aws_region: str = "us-east-1"
    aws_profile: Optional[str] = None

class StandardHiringRequest(BaseModel):
    data_path: str = str(DEFAULT_DATA_PATH)
    team_size: int = 3
    shortlist_size: int = 30
    aws_region: str = "us-east-1"
    aws_profile: Optional[str] = None

class StatusResponse(BaseModel):
    status: str
    version: str
    active_jobs: int = 0

# Store for background jobs
background_jobs = {}

# Routes
@app.get("/", response_model=StatusResponse)
async def root():
    """Get API status"""
    return {
        "status": "running",
        "version": "1.0.0",
        "active_jobs": len(background_jobs)
    }

@app.post("/top-candidates")
async def get_top_candidates(request: TopCandidatesRequest):
    """
    Get top candidates for each role.
    
    This endpoint processes the candidate data and returns the top N candidates
    for each role based on their role-specific scores.
    """
    try:
        # Check if data file exists
        if not Path(request.data_path).exists():
            raise HTTPException(status_code=404, detail=f"Data file not found: {request.data_path}")
        
        # Create orchestrator
        orchestrator = HiringOrchestrator(
            aws_region=request.aws_region,
            aws_profile=request.aws_profile,
            data_path=request.data_path
        )
        
        # Run pipeline steps
        orchestrator._load_and_preprocess_data()
        orchestrator._generate_embeddings()
        orchestrator._build_graph_and_rank()
        
        # Get top candidates by role
        top_candidates = orchestrator.get_top_candidates_by_role(count=request.candidates_per_role)
        
        return top_candidates
    
    except Exception as e:
        logger.exception(f"Error in get_top_candidates: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/run-hiring-pipeline")
async def run_hiring_pipeline(request: StandardHiringRequest, background_tasks: BackgroundTasks):
    """
    Run the standard hiring pipeline to select a team.
    
    This endpoint initiates the full hiring pipeline as a background task
    and returns a job ID that can be used to check the status and get results.
    """
    try:
        # Check if data file exists
        if not Path(request.data_path).exists():
            raise HTTPException(status_code=404, detail=f"Data file not found: {request.data_path}")
        
        # Generate job ID
        import uuid
        job_id = str(uuid.uuid4())
        
        # Create a function to run the pipeline in the background
        def run_pipeline():
            try:
                # Create orchestrator
                orchestrator = HiringOrchestrator(
                    aws_region=request.aws_region,
                    aws_profile=request.aws_profile,
                    team_size=request.team_size,
                    shortlist_size=request.shortlist_size,
                    data_path=request.data_path,
                    output_path=f"results_{job_id}.json"
                )
                
                # Run pipeline
                team = orchestrator.run()
                
                # Save results
                orchestrator.save_results()
                
                # Update job status
                background_jobs[job_id]["status"] = "completed"
                background_jobs[job_id]["results"] = {
                    "team_size": len(team.members),
                    "diversity_metrics": team.diversity_metrics,
                    "total_score": team.total_score,
                    "members": [
                        {
                            "name": member.candidate.metadata.name,
                            "role": member.role,
                            "score": member.score,
                            "justification": member.justification,
                            "location": member.candidate.metadata.location,
                            "skills": member.candidate.metadata.skills,
                        } for member in team.members
                    ]
                }
            
            except Exception as e:
                logger.exception(f"Error in background job {job_id}: {e}")
                background_jobs[job_id]["status"] = "failed"
                background_jobs[job_id]["error"] = str(e)
        
        # Store job information
        background_jobs[job_id] = {
            "status": "running",
            "request": request.dict(),
            "results": None,
            "error": None
        }
        
        # Start the background task
        background_tasks.add_task(run_pipeline)
        
        return {"job_id": job_id, "status": "running"}
    
    except Exception as e:
        logger.exception(f"Error in run_hiring_pipeline: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/job/{job_id}")
async def get_job_status(job_id: str):
    """
    Get the status of a background job.
    
    This endpoint returns the current status of a job and its results
    if the job has completed.
    """
    if job_id not in background_jobs:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    
    job = background_jobs[job_id]
    return job

@app.delete("/job/{job_id}")
async def delete_job(job_id: str):
    """
    Delete a job and its results.
    
    This endpoint deletes a job and its results from the server.
    """
    if job_id not in background_jobs:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    
    # Delete the job
    del background_jobs[job_id]
    
    # Delete the results file if it exists
    results_file = Path(f"results_{job_id}.json")
    if results_file.exists():
        results_file.unlink()
    
    return {"message": f"Job {job_id} deleted"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
