#!/usr/bin/env python3
"""
Script to run the FastAPI application for the hiring system.
"""

import uvicorn
import argparse
import logging

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Hiring System API")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="Host to bind the server to (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000,
                        help="Port to bind the server to (default: 8000)")
    parser.add_argument("--reload", action="store_true",
                        help="Enable auto-reload for development")
    parser.add_argument("--log-level", type=str, default="info",
                        choices=["debug", "info", "warning", "error", "critical"],
                        help="Log level (default: info)")
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = getattr(logging, args.log_level.upper())
    logging.basicConfig(level=log_level)
    
    print(f"Starting Hiring System API on http://{args.host}:{args.port}")
    print("API documentation will be available at /docs")
    
    # Run the server
    uvicorn.run(
        "hiring_system.api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level
    )
