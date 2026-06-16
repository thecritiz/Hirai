"""
Embedding Agent for candidate embedding generation and similarity search
"""

import os
import logging
import pickle
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import torch
from pathlib import Path
import faiss

from sentence_transformers import SentenceTransformer

from hiring_system.utils.models import Candidate
from hiring_system.config.settings import EMBEDDING_MODEL, FAISS_DIMENSION, FAISS_METRIC_TYPE

logger = logging.getLogger(__name__)


class EmbeddingAgent:
    """
    Agent for generating embeddings and performing similarity search using FAISS
    """
    
    def __init__(self, model_name: str = EMBEDDING_MODEL):
        """
        Initialize the embedding agent
        
        Args:
            model_name: Name of the sentence-transformer model to use
        """
        logger.info(f"Initializing EmbeddingAgent with model {model_name}")
        
        # Determine device - use MPS if available on macOS
        self.device = self._get_optimal_device()
        logger.info(f"Using device: {self.device}")
        
        # Load model
        self.model = SentenceTransformer(model_name, device=str(self.device))
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"Model loaded with embedding dimension: {self.embedding_dim}")
        
        # Initialize FAISS index (will be built later)
        self.index = None
        self.candidate_ids = []
        self.candidate_map = {}
    
    def _get_optimal_device(self) -> torch.device:
        """
        Get the optimal device for embedding generation
        
        Returns:
            Torch device (cuda, mps, or cpu)
        """
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch, 'mps') and torch.backends.mps.is_available() and torch.backends.mps.is_built():
            # MPS (Metal Performance Shaders) for Apple Silicon
            return torch.device("mps")
        else:
            return torch.device("cpu")
    
    def generate_embeddings(self, candidates: List[Candidate]) -> List[Candidate]:
        """
        Generate embeddings for a list of candidates
        
        Args:
            candidates: List of candidate objects
            
        Returns:
            Candidates with embeddings added
        """
        logger.info(f"Generating embeddings for {len(candidates)} candidates")
        
        # Prepare content for embedding
        contents = []
        
        for candidate in candidates:
            # Use summary and content fields for better semantic representation
            content = f"{candidate.summary}\n\n"
            
            # Add education info
            education = candidate.metadata.education
            content += f"Education: {education.highest_level}\n"
            content += f"Schools: {', '.join(education.schools)}\n"
            content += f"Subjects: {', '.join(education.subjects)}\n\n"
            
            # Add experience info
            experience = candidate.metadata.experience
            content += f"Experience: {', '.join(experience.roles)}\n"
            
            # Add skills
            skills = candidate.metadata.skills
            content += f"Skills: {', '.join(skills)}\n"
            
            # Add keywords
            content += f"Keywords: {', '.join(candidate.keywords)}"
            
            contents.append(content)
        
        # Generate embeddings
        embeddings = self.model.encode(contents, convert_to_numpy=True, show_progress_bar=True)
        
        # Update candidates with embeddings
        for i, candidate in enumerate(candidates):
            candidate.embedding = embeddings[i].tolist()
        
        logger.info(f"Generated embeddings for {len(candidates)} candidates")
        return candidates
    
    def build_faiss_index(self, candidates: List[Candidate]) -> None:
        """
        Build a FAISS index from candidate embeddings
        
        Args:
            candidates: List of candidates with embeddings
        """
        logger.info("Building FAISS index")
        
        # Extract embeddings
        embeddings = []
        self.candidate_ids = []
        self.candidate_map = {}
        
        for i, candidate in enumerate(candidates):
            if candidate.embedding is not None:
                embedding = np.array(candidate.embedding, dtype=np.float32)
                embeddings.append(embedding)
                self.candidate_ids.append(candidate.id)
                self.candidate_map[candidate.id] = candidate
        
        # Stack embeddings
        embeddings = np.vstack(embeddings)
        
        # Create index
        dimension = embeddings.shape[1]
        
        if FAISS_METRIC_TYPE == "cosine":
            # For cosine similarity, normalize vectors and use dot product
            faiss.normalize_L2(embeddings)
            self.index = faiss.IndexFlatIP(dimension)
        else:
            # Default to L2 distance
            self.index = faiss.IndexFlatL2(dimension)
        
        # Add embeddings to index
        self.index.add(embeddings)
        
        logger.info(f"Built FAISS index with {len(self.candidate_ids)} candidates")
    
    def save_faiss_index(self, index_path: str, metadata_path: str) -> None:
        """
        Save the FAISS index and related metadata
        
        Args:
            index_path: Path to save the index
            metadata_path: Path to save the metadata
        """
        if self.index is None:
            logger.warning("No index to save")
            return
        
        logger.info(f"Saving FAISS index to {index_path}")
        
        # Save index
        faiss.write_index(self.index, index_path)
        
        # Save metadata (candidate IDs and map)
        metadata = {
            "candidate_ids": self.candidate_ids,
            "candidate_map": self.candidate_map
        }
        
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        
        logger.info(f"Saved FAISS index and metadata")
    
    def load_faiss_index(self, index_path: str, metadata_path: str) -> None:
        """
        Load a FAISS index and related metadata
        
        Args:
            index_path: Path to the index
            metadata_path: Path to the metadata
        """
        logger.info(f"Loading FAISS index from {index_path}")
        
        # Load index
        self.index = faiss.read_index(index_path)
        
        # Load metadata
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        
        self.candidate_ids = metadata["candidate_ids"]
        self.candidate_map = metadata["candidate_map"]
        
        logger.info(f"Loaded FAISS index with {len(self.candidate_ids)} candidates")
    
    def search(self, query_text: str, k: int = 10) -> List[Tuple[str, float, Candidate]]:
        """
        Search for similar candidates
        
        Args:
            query_text: Query text
            k: Number of results to return
            
        Returns:
            List of (candidate_id, similarity_score, candidate) tuples
        """
        if self.index is None:
            logger.warning("No index for search")
            return []
        
        logger.info(f"Searching for: {query_text}")
        
        # Generate query embedding
        query_embedding = self.model.encode([query_text], convert_to_numpy=True)
        
        # Prepare for search
        query_embedding = query_embedding.astype(np.float32)
        
        # Normalize if using cosine similarity
        if isinstance(self.index, faiss.IndexFlatIP):
            faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding, k)
        
        # Format results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < 0 or idx >= len(self.candidate_ids):
                continue
                
            candidate_id = self.candidate_ids[idx]
            score = float(scores[0][i])
            candidate = self.candidate_map[candidate_id]
            
            results.append((candidate_id, score, candidate))
        
        logger.info(f"Found {len(results)} results")
        return results
    
    def get_role_matches(self, candidates: List[Candidate], role_descriptions: Dict[str, str], top_k: int = 20) -> List[Candidate]:
        """
        Get candidates that match specific roles
        
        Args:
            candidates: List of candidates with embeddings
            role_descriptions: Dictionary of role names to descriptions
            top_k: Number of candidates to consider per role
            
        Returns:
            Candidates with role match scores added
        """
        logger.info(f"Matching candidates to {len(role_descriptions)} roles")
        
        # Build index if not already built
        if self.index is None:
            self.build_faiss_index(candidates)
        
        # For each role, find matching candidates
        for role_name, description in role_descriptions.items():
            logger.info(f"Finding matches for role: {role_name}")
            
            # Search for candidates matching this role
            matches = self.search(description, k=top_k)
            
            # Update candidate role matches
            for _, score, candidate in matches:
                candidate.role_matches[role_name] = score
                
                # Update best role if this is a better match
                if score > candidate.best_role_score:
                    candidate.best_role = role_name
                    candidate.best_role_score = score
        
        logger.info("Role matching complete")
        return candidates
