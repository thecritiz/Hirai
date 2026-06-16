"""
Graph Reranking Agent that builds a heterogeneous graph and applies PageRank
for candidate scoring
"""

import logging
import networkx as nx
import numpy as np
from typing import Dict, List, Set, Tuple, Any, Optional
from pathlib import Path
import pickle

from hiring_system.utils.models import Candidate
from hiring_system.config.settings import ROLE_DEFINITIONS

logger = logging.getLogger(__name__)


class GraphRerankerAgent:
    """
    Agent for building a heterogeneous graph and scoring candidates using PageRank
    """
    
    def __init__(self):
        """Initialize the graph reranker agent"""
        logger.info("Initializing GraphRerankerAgent")
        self.graph = None
        self.candidate_nodes = []
        self.skill_nodes = []
        self.school_nodes = []
        self.role_nodes = []
        self.candidate_map = {}  # Maps candidate IDs to candidates
    
    def build_graph(self, candidates: List[Candidate], role_weights: Optional[Dict[str, Dict[str, float]]] = None) -> nx.DiGraph:
        """
        Build a heterogeneous graph from candidates
        
        Args:
            candidates: List of candidate objects
            role_weights: Optional dictionary of role weights for skills
            
        Returns:
            NetworkX DiGraph
        """
        logger.info(f"Building graph with {len(candidates)} candidates")
        
        # Initialize graph
        self.graph = nx.DiGraph()
        self.candidate_nodes = []
        self.skill_nodes = []
        self.school_nodes = []
        self.role_nodes = []
        self.candidate_map = {}
        
        # Add role nodes
        for role_name in ROLE_DEFINITIONS.keys():
            role_node = f"ROLE:{role_name}"
            self.graph.add_node(role_node, type="role", name=role_name)
            self.role_nodes.append(role_node)
        
        # Gather all skills and schools
        all_skills = set()
        all_schools = set()
        
        for candidate in candidates:
            # Add to skills set
            all_skills.update(candidate.metadata.skills)
            
            # Add to schools set
            all_schools.update(candidate.metadata.education.schools)
        
        # Add skill nodes
        for skill in all_skills:
            skill_node = f"SKILL:{skill}"
            self.graph.add_node(skill_node, type="skill", name=skill)
            self.skill_nodes.append(skill_node)
        
        # Add school nodes
        for school in all_schools:
            school_node = f"SCHOOL:{school}"
            self.graph.add_node(school_node, type="school", name=school)
            self.school_nodes.append(school_node)
        
        # Add candidate nodes and edges
        for candidate in candidates:
            # Add candidate node
            candidate_node = f"CANDIDATE:{candidate.id}"
            self.graph.add_node(candidate_node, type="candidate", id=candidate.id, 
                              name=candidate.metadata.name)
            self.candidate_nodes.append(candidate_node)
            self.candidate_map[candidate.id] = candidate
            
            # Add candidate-skill edges
            for skill in candidate.metadata.skills:
                skill_node = f"SKILL:{skill}"
                self.graph.add_edge(candidate_node, skill_node, weight=1.0)
                self.graph.add_edge(skill_node, candidate_node, weight=0.5)  # Lower weight for reverse direction
            
            # Add candidate-school edges
            for school in candidate.metadata.education.schools:
                school_node = f"SCHOOL:{school}"
                weight = 1.5 if candidate.metadata.education.is_top25 else 1.0
                self.graph.add_edge(candidate_node, school_node, weight=weight)
                self.graph.add_edge(school_node, candidate_node, weight=weight * 0.5)  # Lower weight for reverse
        
        # Add role-skill edges
        if role_weights is None:
            # Use default role weights from settings
            role_weights = {}
            for role_name, role_def in ROLE_DEFINITIONS.items():
                skill_weights = {}
                for skill in role_def.get('required_skills', []):
                    skill_weights[skill] = 1.0  # Default weight
                role_weights[role_name] = skill_weights
        
        for role_name, skill_weights in role_weights.items():
            role_node = f"ROLE:{role_name}"
            
            for skill, weight in skill_weights.items():
                skill_node = f"SKILL:{skill}"
                
                # Add edge only if skill exists in the graph
                if skill_node in self.skill_nodes:
                    self.graph.add_edge(role_node, skill_node, weight=weight)
                    self.graph.add_edge(skill_node, role_node, weight=weight * 0.5)  # Lower weight for reverse
        
        # Add candidate-candidate edges based on similarity
        self._add_candidate_similarity_edges(candidates)
        
        logger.info(f"Built graph with {len(self.graph.nodes)} nodes and {len(self.graph.edges)} edges")
        return self.graph
    
    def _add_candidate_similarity_edges(self, candidates: List[Candidate], threshold: float = 0.5) -> None:
        """
        Add edges between similar candidates
        
        Args:
            candidates: List of candidates
            threshold: Similarity threshold for adding edges
        """
        # Only connect candidates if they have embeddings
        candidates_with_embeddings = [c for c in candidates if c.embedding is not None]
        
        if len(candidates_with_embeddings) < 2:
            logger.warning("Not enough candidates with embeddings to compute similarities")
            return
        
        # Compute pairwise similarities
        for i, candidate1 in enumerate(candidates_with_embeddings):
            embed1 = np.array(candidate1.embedding)
            
            # Normalize embedding
            embed1 = embed1 / np.linalg.norm(embed1)
            
            for j, candidate2 in enumerate(candidates_with_embeddings[i+1:], i+1):
                embed2 = np.array(candidate2.embedding)
                
                # Normalize embedding
                embed2 = embed2 / np.linalg.norm(embed2)
                
                # Compute cosine similarity
                similarity = float(np.dot(embed1, embed2))
                
                # Add edge if similarity is above threshold
                if similarity > threshold:
                    node1 = f"CANDIDATE:{candidate1.id}"
                    node2 = f"CANDIDATE:{candidate2.id}"
                    
                    self.graph.add_edge(node1, node2, weight=similarity)
                    self.graph.add_edge(node2, node1, weight=similarity)
    
    def compute_pagerank_scores(self, role_name: str, alpha: float = 0.15, max_iter: int = 100) -> Dict[str, float]:
        """
        Compute PageRank scores with bias towards a specific role
        
        Args:
            role_name: Role to use as bias
            alpha: Teleportation probability (restart probability)
            max_iter: Maximum number of iterations
            
        Returns:
            Dictionary of candidate IDs to scores
        """
        if self.graph is None:
            logger.warning("Graph not built")
            return {}
        
        logger.info(f"Computing PageRank scores for role {role_name}")
        
        # Create personalization vector with bias towards role
        personalization = {}
        
        # Set bias towards role
        role_node = f"ROLE:{role_name}"
        if role_node in self.graph:
            for node in self.graph.nodes():
                if node == role_node:
                    personalization[node] = 1.0
                else:
                    personalization[node] = 0.0
        
        # Compute PageRank
        pagerank_scores = nx.pagerank(self.graph, alpha=alpha, personalization=personalization, 
                                     max_iter=max_iter, weight='weight')
        
        # Extract candidate scores
        candidate_scores = {}
        for node, score in pagerank_scores.items():
            if node.startswith("CANDIDATE:"):
                candidate_id = node.split(":", 1)[1]
                candidate_scores[candidate_id] = score
        
        logger.info(f"Computed PageRank scores for {len(candidate_scores)} candidates")
        return candidate_scores
    
    def rerank_candidates(self, candidates: List[Candidate], role_name: str, 
                        top_k: Optional[int] = None) -> List[Candidate]:
        """
        Rerank candidates based on PageRank scores for a role
        
        Args:
            candidates: List of candidates
            role_name: Role to use for scoring
            top_k: Optional limit on number of candidates to return
            
        Returns:
            Reranked candidates with graph scores
        """
        # Build graph if not already built
        if self.graph is None:
            self.build_graph(candidates)
        
        # Compute scores
        scores = self.compute_pagerank_scores(role_name)
        
        # Update candidates with scores
        reranked = []
        for candidate in candidates:
            score = scores.get(candidate.id, 0.0)
            
            # Update candidate's role matches and scores
            if role_name not in candidate.role_matches or score > candidate.role_matches[role_name]:
                candidate.role_matches[role_name] = score
            
            # Update best role if this is a better match
            if score > candidate.best_role_score:
                candidate.best_role = role_name
                candidate.best_role_score = score
                
            reranked.append(candidate)
        
        # Sort by score
        reranked.sort(key=lambda x: x.role_matches[role_name], reverse=True)
        
        # Limit to top_k if specified
        if top_k is not None:
            reranked = reranked[:top_k]
        
        return reranked
    
    def save_graph(self, graph_path: str) -> None:
        """
        Save the graph and related data
        
        Args:
            graph_path: Path to save the graph
        """
        if self.graph is None:
            logger.warning("No graph to save")
            return
        
        logger.info(f"Saving graph to {graph_path}")
        
        # Prepare data for saving
        data = {
            "graph": self.graph,
            "candidate_nodes": self.candidate_nodes,
            "skill_nodes": self.skill_nodes,
            "school_nodes": self.school_nodes,
            "role_nodes": self.role_nodes,
            "candidate_map": self.candidate_map
        }
        
        # Save
        with open(graph_path, 'wb') as f:
            pickle.dump(data, f)
        
        logger.info(f"Saved graph with {len(self.graph.nodes)} nodes")
    
    def load_graph(self, graph_path: str) -> None:
        """
        Load a saved graph and related data
        
        Args:
            graph_path: Path to the saved graph
        """
        logger.info(f"Loading graph from {graph_path}")
        
        # Load
        with open(graph_path, 'rb') as f:
            data = pickle.load(f)
        
        # Restore data
        self.graph = data["graph"]
        self.candidate_nodes = data["candidate_nodes"]
        self.skill_nodes = data["skill_nodes"]
        self.school_nodes = data["school_nodes"]
        self.role_nodes = data["role_nodes"]
        self.candidate_map = data["candidate_map"]
        
        logger.info(f"Loaded graph with {len(self.graph.nodes)} nodes")
    
    def get_graph_score(self, candidate_id: str, role: str) -> float:
        """
        Get the graph-based score for a candidate and role
        
        Args:
            candidate_id: Candidate ID
            role: Role name
            
        Returns:
            Score between 0 and 1
        """
        if self.graph is None:
            logger.warning("Graph not built")
            return 0.0
        
        # Compute pagerank scores if needed
        if not self.candidate_map:
            logger.warning("No candidates in graph")
            return 0.0
        
        candidate = self.candidate_map.get(candidate_id)
        if not candidate:
            logger.warning(f"Candidate {candidate_id} not found in graph")
            return 0.0
        
        # Get score from role matches, or compute if not available
        if role in candidate.role_matches:
            return candidate.role_matches[role]
        else:
            # Compute PageRank scores for this role
            scores = self.compute_pagerank_scores(role)
            return scores.get(candidate_id, 0.0)
