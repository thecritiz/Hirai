"""
Hiring Orchestrator for 100B Jobs
A multi-agent system to select the best team of 5 candidates
"""

from typing import List, Dict, Tuple
import json
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---- Agent Implementations ----

class ParserAgent:
    """Loads and cleans raw JSON candidate data"""
    
    def __call__(self, raw_json_path: str) -> List[Dict]:
        """
        Load and clean raw JSON. 
        Returns a list of profile dicts:
          [{ 'id': str, 'metadata': {...} }, ...]
        """
        with open(raw_json_path, 'r') as f:
            raw_data = json.load(f)
            
        profiles = []
        for i, candidate in enumerate(raw_data):
            # Generate unique ID
            candidate_id = f"candidate_{i}"
            
            # Extract education details
            education = candidate.get('education', {})
            highest_level = education.get('highest_level', '')
            degrees = education.get('degrees', [])
            schools = [degree.get('originalSchool', '') for degree in degrees]
            subjects = [degree.get('subject', '') for degree in degrees]
            gpas = [degree.get('gpa', '') for degree in degrees if degree.get('gpa')]
            is_top50 = any(degree.get('isTop50', False) for degree in degrees)
            is_top25 = any(degree.get('isTop25', False) for degree in degrees)
            
            # Extract work experience details
            experiences = candidate.get('work_experiences', [])
            companies = [exp.get('company', '') for exp in experiences]
            roles = [exp.get('roleName', '') for exp in experiences]
            
            # Infer role categories
            role_categories = self._categorize_roles(roles)
            
            # Create clean profile
            profile = {
                'id': candidate_id,
                'metadata': {
                    'name': candidate.get('name', ''),
                    'email': candidate.get('email', ''),
                    'location': candidate.get('location', ''),
                    'submitted_at': candidate.get('submitted_at', ''),
                    'work_availability': candidate.get('work_availability', []),
                    'annual_salary_expectation': candidate.get('annual_salary_expectation', {}).get('full-time', ''),
                    'education': {
                        'highest_level': highest_level,
                        'schools': schools,
                        'subjects': subjects,
                        'gpas': gpas,
                        'is_top50': is_top50,
                        'is_top25': is_top25
                    },
                    'experience': {
                        'companies': companies,
                        'roles': roles,
                        'role_categories': role_categories,
                        'experience_years': len(experiences) * 1.5  # Rough estimate
                    },
                    'skills': candidate.get('skills', [])
                },
                'content_for_embedding': ''  # Will be filled by FeatureAgent
            }
            profiles.append(profile)
            
        print(f"ParserAgent: Processed {len(profiles)} candidate profiles")
        return profiles
    
    def _categorize_roles(self, roles: List[str]) -> Dict[str, int]:
        """Categorize roles into high-level categories and count occurrences"""
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
            role_lower = role.lower()
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


class FeatureAgent:
    """Extracts features and generates embeddings for candidates"""
    
    def __call__(self, profiles: List[Dict]) -> List[Dict]:
        """
        For each profile, add:
          - summary (2–3 sent extractive)
          - keyword_list
          - embedding vector
        Returns list of enriched profile dicts.
        """
        enriched_profiles = []
        
        for profile in profiles:
            # Generate summary
            summary = self._generate_summary(profile)
            
            # Extract keywords
            keywords = self._extract_keywords(profile)
            
            # Prepare content for embedding
            content_for_embedding = self._prepare_embedding_content(profile, summary, keywords)
            
            # Update profile with new features
            enriched_profile = profile.copy()
            enriched_profile['summary'] = summary
            enriched_profile['keywords'] = keywords
            enriched_profile['content_for_embedding'] = content_for_embedding
            
            enriched_profiles.append(enriched_profile)
            
        print(f"FeatureAgent: Enriched {len(enriched_profiles)} profiles")
        return enriched_profiles
    
    def _generate_summary(self, profile: Dict) -> str:
        """Generate a concise summary of the candidate"""
        metadata = profile['metadata']
        name = metadata['name']
        highest_edu = metadata['education']['highest_level']
        exp_years = metadata['experience']['experience_years']
        top_roles = self._get_top_roles(metadata['experience']['role_categories'])
        skills = ', '.join(metadata['skills']) if metadata['skills'] else "No listed skills"
        
        summary = f"{name} has {highest_edu} education with approximately {exp_years:.1f} years of experience. "
        summary += f"Primary areas: {top_roles}. "
        summary += f"Skills include {skills}."
        
        return summary
    
    def _get_top_roles(self, role_categories: Dict[str, int], top_n: int = 2) -> str:
        """Get top role categories by count"""
        sorted_roles = sorted(role_categories.items(), key=lambda x: x[1], reverse=True)
        top_roles = [role.replace('_', ' ') for role, count in sorted_roles[:top_n] if count > 0]
        if not top_roles:
            return "varied roles"
        return ' and '.join(top_roles)
    
    def _extract_keywords(self, profile: Dict) -> List[str]:
        """Extract keywords from profile"""
        keywords = set()
        
        # Add skills
        skills = profile['metadata'].get('skills', [])
        keywords.update(skills)
        
        # Add education subjects
        subjects = profile['metadata']['education'].get('subjects', [])
        keywords.update([s for s in subjects if s])
        
        # Add role titles (split multi-word titles)
        roles = profile['metadata']['experience'].get('roles', [])
        for role in roles:
            words = role.split()
            for word in words:
                if len(word) > 3:  # Skip short words
                    keywords.add(word)
        
        return list(keywords)
    
    def _prepare_embedding_content(self, profile: Dict, summary: str, keywords: List[str]) -> str:
        """Prepare text content to be embedded"""
        metadata = profile['metadata']
        
        content = f"{summary}\n\n"
        content += f"Education: {metadata['education']['highest_level']}\n"
        content += f"Schools: {', '.join(metadata['education']['schools'])}\n"
        content += f"Subjects: {', '.join([s for s in metadata['education']['subjects'] if s])}\n\n"
        content += f"Experience: {', '.join(metadata['experience']['roles'])}\n"
        content += f"Skills: {', '.join(metadata['skills'])}\n"
        content += f"Keywords: {', '.join(keywords)}"
        
        return content


class EmbeddingAgent:
    """Handles embeddings generation and similarity calculation"""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=100)
        self.profile_embeddings = None
        self.id_to_index = {}
    
    def build_index(self, profiles: List[Dict]):
        """Build TF-IDF embeddings for all profiles"""
        # Extract content for embedding
        contents = [p['content_for_embedding'] for p in profiles]
        
        # Generate TF-IDF vectors
        self.profile_embeddings = self.vectorizer.fit_transform(contents)
        
        # Create ID to index mapping
        self.id_to_index = {p['id']: i for i, p in enumerate(profiles)}
        
        print(f"EmbeddingAgent: Built index with {len(profiles)} profiles")
    
    def get_similarity(self, query_text: str, top_k: int = 30) -> List[Tuple[str, float]]:
        """Get top_k profiles most similar to query text"""
        # Generate query embedding
        query_embedding = self.vectorizer.transform([query_text])
        
        # Calculate cosine similarity
        similarities = cosine_similarity(query_embedding, self.profile_embeddings)[0]
        
        # Create ID-score pairs
        id_scores = [(pid, float(similarities[idx])) for pid, idx in self.id_to_index.items()]
        
        # Sort by similarity score (descending)
        id_scores.sort(key=lambda x: x[1], reverse=True)
        
        return id_scores[:top_k]


class RerankingAgent:
    """Reranks candidates based on similarity and other factors"""
    
    def __init__(self):
        self.embedding_agent = EmbeddingAgent()
        self.role_definitions = {
            "Engineering Lead": (
                "Senior engineer with strong technical and leadership skills. "
                "Experience with system architecture, team management, and full-stack development. "
                "Strong problem-solving abilities and communication skills."
            ),
            "Product Manager": (
                "Strategic product thinker with experience defining product vision and roadmap. "
                "Data-driven decision maker with strong user empathy. Background in product development "
                "and cross-functional team coordination."
            ),
            "Senior Developer": (
                "Experienced software engineer with strong coding skills. Deep knowledge of "
                "software development practices, architecture patterns, and technical implementation. "
                "Experience with multiple languages and frameworks."
            ),
            "Marketing/Growth Lead": (
                "Data-driven marketer with experience growing user acquisition and retention. "
                "Strategic thinker with background in digital marketing, analytics, and growth hacking. "
                "Creative problem solver with strong communication skills."
            ),
            "Finance/Operations": (
                "Experienced in financial planning, accounting, and operations management. "
                "Strong analytical skills and attention to detail. Background in financial reporting, "
                "budgeting, and operational efficiency."
            )
        }
        
    def __call__(self, profiles: List[Dict], top_n: int) -> List[Dict]:
        """
        Inputs: all enriched profiles.
        Returns: top_n profiles, sorted by descending score.
        """
        # Build embedding index
        self.embedding_agent.build_index(profiles)
        
        # Get candidates for each role
        role_candidates = {}
        for role, description in self.role_definitions.items():
            # Get similar candidates to role description
            similar_ids = self.embedding_agent.get_similarity(description, top_k=top_n)
            
            # Map back to profiles
            id_to_profile = {p['id']: p for p in profiles}
            candidates = []
            for cand_id, sim_score in similar_ids:
                profile = id_to_profile[cand_id].copy()
                profile['role_match'] = role
                profile['sim_score'] = sim_score
                candidates.append(profile)
                
            role_candidates[role] = candidates
            
        # Combine and adjust scores
        all_candidates = []
        for role, candidates in role_candidates.items():
            for i, candidate in enumerate(candidates):
                # Add position-based boost (earlier = better)
                position_boost = 0.05 * (len(candidates) - i) / len(candidates)
                
                # Add education boost
                edu_boost = 0
                if candidate['metadata']['education']['is_top25']:
                    edu_boost += 0.1
                elif candidate['metadata']['education']['is_top50']:
                    edu_boost += 0.05
                    
                # Add experience boost
                exp_years = candidate['metadata']['experience']['experience_years']
                exp_boost = min(0.15, 0.02 * exp_years)
                
                # Calculate rerank score
                rerank_score = candidate['sim_score'] + position_boost + edu_boost + exp_boost
                candidate['rerank_score'] = rerank_score
                candidate['score_components'] = {
                    'similarity': candidate['sim_score'],
                    'position': position_boost,
                    'education': edu_boost,
                    'experience': exp_boost
                }
                
                all_candidates.append(candidate)
                
        # Sort by rerank score
        all_candidates.sort(key=lambda x: x['rerank_score'], reverse=True)
        
        # Get top candidates
        top_candidates = all_candidates[:top_n]
        
        print(f"RerankingAgent: Selected top {len(top_candidates)} candidates")
        return top_candidates


class DiversityAgent:
    """Ensures diversity in the final team selection"""
    
    def __call__(self, candidates: List[Dict], team: List[Dict], team_size: int) -> List[Dict]:
        """
        Greedy marginal-gain selection:
          Pick `team_size` profiles from `candidates`, given already-picked `team`.
        Returns final team list (length == team_size).
        """
        # Start with any team members already selected
        final_team = team.copy()
        
        # Maintain copy of candidates that excludes any already in the team
        team_ids = {member['id'] for member in team}
        available_candidates = [c for c in candidates if c['id'] not in team_ids]
        
        # Sort candidates by initial score for starting point
        available_candidates.sort(key=lambda x: x['rerank_score'], reverse=True)
        
        # Keep track of key diversity dimensions
        roles_covered = {member.get('role_match', '') for member in team if 'role_match' in member}
        locations = {member['metadata']['location'] for member in team}
        education_backgrounds = set()
        for member in team:
            education_backgrounds.update(member['metadata']['education']['subjects'])
        
        # Helper to calculate diversity score
        def calculate_diversity_gain(candidate, current_team):
            score = 0
            
            # Role diversity
            role = candidate.get('role_match', '')
            if role and role not in roles_covered:
                score += 20  # Major boost for new role
                
            # Location diversity
            location = candidate['metadata']['location']
            if location not in locations:
                score += 10
                
            # Education diversity
            for subject in candidate['metadata']['education']['subjects']:
                if subject and subject not in education_backgrounds:
                    score += 5
                    
            # Skill complementarity
            candidate_skills = set(candidate['metadata']['skills'])
            team_skills = set()
            for member in current_team:
                team_skills.update(member['metadata']['skills'])
            
            new_skills = candidate_skills - team_skills
            score += len(new_skills) * 3
            
            return score
        
        # Greedy selection
        while len(final_team) < team_size and available_candidates:
            best_candidate = None
            best_score = -1
            
            for candidate in available_candidates:
                # Base score from reranking
                base_score = candidate['rerank_score'] * 100  # Scale up for easier comparisons
                
                # Diversity gain
                diversity_gain = calculate_diversity_gain(candidate, final_team)
                
                # Combined score
                total_score = base_score + diversity_gain
                
                if total_score > best_score:
                    best_score = total_score
                    best_candidate = candidate
            
            if best_candidate:
                # Add diversity metrics to the candidate
                best_candidate['diversity_score'] = best_score - (best_candidate['rerank_score'] * 100)
                
                # Add to team
                final_team.append(best_candidate)
                
                # Update diversity tracking
                if 'role_match' in best_candidate:
                    roles_covered.add(best_candidate['role_match'])
                locations.add(best_candidate['metadata']['location'])
                for subject in best_candidate['metadata']['education']['subjects']:
                    if subject:
                        education_backgrounds.add(subject)
                
                # Remove from available candidates
                available_candidates.remove(best_candidate)
        
        print(f"DiversityAgent: Selected final team of {len(final_team)} members")
        return final_team


class JustifierAgent:
    """Generates justifications for selected candidates"""
    
    def __call__(self, candidate: Dict, team: List[Dict]) -> str:
        """
        Given one candidate and the current team context,
        return a 2–3 sentence rationale string.
        """
        # Basic candidate info
        name = candidate['metadata']['name']
        role = candidate.get('role_match', 'team member')
        
        # Extract education highlights
        education = []
        if candidate['metadata']['education']['is_top25']:
            education.append("top-25 school graduate")
        elif candidate['metadata']['education']['is_top50']:
            education.append("top-50 school graduate")
            
        highest_edu = candidate['metadata']['education']['highest_level']
        if highest_edu:
            education.append(f"has {highest_edu}")
            
        subjects = [s for s in candidate['metadata']['education']['subjects'] if s]
        if subjects:
            education.append(f"studied {', '.join(subjects[:2])}")
            
        edu_str = " and ".join(education) if education else ""
        
        # Experience highlights
        exp_years = candidate['metadata']['experience']['experience_years']
        companies = candidate['metadata']['experience']['companies']
        roles = candidate['metadata']['experience']['roles']
        
        exp_str = f"has {exp_years:.1f} years of experience"
        if roles:
            top_roles = roles[:2]
            exp_str += f" including roles as {' and '.join(top_roles)}"
            
        # Skills highlights
        skills = candidate['metadata']['skills']
        skills_str = f"skilled in {', '.join(skills)}" if skills else ""
        
        # Diversity contribution
        diversity_points = []
        
        if len(team) > 0:
            # Location diversity
            team_locations = {m['metadata']['location'] for m in team}
            if candidate['metadata']['location'] not in team_locations:
                diversity_points.append("geographic diversity")
                
            # Education diversity
            team_subjects = set()
            for m in team:
                team_subjects.update(m['metadata']['education']['subjects'])
            
            new_subjects = set(subjects) - team_subjects
            if new_subjects:
                diversity_points.append("educational background diversity")
                
            # Skill complementarity
            team_skills = set()
            for m in team:
                team_skills.update(m['metadata']['skills'])
                
            new_skills = set(skills) - team_skills
            if new_skills:
                diversity_points.append("complementary skills")
        
        diversity_str = ""
        if diversity_points:
            diversity_str = f"Adds {' and '.join(diversity_points)} to the team."
        
        # Put it all together
        justification = f"Selected as {role}. "
        
        if edu_str and exp_str:
            justification += f"{name} {edu_str} and {exp_str}. "
        elif edu_str:
            justification += f"{name} {edu_str}. "
        elif exp_str:
            justification += f"{name} {exp_str}. "
            
        if skills_str:
            justification += f"{skills_str.capitalize()}. "
            
        if diversity_str:
            justification += f"{diversity_str}"
            
        return justification.strip()


# ---- Orchestrator ----

class HiringOrchestrator:
    def __init__(self,
                 parser: ParserAgent,
                 featureer: FeatureAgent,
                 reranker: RerankingAgent,
                 diverifier: DiversityAgent,
                 justifier: JustifierAgent,
                 shortlist_n: int = 30,
                 final_team_size: int = 5):
        self.parser = parser
        self.featureer = featureer
        self.reranker = reranker
        self.diverifier = diverifier
        self.justifier = justifier
        self.shortlist_n = shortlist_n
        self.final_team_size = final_team_size

    def run(self, raw_json_path: str) -> List[Tuple[Dict, str]]:
        """
        Run the full hiring pipeline:
        1. Parse JSON data
        2. Extract features
        3. Rerank candidates
        4. Select diverse team
        5. Generate justifications
        
        Returns a list of (candidate, justification) tuples
        """
        # 1️⃣ Parse
        print("\n=== 1. Parsing Candidate Data ===")
        profiles = self.parser(raw_json_path)
        
        # 2️⃣ Feature extraction
        print("\n=== 2. Extracting Features ===")
        enriched = self.featureer(profiles)
        
        # 3️⃣ Initial reranking
        print("\n=== 3. Reranking Candidates ===")
        top_candidates = self.reranker(enriched, top_n=self.shortlist_n)
        
        # 4️⃣ Diversity-based final selection
        print("\n=== 4. Building Diverse Team ===")
        final_team = self.diverifier(top_candidates, team=[], team_size=self.final_team_size)
        
        # 5️⃣ Justifications
        print("\n=== 5. Generating Justifications ===")
        team_with_reasons = []
        for i, candidate in enumerate(final_team):
            context = final_team[:i]  # Team members selected so far
            reason = self.justifier(candidate, context)
            team_with_reasons.append((candidate, reason))
            print(f"Justified {candidate['metadata']['name']} for role: {candidate.get('role_match', 'team member')}")
        
        return team_with_reasons
    
    def display_team(self, team_with_reasons: List[Tuple[Dict, str]]):
        """
        Display the selected team with justifications
        """
        print("\n" + "=" * 80)
        print("SELECTED TEAM OF 5".center(80))
        print("=" * 80 + "\n")
        
        for i, (candidate, reason) in enumerate(team_with_reasons, 1):
            name = candidate['metadata']['name']
            role = candidate.get('role_match', 'Team Member')
            location = candidate['metadata']['location']
            score = candidate.get('rerank_score', 0)
            skills = ', '.join(candidate['metadata']['skills']) if candidate['metadata']['skills'] else 'No listed skills'
            
            print(f"{i}. {name} - {role} (Score: {score:.3f})")
            print(f"   Location: {location}")
            print(f"   Skills: {skills}")
            print(f"   Justification: {reason}")
            print("-" * 80)


# ---- Main Execution ----

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("100B JOBS: AI-POWERED HIRING SYSTEM".center(80))
    print("=" * 80 + "\n")
    
    # Initialize agents
    parser = ParserAgent()
    featureer = FeatureAgent()
    reranker = RerankingAgent()
    diverifier = DiversityAgent()
    justifier = JustifierAgent()
    
    # Create orchestrator
    orchestrator = HiringOrchestrator(
        parser=parser,
        featureer=featureer,
        reranker=reranker,
        diverifier=diverifier,
        justifier=justifier,
        shortlist_n=30,
        final_team_size=5
    )
    
    # Run the pipeline
    team_with_reasons = orchestrator.run("main_data.json")
    
    # Display results
    orchestrator.display_team(team_with_reasons)
    
    print("\nHiring process complete! Selected a diverse team of 5 candidates.")
