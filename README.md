# AI-Powered Hiring System

A sophisticated hiring system that uses AI, graph algorithms, and LLMs to select optimal teams based on candidates' skills, diversity, and role-specific requirements.

## Overview

The AI-Powered Hiring System processes candidate data through a pipeline of specialized agents to select the best possible team for a startup. It combines natural language processing, graph algorithms, and large language models to evaluate candidates holistically and build well-rounded teams.

## Features

- **Data Processing**: Extract and structure candidate data from JSON/CSV files
- **Semantic Matching**: Generate embeddings for candidates to find semantic similarities
- **Graph-Based Ranking**: Build heterogeneous skill-candidate-role graphs and use PageRank for scoring
- **Diversity Optimization**: Select teams with complementary skills and diverse backgrounds
- **LLM-Powered Justifications**: Generate natural language justifications for selections using Claude
- **RESTful API**: Access all functionality through a FastAPI interface

## Architecture

### Core Components

The system is built around a modular, agent-based architecture:

1. **Orchestration Layer**:
   - `HiringOrchestrator`: Standard sequential pipeline
   - `HiringCrewOrchestrator`: Alternative using CrewAI for autonomous agents

2. **Agent System**:
   - `PreprocessingAgent`: Extract and structure candidate data
   - `EmbeddingAgent`: Generate vector embeddings for semantic matching
   - `GraphRerankerAgent`: Create heterogeneous graphs and rank using PageRank
   - `DiversityAgent`: Optimize team composition for diversity metrics
   - `JustifierAgent`: Generate natural language justifications using Claude

3. **Data Models**:
   - `Candidate`: Core candidate data structure
   - `Team`/`TeamMember`: Selected team representations
   - Various supporting models for education, experience, etc.

4. **API Layer**:
   - FastAPI application exposing core functionality
  
     
### Design:
```mermaid
classDiagram
    class HiringOrchestrator {
        -PreprocessingAgent preprocessing_agent
        -EmbeddingAgent embedding_agent
        -GraphRerankerAgent graph_agent
        -DiversityAgent diversity_agent
        -JustifierAgent justifier_agent
        -BedrockClient bedrock_client
        -List~Candidate~ candidates
        -List~Candidate~ shortlisted_candidates
        -Team final_team
        +run() Team
        +_load_and_preprocess_data()
        +_generate_embeddings()
        +_build_graph_and_rank()
        +_select_team()
        +_generate_justifications()
        +save_results()
        +display_team()
        +get_top_candidates_by_role() Dict
        +run_with_tool_use() Dict
    }

    class HiringCrewOrchestrator {
        -Agent data_agent
        -Agent embedding_crew_agent
        -Agent ranking_agent
        -Agent team_selection_agent
        -Agent justification_agent
        +run() Team
        +_create_data_agent() Agent
        +_create_embedding_agent() Agent
        +_create_ranking_agent() Agent
        +_create_team_selection_agent() Agent
        +_create_justification_agent() Agent
    }

    class PreprocessingAgent {
        +process_json_file(file_path) List~Candidate~
        +process_csv_file(file_path) List~Candidate~
        +enrich_candidates(candidates) List~Candidate~
        -_process_candidate(raw_data, id) Candidate
        -_categorize_roles(roles) Dict
        +generate_candidate_summary(candidate) String
        +extract_keywords(candidate) List~String~
    }

    class EmbeddingAgent {
        -SentenceTransformer model
        -faiss.Index index
        -List~String~ candidate_ids
        -Dict candidate_map
        +generate_embeddings(candidates) List~Candidate~
        +build_faiss_index(candidates)
        +save_faiss_index(index_path, metadata_path)
        +load_faiss_index(index_path, metadata_path)
        +search(query_text, k) List~Tuple~
        +get_role_matches(candidates, role_descriptions, top_k) List~Candidate~
    }

    class GraphRerankerAgent {
        -DiGraph graph
        -List candidate_nodes
        -List skill_nodes
        -List school_nodes
        -List role_nodes
        -Dict candidate_map
        +build_graph(candidates, role_weights) DiGraph
        -_add_candidate_similarity_edges(candidates, threshold)
        +compute_pagerank_scores(role_name, alpha, max_iter) Dict
        +rerank_candidates(candidates, role_name, top_k) List~Candidate~
        +save_graph(graph_path)
        +load_graph(graph_path)
        +get_graph_score(candidate_id, role) float
    }

    class DiversityAgent {
        -Dict diversity_weights
        +select_team(candidates, team_size, role_preference) List~Candidate~
        -_calculate_diversity_gain(candidate, team_locations, team_education, team_skills, team_roles) float
        +calculate_skill_gap(candidate_id, candidate_skills, team_skills) Dict
        +calculate_diversity_contribution(candidate_id, location, education, team_locations, team_education) Dict
        +create_team_object(selected_candidates) Team
    }

    class JustifierAgent {
        -BedrockClient bedrock_client
        +justify_candidate(candidate, team_so_far) String
        +justify_team(team) Team
        +justify_selection_with_reasoning(candidate, role, team_so_far) Dict
        +get_final_team_assessment(team) String
        +get_role_candidates(candidates, role, count) List~Dict~
    }

    class BedrockClient {
        -boto3.client client
        -String model_id
        +generate_response(prompt, system_prompt, max_tokens) String
        +generate_with_reasoning(prompt) Dict
    }

    class Candidate {
        +String id
        +CandidateMetadata metadata
        +String summary
        +List~String~ keywords
        +List~float~ embedding
        +Dict~String,float~ role_matches
        +String best_role
        +float best_role_score
        +float diversity_score
        +float final_score
        +String justification
    }

    class Team {
        +List~TeamMember~ members
        +Dict~String,float~ diversity_metrics
        +float total_score
    }

    HiringOrchestrator --* PreprocessingAgent
    HiringOrchestrator --* EmbeddingAgent
    HiringOrchestrator --* GraphRerankerAgent
    HiringOrchestrator --* DiversityAgent
    HiringOrchestrator --* JustifierAgent
    HiringOrchestrator --* BedrockClient
    HiringOrchestrator --o Candidate : processes
    HiringOrchestrator --o Team : produces

    HiringCrewOrchestrator --* PreprocessingAgent
    HiringCrewOrchestrator --* EmbeddingAgent
    HiringCrewOrchestrator --* GraphRerankerAgent
    HiringCrewOrchestrator --* DiversityAgent
    HiringCrewOrchestrator --* JustifierAgent

    JustifierAgent --* BedrockClient
    Team --* TeamMember
    TeamMember --* Candidate
```


### Sequence Flow
####Sequence Diagram[the main flow]
```mermaid
sequenceDiagram
    participant Main
    participant HO as HiringOrchestrator
    participant PA as PreprocessingAgent
    participant EA as EmbeddingAgent
    participant GA as GraphRerankerAgent
    participant DA as DiversityAgent
    participant JA as JustifierAgent
    participant BC as BedrockClient

    Main->>+HO: run()
    
    %% Step 1: Load and preprocess data
    HO->>+PA: process_json_file(data_path)
    PA-->>-HO: raw candidates
    HO->>+PA: enrich_candidates(candidates)
    PA-->>-HO: enriched candidates
    
    %% Step 2: Generate embeddings
    HO->>+EA: generate_embeddings(candidates)
    EA-->>-HO: candidates with embeddings
    HO->>+EA: build_faiss_index(candidates)
    EA-->>-HO: index built
    
    %% Step 3: Build graph and rank candidates
    HO->>+GA: build_graph(candidates)
    GA-->>-HO: graph built
    
    loop For each role
        HO->>+GA: rerank_candidates(candidates, role, shortlist_size)
        GA->>GA: compute_pagerank_scores(role)
        GA-->>-HO: ranked candidates for role
    end
    
    %% Step 4: Select team
    HO->>+DA: select_team(shortlisted_candidates, team_size, true)
    loop For each candidate
        DA->>DA: _calculate_diversity_gain(candidate, ...)
    end
    DA-->>-HO: selected candidates
    
    %% Step 5: Generate justifications
    HO->>+DA: create_team_object(selected_candidates)
    DA-->>-HO: team object
    
    HO->>+JA: justify_team(team)
    loop For each team member
        JA->>+BC: generate_response(candidate_prompt, system_prompt)
        BC-->>-JA: justification
    end
    JA-->>-HO: team with justifications
    
    %% Step 6: Save and display results
    HO->>HO: save_results(output_path)
    HO->>HO: display_team()
    
    HO-->>-Main: final team

```


####API Sequence Flow
```mermaid
sequenceDiagram
    participant Client
    participant API
    participant BG as BackgroundTasks
    participant HO as HiringOrchestrator
    
    %% Get top candidates endpoint
    Client->>+API: POST /top-candidates
    API->>+HO: Create HiringOrchestrator
    API->>+HO: _load_and_preprocess_data()
    HO-->>-API: candidates loaded
    API->>+HO: _generate_embeddings()
    HO-->>-API: embeddings generated
    API->>+HO: _build_graph_and_rank()
    HO-->>-API: graph built
    API->>+HO: get_top_candidates_by_role(count)
    HO-->>-API: top candidates per role
    API-->>-Client: top candidates JSON
    
    %% Run hiring pipeline endpoint
    Client->>+API: POST /run-hiring-pipeline
    API->>+BG: add_task(run_pipeline)
    API-->>-Client: job_id
    
    BG->>+HO: Create HiringOrchestrator
    BG->>+HO: run()
    Note over HO: Full pipeline execution
    HO-->>-BG: final team
    BG->>BG: Update job status & results
    
    %% Check job status
    Client->>+API: GET /job/{job_id}
    API-->>-Client: job status & results

```

## Installation

1. Clone this repository
```bash
git clone https://github.com/your-username/ai-hiring-system.git
cd ai-hiring-system
```

2. Create and activate a virtual environment
```bash
python -m venv hiring_venv
source hiring_venv/bin/activate  # On Windows: hiring_venv\Scripts\activate
```

3. Install dependencies
```bash
pip install -e .
```

4. Configure AWS credentials (for Claude access)
```bash
aws configure
```

## Usage

### Command Line Interface

Run the standard hiring pipeline:

```bash
python -m hiring_system.main --data path/to/candidates.json --team-size 5
```

Get top candidates for each role:

```bash
python -m hiring_system.main --data path/to/candidates.json --top-candidates --top-n 3
```

Use Claude's tool use for final selection:

```bash
python -m hiring_system.main --data path/to/candidates.json --use-tool-use
```

### API

Start the API server:

```bash
python run_api.py
```

Example API requests:

```bash
# Get top candidates for each role
curl -X POST "http://localhost:8000/top-candidates" \
  -H "Content-Type: application/json" \
  -d '{"data_path": "path/to/candidates.json", "candidates_per_role": 3}'

# Run the full hiring pipeline
curl -X POST "http://localhost:8000/run-hiring-pipeline" \
  -H "Content-Type: application/json" \
  -d '{"data_path": "path/to/candidates.json", "team_size": 5}'
```

## Directory Structure

```
hiring_system/
├── __init__.py
├── api.py          # FastAPI application
├── main.py         # Command-line interface
├── requirements.txt
├── agents/         # Specialized agents
│   ├── __init__.py
│   ├── preprocessing_agent.py
│   ├── embedding_agent.py
│   ├── graph_reranker_agent.py
│   ├── diversity_agent.py
│   └── justifier_agent.py
├── config/         # Configuration
│   ├── __init__.py
│   └── settings.py
├── pipeline/       # Orchestration
│   ├── __init__.py
│   └── orchestrator.py
└── utils/          # Utilities
    ├── __init__.py
    ├── bedrock_client.py
    ├── models.py
    ├── tool_handlers.py
    └── tool_specs.py
```

## Technologies Used

- **NLP**: SentenceTransformer for embedding generation
- **Graph Algorithms**: NetworkX for graph construction and PageRank scoring
- **Vector Search**: FAISS for similarity searches
- **LLM Integration**: AWS Bedrock (Claude) for justification generation
- **API Framework**: FastAPI for exposing functionality
- **Agent Frameworks**: Optional CrewAI integration
