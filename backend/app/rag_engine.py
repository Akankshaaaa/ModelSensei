from typing import List, Dict, Any, Optional
import os
import json
import requests
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from .document_processor import DocumentProcessor
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class RAGEngine:
    def __init__(self, db_dir: str):
        """Initialize the RAG engine."""
        self.db_dir = db_dir
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # OpenRouter API configuration
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY not found in environment variables")
            
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"
        
        try:
            self.vectordb = Chroma(
                persist_directory=db_dir,
                embedding_function=self.embeddings
            )
            self.retriever = self.vectordb.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 5, "fetch_k": 10}
            )
        except Exception as e:
            print(f"Error initializing vector database: {str(e)}")
            self.vectordb = None
            self.retriever = None
        
        # Initialize the LLM
        try:
            self.llm = Ollama(model="llama2")
        except Exception as e:
            print(f"Error initializing LLM: {str(e)}")
            self.llm = None
        
        # Create the prompt template for model recommendation
        self.prompt_template = PromptTemplate(
            template="""You are an expert machine learning advisor. Your task is to recommend the best ML model or family of models for a given use case.

Context information from ML documentation:
{context}

User's ML problem:
Problem Type: {problem_type}
Objective: {objective}
Data Format: {data_format}
Input Dimensions: {input_dimensions}
Feature Types: {feature_types}
Data Quality Issues: {data_quality_issues}
Domain: {domain}
Computational Constraints: {computational_constraints}
Interpretability Required: {interpretability_required}
Preprocessing Level: {preprocessing_level}
Special Requirements: {special_requirements}

Based on the context information and the user's ML problem, recommend the most suitable ML model(s).
For each recommended model, provide:
1. A brief description of the model
2. Why it's suitable for this specific problem
3. Pros and cons of using this model
4. Implementation tips or considerations

Format your response as a JSON object with the following structure:
{{
  "recommendations": [
    {{
      "model_name": "Name of the model",
      "suitability_score": 0.95,
      "description": "Brief description of the model",
      "pros": ["Pro 1", "Pro 2", ...],
      "cons": ["Con 1", "Con 2", ...],
      "implementation_tips": ["Tip 1", "Tip 2", ...]
    }},
    ...
  ],
  "reasoning": "Explanation of why these models were recommended"
}}

Ensure your response is valid JSON.
""",
            input_variables=[
                "context", "problem_type", "objective", "data_format", 
                "input_dimensions", "feature_types", "data_quality_issues", 
                "domain", "computational_constraints", "interpretability_required", 
                "preprocessing_level", "special_requirements"
            ]
        )
        
    def get_completion(self, prompt: str) -> str:
        """Get completion from OpenRouter API."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "http://localhost:3000",
            "X-Title": "ML Model Recommender"
        }
        
        data = {
            "model": "openrouter/quasar-alpha",
            "messages": [
                {
                    "role": "system", 
                    "content": "You are an expert ML advisor. Always respond with valid JSON containing model recommendations."
                },
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            "temperature": 0.7,
            "max_tokens": 2000
        }
        
        try:
            response = requests.post(self.api_url, headers=headers, json=data)
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
            print(f"API Response: {content[:200]}...")  # Debug log
            return content
        except Exception as e:
            print(f"Error calling OpenRouter API: {str(e)}")
            return None

    def get_recommendations(self, query_params: Dict[str, Any]) -> Dict[str, Any]:
        """Get model recommendations based on query parameters."""
        try:
            # Get relevant documents
            nl_query = self._construct_query(query_params)
            if not self.retriever:
                raise ValueError("Retriever not initialized")
                
            docs = self.retriever.get_relevant_documents(nl_query)
            context = "\n\n".join([doc.page_content for doc in docs])
            
            # Construct the prompt
            prompt = f"""Based on the following context and requirements, recommend suitable ML models.

Context from ML documentation:
{context}

User's requirements:
- Problem Type: {query_params.get('problem_type', 'Not specified')}
- Objective: {query_params.get('objective', 'Not specified')}
- Data Format: {query_params.get('data_format', 'Not specified')}
- Feature Types: {', '.join(query_params.get('feature_types', []))}
- Data Quality Issues: {', '.join(query_params.get('data_quality_issues', []))}
- Domain: {query_params.get('domain', 'Not specified')}
- Interpretability Required: {query_params.get('interpretability_required', 'Not specified')}
- Preprocessing Level: {query_params.get('preprocessing_level', 'Not specified')}

Provide your response in the following JSON format:
{{
  "recommendations": [
    {{
      "model_name": "Name of the model",
      "suitability_score": 0.95,
      "description": "Brief description",
      "pros": ["Pro 1", "Pro 2"],
      "cons": ["Con 1", "Con 2"],
      "implementation_tips": ["Tip 1", "Tip 2"]
    }}
  ],
  "reasoning": "Explanation of recommendations"
}}"""

            # Get completion from OpenRouter
            print("Sending request to OpenRouter...")  # Debug log
            response = self.get_completion(prompt)
            
            if response:
                try:
                    result = json.loads(response)
                    print("Successfully parsed response")  # Debug log
                    return result
                except json.JSONDecodeError as e:
                    print(f"Failed to parse API response: {str(e)}")
                    return self._get_fallback_recommendation()
            else:
                print("No response from API")
                return self._get_fallback_recommendation()
                
        except Exception as e:
            print(f"Error in get_recommendations: {str(e)}")
            return self._get_fallback_recommendation()

    def _construct_query(self, params: Dict[str, Any]) -> str:
        """Construct a natural language query from parameters."""
        query_parts = [
            f"Recommend ML models for {params.get('problem_type', 'classification')} problem",
            f"with objective: {params.get('objective', 'not specified')}"
        ]
        
        # Handle multiple data formats
        if params.get('data_format'):
            formats = [
                f"{fmt['type']} ({', '.join(f'{k}={v}' for k,v in fmt['details'].items())})"
                for fmt in params['data_format']
            ]
            query_parts.append(f"using data formats: {', '.join(formats)}")
            
        if params.get('feature_types'):
            query_parts.append(f"using {', '.join(params['feature_types'])} features")
            
        if params.get('data_quality_issues'):
            query_parts.append(f"handling {', '.join(params['data_quality_issues'])} issues")
            
        return " ".join(query_parts)

    def _get_fallback_recommendation(self):
        return {
            "recommendations": [
                {
                    "model_name": "Random Forest",
                    "suitability_score": 0.9,
                    "description": "A versatile ensemble learning method that combines multiple decision trees.",
                    "pros": [
                        "Easy to use and understand",
                        "Handles both numerical and categorical features",
                        "Good performance out of the box"
                    ],
                    "cons": [
                        "Can be computationally intensive for large datasets",
                        "May require more memory than simpler models"
                    ],
                    "implementation_tips": [
                        "Start with default parameters",
                        "Use cross-validation for hyperparameter tuning",
                        "Consider feature importance for feature selection"
                    ]
                }
            ],
            "reasoning": "Fallback recommendation provided due to error in processing request."
        }

if __name__ == "__main__":
    # Example usage
    db_dir = "../data/vectordb"
    
    # Make sure the vector database exists
    if not os.path.exists(db_dir):
        data_dir = "../data/references"
        processor = DocumentProcessor(data_dir, db_dir)
        processor.process_and_index()
    
    # Initialize the RAG engine
    rag_engine = RAGEngine(db_dir)
    
    # Example query
    query_params = {
        "problem_type": "classification",
        "objective": "predict customer churn with high accuracy",
        "data_format": "tabular",
        "input_dimensions": {"samples": 10000, "features": 20},
        "feature_types": ["numerical", "categorical"],
        "data_quality_issues": ["missing values", "class imbalance"],
        "domain": "telecommunications",
        "computational_constraints": {"training_time": "not critical", "inference_time": "fast"},
        "interpretability_required": True,
        "preprocessing_level": "basic",
        "special_requirements": ["handle class imbalance"]
    }
    
    recommendations = rag_engine.get_recommendations(query_params)
    print(json.dumps(recommendations, indent=2)) 