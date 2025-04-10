from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import uvicorn
import os
import json
from .rag_engine import RAGEngine
from .document_processor import DocumentProcessor
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI(title="ML Model Recommender API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Update the data models
class DataFormatDetails(BaseModel):
    type: str
    details: Dict[str, Any]

class ModelRecommendationRequest(BaseModel):
    problem_type: str = Field(..., description="Type of ML problem (e.g., classification, regression)")
    objective: str = Field(..., description="Goal of the ML task")
    data_format: List[DataFormatDetails] = Field(..., description="Format and details of the input data")
    feature_types: List[str] = Field([], description="Types of features (e.g., numerical, categorical)")
    data_quality_issues: List[str] = Field([], description="Data quality issues (e.g., missing values, outliers)")
    domain: Optional[str] = Field(None, description="Domain of application (e.g., healthcare, finance)")
    computational_constraints: Optional[Dict[str, Any]] = Field(None, description="Computational constraints")
    interpretability_required: Optional[bool] = Field(None, description="Whether model interpretability is required")
    preprocessing_level: Optional[str] = Field(None, description="Level of preprocessing already done")
    special_requirements: List[str] = Field([], description="Special requirements (e.g., handle imbalance)")
    
class ModelRecommendation(BaseModel):
    model_name: str
    suitability_score: float
    description: str
    pros: List[str]
    cons: List[str]
    implementation_tips: List[str]
    
    model_config = {
        'protected_namespaces': ()
    }

class ModelRecommendationResponse(BaseModel):
    recommendations: List[ModelRecommendation]
    reasoning: str

# Dependency to get the RAG engine
def get_rag_engine():
    db_dir = "../data/vectordb"
    
    # Check if vector database exists, if not create it
    if not os.path.exists(db_dir):
        data_dir = "../data/references"
        processor = DocumentProcessor(data_dir, db_dir)
        processor.process_and_index()
    
    return RAGEngine(db_dir)

@app.get("/")
def read_root():
    return {"message": "Welcome to the ML Model Recommender API"}

@app.post("/recommend", response_model=ModelRecommendationResponse)
async def recommend_models(
    request: ModelRecommendationRequest,
    rag_engine: RAGEngine = Depends(get_rag_engine)
):
    """Get ML model recommendations based on problem specifications."""
    try:
        # Convert request to format expected by RAG engine
        query_params = request.dict()
        
        # Get recommendations from RAG engine
        result = rag_engine.get_recommendations(query_params)
        
        return JSONResponse(
            content=result,
            headers={
                "Access-Control-Allow-Origin": "http://localhost:3000",
                "Access-Control-Allow-Methods": "POST, OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type",
            }
        )
    except Exception as e:
        print(f"Error in get_recommendations: {str(e)}")
        # Return a more graceful error response
        return JSONResponse(
            status_code=500,
            content={
                "detail": "Internal server error occurred while generating recommendations.",
                "error": str(e)
            },
            headers={
                "Access-Control-Allow-Origin": "http://localhost:3000",
                "Access-Control-Allow-Methods": "POST, OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type",
            }
        )

@app.get("/problem-types")
def get_problem_types():
    """Get a list of supported problem types."""
    return {
        "problem_types": [
            "classification",
            "regression",
            "clustering",
            "time_series",
            "natural_language_processing",
            "computer_vision",
            "recommendation",
            "anomaly_detection",
            "reinforcement_learning"
        ]
    }

@app.get("/data-formats")
def get_data_formats():
    """Get a list of supported data formats."""
    return {
        "data_formats": [
            "tabular",
            "time_series",
            "text",
            "image",
            "audio",
            "video",
            "graph",
            "mixed"
        ]
    }

@app.get("/feature-types")
def get_feature_types():
    """Get a list of supported feature types."""
    return {
        "feature_types": [
            "numerical",
            "categorical",
            "ordinal",
            "binary",
            "text",
            "image",
            "datetime",
            "geospatial"
        ]
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 