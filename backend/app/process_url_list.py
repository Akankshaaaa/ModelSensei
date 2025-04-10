import os
import re
import json
from typing import List, Dict, Any
from process_references import process_reference_url

def categorize_url(url: str) -> Dict[str, Any]:
    """Automatically categorize a URL based on its content."""
    metadata = {
        "source": url.split("//")[1].split("/")[0],
        "tags": [],
        "problem_types": [],
        "model_families": []
    }
    
    # Extract information from URL patterns
    url_lower = url.lower()
    
    # Problem types
    if any(term in url_lower for term in ["classification", "classifier"]):
        metadata["problem_types"].append("classification")
    if any(term in url_lower for term in ["regression"]):
        metadata["problem_types"].append("regression")
    if any(term in url_lower for term in ["clustering", "cluster"]):
        metadata["problem_types"].append("clustering")
    if any(term in url_lower for term in ["reinforcement"]):
        metadata["problem_types"].append("reinforcement_learning")
    if any(term in url_lower for term in ["time series", "timeseries", "forecast"]):
        metadata["problem_types"].append("time_series")
    
    # Model families
    if any(term in url_lower for term in ["tree", "forest", "boost"]):
        metadata["model_families"].append("tree_based")
    if any(term in url_lower for term in ["neural", "deep", "network"]):
        metadata["model_families"].append("neural_networks")
    if any(term in url_lower for term in ["transformer", "bert", "gpt"]):
        metadata["model_families"].append("transformers")
    if any(term in url_lower for term in ["linear", "logistic", "regression"]):
        metadata["model_families"].append("linear_models")
    
    # Special considerations
    if any(term in url_lower for term in ["explain", "interpret"]):
        metadata["tags"].append("explainability")
    if any(term in url_lower for term in ["select", "choose"]):
        metadata["tags"].append("model_selection")
    
    return metadata

def process_url_list_file(file_path: str, output_dir: str) -> List[str]:
    """Process a file containing a list of URLs."""
    os.makedirs(output_dir, exist_ok=True)
    
    processed_files = []
    
    with open(file_path, 'r') as f:
        urls = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    
    for url in urls:
        metadata = categorize_url(url)
        filepath = process_reference_url(url, output_dir, metadata)
        if filepath:
            processed_files.append(filepath)
    
    # Save a summary of processed URLs
    summary = {
        "total_processed": len(processed_files),
        "urls": [{"url": url, "metadata": categorize_url(url)} for url in urls]
    }
    
    with open(os.path.join(output_dir, "reference_summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)
    
    return processed_files

if __name__ == "__main__":
    url_list_file = "../data/models_urls.txt"
    output_dir = "../data/references"
    
    processed_files = process_url_list_file(url_list_file, output_dir)
    print(f"Processed {len(processed_files)} reference files") 