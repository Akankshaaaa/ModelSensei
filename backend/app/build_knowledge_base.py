import os
import shutil
from document_processor import DocumentProcessor
from process_url_list import process_url_list_file

def build_knowledge_base(url_list_file, references_dir, vectordb_dir, rebuild=False):
    """Build the knowledge base from URLs and create vector database."""
    # Step 1: Process URLs and create reference documents
    if rebuild and os.path.exists(references_dir):
        print(f"Removing existing references directory: {references_dir}")
        shutil.rmtree(references_dir)
    
    os.makedirs(references_dir, exist_ok=True)
    
    # Ensure the URL list file exists
    if not os.path.exists(url_list_file):
        raise FileNotFoundError(f"URL list file not found at {url_list_file}. Please make sure the file exists.")
    
    print(f"Processing URLs from {url_list_file}")
    processed_files = process_url_list_file(url_list_file, references_dir)
    print(f"Processed {len(processed_files)} reference files")
    
    # Step 2: Create vector database from reference documents
    if rebuild and os.path.exists(vectordb_dir):
        print(f"Removing existing vector database: {vectordb_dir}")
        shutil.rmtree(vectordb_dir)
    
    print(f"Creating vector database at {vectordb_dir}")
    processor = DocumentProcessor(references_dir, vectordb_dir)
    vectordb = processor.process_and_index()
    
    print("Knowledge base built successfully!")
    return vectordb

if __name__ == "__main__":
    # Define paths
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    url_list_file = os.path.join(data_dir, "models_urls.txt")
    references_dir = os.path.join(data_dir, "references")
    vectordb_dir = os.path.join(data_dir, "vectordb")
    
    # Create data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    
    # Look for models_urls.txt in various locations
    possible_locations = [
        url_list_file,  # ../data/models_urls.txt
        os.path.join(project_root, "models_urls.txt"),  # ../../models_urls.txt
        os.path.join(os.path.dirname(__file__), "models_urls.txt")  # ./models_urls.txt
    ]
    
    source_file = None
    for location in possible_locations:
        if os.path.exists(location):
            source_file = location
            break
    
    if source_file and source_file != url_list_file:
        print(f"Copying models_urls.txt from {source_file} to {url_list_file}")
        shutil.copy(source_file, url_list_file)
    elif not os.path.exists(url_list_file):
        # If the file doesn't exist anywhere, create a simple version with a few URLs
        print(f"Creating a basic models_urls.txt file at {url_list_file}")
        with open(url_list_file, 'w') as f:
            f.write("\n".join([
                "https://scikit-learn.org/stable/user_guide.html",
                "https://www.tensorflow.org/guide",
                "https://pytorch.org/tutorials/",
                "https://en.wikipedia.org/wiki/Machine_learning"
            ]))
    
    try:
        build_knowledge_base(url_list_file, references_dir, vectordb_dir, rebuild=True)
    except Exception as e:
        print(f"Error building knowledge base: {e}")
        print("\nTroubleshooting tips:")
        print("1. Make sure models_urls.txt exists in one of these locations:")
        for loc in possible_locations:
            print(f"   - {loc}")
        print("2. Check that you have write permissions to the data directory")
        print("3. Ensure all required packages are installed") 