import os
import json
from typing import List, Dict, Any
import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document
import html2text

# Make sure paths are relative to the backend directory
DATA_DIR = "../data"  # If running from app directory
# or
# DATA_DIR = "data"     # If running from backend directory

# Add directory creation
os.makedirs(f"{DATA_DIR}/references", exist_ok=True)
os.makedirs(f"{DATA_DIR}/vectordb", exist_ok=True)

class DocumentProcessor:
    def __init__(self, data_dir: str, db_dir: str):
        """Initialize the document processor.
        
        Args:
            data_dir: Directory containing reference markdown files
            db_dir: Directory to store the vector database
        """
        self.data_dir = data_dir
        self.db_dir = db_dir
        self.embeddings = HuggingFaceEmbeddings()
        
    def load_documents(self) -> List[Document]:
        documents = []
        
        if not os.path.exists(self.data_dir):
            print(f"Data directory not found: {self.data_dir}")
            return documents
            
        for filename in os.listdir(self.data_dir):
            if filename.endswith(('.md', '.txt')):
                file_path = os.path.join(self.data_dir, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8') as file:
                        content = file.read()
                        documents.append(
                            Document(
                                page_content=content,
                                metadata={"source": filename}
                            )
                        )
                except Exception as e:
                    print(f"Error reading file {filename}: {str(e)}")
                    
        print(f"Loaded {len(documents)} documents")
        return documents
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        if not documents:
            print("No documents to split")
            return []
            
        splitter = MarkdownTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(documents)
        print(f"Split into {len(chunks)} chunks")
        return chunks
    
    def create_vector_db(self, chunks: List[Document]):
        if not chunks:
            print("No chunks to index")
            # Return a default response or raise an exception
            return None
            
        if not os.path.exists(self.db_dir):
            os.makedirs(self.db_dir)
            
        return Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.db_dir
        )
    
    def process_and_index(self):
        documents = self.load_documents()
        if not documents:
            raise Exception("No documents found to process")
            
        chunks = self.split_documents(documents)
        if not chunks:
            raise Exception("No chunks created from documents")
            
        vectordb = self.create_vector_db(chunks)
        if not vectordb:
            raise Exception("Failed to create vector database")
            
        return vectordb

if __name__ == "__main__":
    data_dir = "../data/references"
    db_dir = "../data/vectordb"
    
    processor = DocumentProcessor(data_dir, db_dir)
    processor.process_and_index() 