import os
import requests
from bs4 import BeautifulSoup
import markdown
import json
from typing import List, Dict, Any
import html2text

def fetch_url_content(url: str) -> str:
    """Fetch content from a URL."""
    response = requests.get(url)
    response.raise_for_status()
    return response.text

def html_to_markdown(html_content: str, url: str) -> str:
    """Convert HTML content to markdown."""
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Remove navigation, sidebars, etc.
    for element in soup.select('nav, .sidebar, .navigation, .footer, script, style'):
        element.decompose()
    
    # Extract the main content
    main_content = soup.select_one('main, .main-content, article, .content, .documentation, #content')
    if main_content:
        html_content = str(main_content)
    
    # Convert to markdown
    converter = html2text.HTML2Text()
    converter.ignore_links = False
    converter.ignore_images = False
    converter.ignore_tables = False
    markdown_content = converter.handle(html_content)
    
    # Add source reference
    markdown_content = f"# Content from {url}\n\n{markdown_content}"
    
    return markdown_content

def process_reference_url(url: str, output_dir: str, metadata: Dict[str, Any] = None) -> str:
    """Process a reference URL and save as markdown."""
    try:
        html_content = fetch_url_content(url)
        markdown_content = html_to_markdown(html_content, url)
        
        # Create a filename from the URL
        filename = url.split('/')[-1].split('.')[0]
        if not filename:
            filename = "index"
        
        filepath = os.path.join(output_dir, f"{filename}.md")
        
        # Add metadata as YAML frontmatter
        if metadata:
            yaml_metadata = "---\n"
            for key, value in metadata.items():
                yaml_metadata += f"{key}: {value}\n"
            yaml_metadata += "---\n\n"
            markdown_content = yaml_metadata + markdown_content
        
        # Save to file
        with open(filepath, 'w') as f:
            f.write(markdown_content)
            
        return filepath
    except Exception as e:
        print(f"Error processing {url}: {e}")
        return None

def process_reference_urls(urls: List[Dict[str, Any]], output_dir: str) -> List[str]:
    """Process multiple reference URLs."""
    os.makedirs(output_dir, exist_ok=True)
    
    processed_files = []
    for url_info in urls:
        url = url_info['url']
        metadata = url_info.get('metadata', {})
        filepath = process_reference_url(url, output_dir, metadata)
        if filepath:
            processed_files.append(filepath)
    
    return processed_files

if __name__ == "__main__":
    # Example usage
    reference_urls = [
        {
            "url": "https://scikit-learn.org/stable/modules/tree.html",
            "metadata": {
                "model_family": "decision_trees",
                "problem_types": ["classification", "regression"],
                "tags": ["interpretable", "non-parametric"]
            }
        }
    ]
    
    output_dir = "../data/references"
    processed_files = process_reference_urls(reference_urls, output_dir)
    print(f"Processed {len(processed_files)} reference files") 