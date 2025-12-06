import os
from typing import List, Dict

def load_documents(docs_dir: str) -> List[Dict[str, str]]:
    documents = []
    
    if not os.path.exists(docs_dir):
        print(f"Warning: Directory {docs_dir} does not exist.")
        return documents

    for root, _, files in os.walk(docs_dir):
        for filename in files:
            if filename.endswith(".txt"):
                filepath = os.path.join(root, filename)
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        content = f.read()
                        documents.append({"filename": filename, "content": content})
                except Exception as e:
                    print(f"Error reading file {filename}: {e}")
                
    return documents
