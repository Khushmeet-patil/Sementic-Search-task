import os

# Project Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS_DIR = os.path.join(BASE_DIR, "docs")


CHUNK_SIZE = 100  
OVERLAP = 20      

#Model
EMBEDDING_MODEL_NAME = "all-mpnet-base-v2"
