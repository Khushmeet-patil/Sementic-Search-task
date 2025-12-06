import os

# Project Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS_DIR = os.path.join(BASE_DIR, "docs")
CHROMA_DB_DIR = os.path.join(BASE_DIR, "chroma_db")
COLLECTION_NAME = "documents_collection"



CHUNK_SIZE = 100  
OVERLAP = 20      

#Model
EMBEDDING_MODEL_NAME = "all-mpnet-base-v2"


#Best accuracy model
#bge-large-en-v1.5