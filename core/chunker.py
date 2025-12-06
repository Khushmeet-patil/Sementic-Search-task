from typing import List

def chunk_text(text: str, chunk_size: int = 100, overlap: int = 20) -> List[str]:
    if not text:
        return []
        
    words = text.split()
    
    if len(words) <= chunk_size:
        return [text]
        
    chunks = []
    start = 0
    
    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        
        # Join words back into a string
        chunk_text = " ".join(chunk_words)
        chunks.append(chunk_text)
        
        # Move the window
        start += (chunk_size - overlap)
        
        # Break if we've reached the end to avoid infinite loops or tiny tail chunks if logic changes
        if start >= len(words):
            break
            
    return chunks
