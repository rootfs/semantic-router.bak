#!/usr/bin/env python3
"""
Generate Embeddings for Experience Database

This script generates embeddings for the experience database entries
so they can be loaded directly by the Go cluster router without
needing the embedding model at runtime.

Usage:
    python generate_embeddings.py [--model MODEL] [--output OUTPUT]

Requirements:
    pip install sentence-transformers torch
"""

import json
import argparse
import os
from typing import List, Dict, Any

def load_experience_db(path: str) -> List[Dict[str, Any]]:
    """Load experience database from JSON file."""
    with open(path, "r") as f:
        return json.load(f)

def save_experience_db(data: List[Dict[str, Any]], path: str):
    """Save experience database to JSON file."""
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

def generate_embeddings_batch(texts: List[str], model_name: str = "BAAI/bge-base-en-v1.5") -> List[List[float]]:
    """Generate embeddings for a batch of texts using sentence-transformers."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("Error: sentence-transformers not installed")
        print("Install with: pip install sentence-transformers")
        raise
    
    print(f"Loading model: {model_name}")
    model = SentenceTransformer(model_name)
    
    print(f"Generating embeddings for {len(texts)} texts...")
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    
    return [emb.tolist() for emb in embeddings]

def main():
    parser = argparse.ArgumentParser(description='Generate embeddings for experience database')
    parser.add_argument('--input', type=str, default='experience_db.json',
                       help='Input experience database JSON file')
    parser.add_argument('--output', type=str, default='experience_db_with_embeddings.json',
                       help='Output file with embeddings')
    parser.add_argument('--model', type=str, default='BAAI/bge-base-en-v1.5',
                       help='Sentence transformer model to use')
    parser.add_argument('--batch-size', type=int, default=100,
                       help='Batch size for embedding generation')
    args = parser.parse_args()
    
    # Check input file exists
    if not os.path.exists(args.input):
        print(f"Error: {args.input} not found")
        return
    
    # Load experience database
    print(f"Loading experience database from {args.input}...")
    exp_db = load_experience_db(args.input)
    print(f"Loaded {len(exp_db)} entries")
    
    # Extract texts that need embeddings
    texts_to_embed = []
    indices_to_embed = []
    
    for i, entry in enumerate(exp_db):
        # Skip entries that already have embeddings
        if "embedding" in entry and entry["embedding"]:
            continue
        
        # Get query text
        query_text = entry.get("query_text", "")
        if query_text:
            texts_to_embed.append(query_text)
            indices_to_embed.append(i)
    
    if not texts_to_embed:
        print("All entries already have embeddings, nothing to do")
        return
    
    print(f"Generating embeddings for {len(texts_to_embed)} entries...")
    
    # Generate embeddings in batches
    embeddings = generate_embeddings_batch(texts_to_embed, args.model)
    
    # Update experience database with embeddings
    for idx, embedding in zip(indices_to_embed, embeddings):
        exp_db[idx]["embedding"] = embedding
    
    # Remove query_text to save space (embedding is sufficient for routing)
    for entry in exp_db:
        if "query_text" in entry and "embedding" in entry:
            # Keep query_text in metadata for debugging
            if "metadata" not in entry:
                entry["metadata"] = {}
            entry["metadata"]["query_text_preview"] = entry["query_text"][:100] + "..." if len(entry["query_text"]) > 100 else entry["query_text"]
            del entry["query_text"]
    
    # Save updated experience database
    print(f"Saving to {args.output}...")
    save_experience_db(exp_db, args.output)
    
    # Print statistics
    embedding_dim = len(embeddings[0]) if embeddings else 0
    print(f"\nDone!")
    print(f"  Total entries: {len(exp_db)}")
    print(f"  Embedding dimension: {embedding_dim}")
    print(f"  Output file: {args.output}")
    print(f"  File size: {os.path.getsize(args.output) / 1024 / 1024:.2f} MB")

if __name__ == "__main__":
    main()

