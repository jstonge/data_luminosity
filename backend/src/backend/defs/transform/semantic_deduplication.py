"""
Semantic deduplication asset for training data purification.

Uses embeddings and cosine similarity to remove semantically similar texts,
improving training data quality by eliminating near-duplicates and paraphrases.
"""

import pandas as pd
import dagster as dg
import duckdb
import filelock
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
import json
from datetime import datetime


def serialize_duckdb_query(duckdb_path: str, sql: str):
    """Execute SQL statement with file lock to guarantee cross-process concurrency."""
    lock_path = f"{duckdb_path}.lock"
    with filelock.FileLock(lock_path):
        conn = duckdb.connect(duckdb_path)
        try:
            result = conn.execute(sql)
            if sql.strip().upper().startswith('SELECT'):
                return result.fetchall()
            return result
        finally:
            conn.close()


def create_table_from_df(duckdb_path: str, table_name: str, df: pd.DataFrame):
    """Create DuckDB table from pandas DataFrame with file lock."""
    lock_path = f"{duckdb_path}.lock"
    with filelock.FileLock(lock_path):
        conn = duckdb.connect(duckdb_path)
        try:
            conn.execute(f"create or replace table {table_name} as select * from df")
        finally:
            conn.close()


def compute_similarity_batches(texts, embedding_model, batch_size=32):
    """Compute embeddings for texts in batches to handle memory efficiently."""
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        batch_embeddings = embedding_model.encode(batch_texts, convert_to_tensor=False)
        embeddings.extend(batch_embeddings)
    return np.array(embeddings)


def find_similar_indices(similarity_matrix, threshold):
    """Find indices of texts to remove based on similarity threshold."""
    indices_to_remove = set()
    n = len(similarity_matrix)
    
    for i in range(n):
        if i in indices_to_remove:
            continue
        for j in range(i+1, n):
            if j in indices_to_remove:
                continue
            if similarity_matrix[i][j] >= threshold:
                indices_to_remove.add(j)  # Remove the later occurrence
    
    return indices_to_remove


@dg.asset(
    kinds={"python"}, 
    deps=["combined_annotations"],
    group_name="transform"
)
def deduplicated_annotations(context: dg.AssetExecutionContext) -> dg.MaterializeResult:
    """
    Remove exact and semantic duplicates from combined annotations.
    
    Uses embeddings and cosine similarity to identify and remove semantically
    similar texts, creating a cleaner dataset for model training.
    """
    
    # Configuration
    enable_semantic_dedup = True
    similarity_threshold = 0.85  # Higher = more strict (fewer duplicates removed)
    embedding_model_name = "all-MiniLM-L6-v2"  # Fast, good quality embeddings
    batch_size = 32
    
    print(f"Semantic deduplication: {'enabled' if enable_semantic_dedup else 'disabled'}")
    print(f"Similarity threshold: {similarity_threshold}")
    print(f"Embedding model: {embedding_model_name}")
    
    # Load data from combined_annotations
    query = """
        SELECT 
            text,
            has_data_statement,
            statement_category,
            source
        FROM main.combined_annotations 
        WHERE text IS NOT NULL 
        AND has_data_statement IS NOT NULL
    """
    
    annotations_data = serialize_duckdb_query("/tmp/data_luminosity.duckdb", query)
    
    if not annotations_data:
        context.log.warning("No annotation data found")
        return dg.MaterializeResult(
            metadata={
                "error": "No annotation data found",
                "total_samples": 0
            }
        )
    
    # Convert to DataFrame
    df = pd.DataFrame(annotations_data, columns=['text', 'has_data_statement', 'statement_category', 'source'])
    original_count = len(df)
    
    print(f"Loaded {original_count} annotations for deduplication")
    
    # Clean data first
    print("Cleaning annotation data...")
    
    # Map to binary sentiment (yes/no)
    df['sentiment'] = df['has_data_statement'].map({'yes': 'yes', 'no': 'no'})
    
    # Remove maybe/uncertain cases for cleaner training data
    df = df[df['sentiment'].notna()]
    df = df[~df['sentiment'].str.contains('maybe', na=False)]
    df = df[~df['sentiment'].str.contains('uncertain', na=False)]
    
    cleaned_count = len(df)
    print(f"Cleaned data: {original_count} -> {cleaned_count} texts")
    
    # Step 1: Remove exact duplicates
    df = df.drop_duplicates(subset='text')
    exact_dedup_count = len(df)
    exact_removed = cleaned_count - exact_dedup_count
    
    print(f"Removed {exact_removed} exact duplicate texts")
    
    # Step 2: Semantic deduplication using embeddings
    semantic_removed = 0
    semantic_dedup_success = False
    
    if enable_semantic_dedup and exact_dedup_count > 1:
        print(f"Starting semantic deduplication with threshold {similarity_threshold}...")
        
        try:
            # Load embedding model
            embedding_model = SentenceTransformer(embedding_model_name)
            print(f"Loaded embedding model: {embedding_model_name}")
            
            # Get texts for embedding
            texts = df['text'].tolist()
            
            print(f"Computing embeddings for {len(texts)} texts...")
            # Compute embeddings in batches
            embeddings = compute_similarity_batches(texts, embedding_model, batch_size)
            
            print("Computing similarity matrix...")
            # Compute similarity matrix
            similarity_matrix = cosine_similarity(embeddings)
            
            print("Finding similar texts to remove...")
            # Find indices to remove
            indices_to_remove = find_similar_indices(similarity_matrix, similarity_threshold)
            
            # Remove similar texts
            if indices_to_remove:
                df = df.iloc[~df.index.isin(indices_to_remove)].reset_index(drop=True)
                semantic_removed = len(indices_to_remove)
                print(f"Removed {semantic_removed} semantically similar texts")
            else:
                print("No semantically similar texts found to remove")
            
            semantic_dedup_success = True
            
        except Exception as e:
            context.log.warning(f"Semantic deduplication failed: {e}. Continuing with exact deduplication only.")
            semantic_removed = 0
            semantic_dedup_success = False
    elif not enable_semantic_dedup:
        print("Semantic deduplication disabled")
    else:
        print("Too few texts for semantic deduplication")
    
    final_count = len(df)
    total_removed = original_count - final_count
    
    print(f"Deduplication complete: {original_count} -> {final_count} texts ({total_removed} removed total)")
    
    # Save deduplicated data to DuckDB
    create_table_from_df("/tmp/data_luminosity.duckdb", "main.deduplicated_annotations", df)
    
    # Calculate statistics
    source_distribution = df['source'].value_counts().to_dict()
    sentiment_distribution = df['sentiment'].value_counts().to_dict()
    
    return dg.MaterializeResult(
        metadata={
            "deduplication_date": datetime.now().isoformat(),
            "original_count": original_count,
            "cleaned_count": cleaned_count,
            "final_count": final_count,
            "exact_duplicates_removed": exact_removed,
            "semantic_duplicates_removed": semantic_removed,
            "total_removed": total_removed,
            "removal_percentage": round((total_removed / original_count) * 100, 2) if original_count > 0 else 0,
            "semantic_deduplication_enabled": enable_semantic_dedup,
            "semantic_deduplication_success": semantic_dedup_success,
            "similarity_threshold": similarity_threshold,
            "embedding_model": embedding_model_name,
            "source_distribution": source_distribution,
            "sentiment_distribution": sentiment_distribution,
            "columns": list(df.columns)
        }
    )