"""
Combined annotations asset that merges Label Studio annotations with literature review data.

This asset queries Label Studio for current annotations and combines them with the
external literature review datasets to create a unified training dataset.
"""

import pandas as pd
import dagster as dg
from dagster_duckdb import DuckDBResource
from backend.defs.resources import LabelStudioResource


def _safe_get_choice(ann, default='unknown'):
    """Safely extract choice from Label Studio annotation structure"""
    try:
        annotations = ann.get('annotations', [])
        if not annotations:
            return default
            
        result = annotations[0].get('result', [])
        if not result:
            return default
            
        value = result[0].get('value', {})
        choices = value.get('choices', [])
        if not choices:
            return default
            
        return choices[0]
    except (IndexError, KeyError, TypeError):
        return default




@dg.asset(
        kinds={"duckdb"}, 
        deps=["dispatch_annotations"], 
        group_name="transform"
)
def label_studio_annotations(duckdb: DuckDBResource, ls_client: LabelStudioResource) -> dg.MaterializeResult:
    """Get current Label Studio annotations and convert to simple schema"""
    # Get current Label Studio annotations
    client = ls_client.get_client()
    client.LS_TOK = '8ecf272719351405d5c1dee84c97a9c304a9e96e'
    
    # Get annotations from the Dark Data project (project ID 42 based on the code)
    ls_annotations = client.get_annotations_LS(proj_id=42, only_annots=True)
    
    # Convert LS annotations to simple schema format
    ls_df_simple = pd.DataFrame({
        'source': 'label_studio',
        'text': [ann.get('data', {}).get('text', '') for ann in ls_annotations],
        'venue': None,  # Label Studio data doesn't have venue info
        'publication_year': None,  # Label Studio data doesn't have publication year
        'has_data_statement': [
            _safe_get_choice(ann, 'unknown') for ann in ls_annotations
        ],
        'statement_category': [
            # Map LS choices to our categories
            {'yes': 'has_statement', 'no': 'no_statement', 'maybe': 'uncertain'}.get(
                _safe_get_choice(ann, 'other'), 'other'
            )
            for ann in ls_annotations
        ]
    })
    
    with duckdb.get_connection() as conn:
        conn.execute("create or replace table main.label_studio as select * from ls_df_simple")
    
    # Calculate metadata
    total_rows = len(ls_df_simple)
    has_statement_count = len(ls_df_simple[ls_df_simple['has_data_statement'] == 'yes'])
    category_counts = ls_df_simple['statement_category'].value_counts().to_dict()
    
    return dg.MaterializeResult(
        metadata={
            "total_rows": total_rows,
            "total_annotations": len(ls_annotations),
            "has_data_statement_count": has_statement_count,
            "has_data_statement_percentage": round((has_statement_count / total_rows) * 100, 1) if total_rows > 0 else 0,
            "statement_categories": category_counts,
            "columns": list(ls_df_simple.columns)
        }
    )


@dg.asset(
    kinds={"duckdb"}, 
    deps=[
        "federer_data_2018",
        "grant_impact_2018", 
        "jones_observed_2025",
        "mcguinness_descriptive_2021",
        "karcher_replication_2025",
        "label_studio_annotations"
    ],
    group_name="transform"   
)
def combined_annotations(duckdb: DuckDBResource) -> dg.MaterializeResult:
    """Combine Label Studio annotations with literature review datasets"""
    query = """
        create or replace table main.combined_annotations as (
            -- Literature review datasets
            select * from main.federer_data_2018
            union all
            select * from main.grant_impact_2018
            union all 
            select * from main.jones_observed_2025
            union all
            select * from main.mcguinness_descriptive_2021
            union all
            select * from main.karcher_replication_2025
            union all
            -- Label Studio annotations
            select * from main.label_studio
        )
    """
    with duckdb.get_connection() as conn:
        conn.execute(query)
    
    # Calculate metadata from combined table
    metadata_query = """
        select 
            source,
            count(*) as row_count,
            sum(case when has_data_statement = 'yes' then 1 else 0 end) as has_statement_count,
            count(distinct statement_category) as unique_categories
        from main.combined_annotations 
        group by source
        order by source
    """
    
    # Get overall stats
    total_query = """
        select 
            count(*) as total_rows,
            sum(case when has_data_statement = 'yes' then 1 else 0 end) as total_has_statement,
            count(distinct source) as source_count
        from main.combined_annotations
    """
    
    with duckdb.get_connection() as conn:
        source_stats = conn.execute(metadata_query).fetchall()
        total_stats = conn.execute(total_query).fetchone()
    
    # Format metadata
    source_breakdown = {}
    for source, row_count, has_statement_count, unique_categories in source_stats:
        source_breakdown[source] = {
            "row_count": row_count,
            "has_statement_count": has_statement_count,
            "has_statement_percentage": round((has_statement_count / row_count) * 100, 1) if row_count > 0 else 0,
            "unique_categories": unique_categories
        }
    
    return dg.MaterializeResult(
        metadata={
            "total_rows": total_stats[0],
            "total_has_statement_count": total_stats[1], 
            "total_has_statement_percentage": round((total_stats[1] / total_stats[0]) * 100, 1) if total_stats[0] > 0 else 0,
            "source_count": total_stats[2],
            "source_breakdown": source_breakdown
        }
    )