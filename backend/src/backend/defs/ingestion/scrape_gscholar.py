"""
Google Scholar venue scraping asset.

Scrapes Google Scholar's top venues by field and combines them into a unified dataset.
"""

import pandas as pd
import dagster as dg
from dagster_duckdb import DuckDBResource




@dg.asset(
        kinds={"duckdb"}, 
        group_name="ingestion"
)
def gscholar_venues(duckdb: DuckDBResource) -> dg.MaterializeResult:
    """Scrape Google Scholar top venues by academic field"""
    
    # Define field categories and their URLs
    fields = {
        'Business Economics Management': 'https://scholar.google.com/citations?view_op=top_venues&hl=en&vq=bus',
        'Chemistry Material Science': 'https://scholar.google.com/citations?view_op=top_venues&hl=en&vq=chm',
        'Engineering Computer Science': 'https://scholar.google.com/citations?view_op=top_venues&hl=en&vq=eng',
        'Health Medicine': 'https://scholar.google.com/citations?view_op=top_venues&hl=en&vq=med',
        'Humanities Literature Arts': 'https://scholar.google.com/citations?view_op=top_venues&hl=en&vq=hum',
        'Life Earth Sciences': 'https://scholar.google.com/citations?view_op=top_venues&hl=en&vq=bio',
        'Physics Mathematics': 'https://scholar.google.com/citations?view_op=top_venues&hl=en&vq=phy',
        'Social Sciences': 'https://scholar.google.com/citations?view_op=top_venues&hl=en&vq=soc'
    }
    
    # Scrape data from each field
    field_dataframes = []
    field_stats = {}
    
    for field_name, url in fields.items():
        try:
            df_list = pd.read_html(url)
            df = df_list[0]  # Take the first table
            df['field'] = field_name  # Add field identifier
            field_dataframes.append(df)
            field_stats[field_name] = len(df)
        except Exception as e:
            print(f"Error scraping {field_name}: {e}")
            field_stats[field_name] = 0
    
    # Combine all fields
    if field_dataframes:
        combined_data = pd.concat(field_dataframes, axis=0, ignore_index=True)
        
        # Create simple schema
        df_simple = pd.DataFrame({
            'source': 'gscholar_venues',
            'venue': combined_data.iloc[:, 0],  # First column is typically the venue name
            'field': combined_data['field'],
            'h5_index': combined_data.iloc[:, 1] if combined_data.shape[1] > 1 else None,  # Second column is h5-index
            'h5_median': combined_data.iloc[:, 2] if combined_data.shape[1] > 2 else None   # Third column is h5-median
        })
        
        with duckdb.get_connection() as conn:
            conn.execute("create or replace table main.gscholar_venues as select * from df_simple")
        
        # Calculate metadata
        total_venues = len(df_simple)
        unique_fields = len(field_stats)
        successful_fields = len([f for f, count in field_stats.items() if count > 0])
        
        return dg.MaterializeResult(
            metadata={
                "total_venues": total_venues,
                "unique_fields": unique_fields,
                "successful_fields": successful_fields,
                "field_breakdown": field_stats,
                "columns": list(df_simple.columns)
            }
        )
    else:
        # Create empty table if no data was scraped
        df_simple = pd.DataFrame({
            'source': [],
            'venue': [],
            'field': [],
            'h5_index': [],
            'h5_median': []
        })
        
        with duckdb.get_connection() as conn:
            conn.execute("create or replace table main.gscholar_venues as select * from df_simple")
        
        return dg.MaterializeResult(
            metadata={
                "total_venues": 0,
                "unique_fields": 0,
                "successful_fields": 0,
                "field_breakdown": field_stats,
                "error": "No data was successfully scraped"
            }
        )