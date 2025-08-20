import pandas as pd
import dagster as dg
import duckdb
import filelock


def serialize_duckdb_query(duckdb_path: str, sql: str):
    """Execute SQL statement with file lock to guarantee cross-process concurrency."""
    lock_path = f"{duckdb_path}.lock"
    with filelock.FileLock(lock_path):
        conn = duckdb.connect(duckdb_path)
        try:
            return conn.execute(sql)
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


@dg.asset(kinds={"duckdb"}, key=["target", "main", "federer2018"], group_name="literature_review")
def federer2018() -> dg.MaterializeResult:
    df = pd.read_csv(
        'src/backend/defs/data/lit_review/federer_data_2018/final_full_coded_set.csv', encoding='ISO-8859-1'
        )
    
    # Simple schema
    df_simple = pd.DataFrame({
        'source': 'federer2018',
        'text': df['data_statement'],
        'venue': None,
        'publication_year': pd.to_datetime(df['publication_date.x']).dt.year,
        'has_data_statement': df['statement_category'].isin(['in paper and si', 'in paper', 'repository', 'access restricted', 'combination', 'upon request', 'in si', 'location not stated', 'other']).map({True: 'yes', False: 'no'}),
        'statement_category': df['statement_category'].map({
            'in paper and si': 'available_paper_si',
            'in paper': 'available_paper',
            'repository': 'available_repository',
            'access restricted': 'restricted',
            'combination': 'available_mixed',
            'upon request': 'upon_request',
            'in si': 'available_si',
            'location not stated': 'available_unspecified',
            'other': 'other'
        }).fillna('other')
    })
    
    create_table_from_df("/tmp/data_luminosity.duckdb", "main.federer2018", df_simple)
    
    # Calculate metadata
    total_rows = len(df_simple)
    has_statement_count = len(df_simple[df_simple['has_data_statement'] == 'yes'])
    category_counts = df_simple['statement_category'].value_counts().to_dict()
    
    return dg.MaterializeResult(
        metadata={
            "total_rows": total_rows,
            "has_data_statement_count": has_statement_count,
            "has_data_statement_percentage": round((has_statement_count / total_rows) * 100, 1) if total_rows > 0 else 0,
            "statement_categories": category_counts,
            "columns": list(df_simple.columns)
        }
    )


@dg.asset(kinds={"duckdb"}, key=["target", "main", "grant2018"], group_name="literature_review")
def grant2018() -> dg.MaterializeResult:
    df = pd.read_excel('src/backend/defs/data/lit_review/grant_impact_2018/NatureDataAvailabilityStatementsdataset2018.xlsx')
    
    # Simple schema
    df_simple = pd.DataFrame({
        'source': 'grant2018',
        'text': df['Data availability statement text'],
        'venue': df['Journal title'],
        'publication_year': None,
        'has_data_statement': 'yes',  # All papers in this dataset have statements
        'statement_category': df['Statement type (1, 2, 3, 4)'].map({
            1.0: 'upon_request',          # Data available from author on request
            2.0: 'available_paper_si',    # Data in manuscript or supplementary material
            3.0: 'available_repository',  # Data publicly available (e.g., repository)
            4.0: 'available_paper_si'     # Figure source data included with manuscript
        }).fillna('other')
    })
    
    create_table_from_df("/tmp/data_luminosity.duckdb", "main.grant2018", df_simple)
    
    # Calculate metadata
    total_rows = len(df_simple)
    has_statement_count = len(df_simple[df_simple['has_data_statement'] == 'yes'])
    category_counts = df_simple['statement_category'].value_counts().to_dict()
    
    return dg.MaterializeResult(
        metadata={
            "total_rows": total_rows,
            "has_data_statement_count": has_statement_count,
            "has_data_statement_percentage": round((has_statement_count / total_rows) * 100, 1) if total_rows > 0 else 0,
            "statement_categories": category_counts,
            "columns": list(df_simple.columns)
        }
    )


@dg.asset(kinds={"duckdb"}, key=["target", "main", "jones_observed_2025"], group_name="literature_review")
def jones_observed_2025() -> dg.MaterializeResult:
    csv_files = [
        'src/backend/defs/data/lit_review/jones_observed_2025/ASDC_AIES.csv',
        'src/backend/defs/data/lit_review/jones_observed_2025/ASDC_AIJ.csv',
        'src/backend/defs/data/lit_review/jones_observed_2025/ASDC_AI_in_Geo.csv',
        'src/backend/defs/data/lit_review/jones_observed_2025/ASDC_MWR.csv'
    ]
    
    dfs = []
    for file_path in csv_files:
        df = pd.read_csv(file_path)
        dfs.append(df)
    
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # Simple schema
    df_simple = pd.DataFrame({
        'source': 'jones_observed_2025',
        'text': None,  # No statement text in this dataset
        'venue': combined_df['Journal name'],
        'publication_year': combined_df['Year of publication'].astype(str),
        'has_data_statement': combined_df['Is there an ASDC?'].map({
            'yes': 'yes',
            'no': 'no'
        }).fillna('unknown'),
        'statement_category': combined_df['Is there an ASDC?'].map({
            'yes': 'has_statement',
            'no': 'no_statement'
        }).fillna('other')
    })
    
    create_table_from_df("/tmp/data_luminosity.duckdb", "main.jones_observed_2025", df_simple)
    
    # Calculate metadata
    total_rows = len(df_simple)
    has_statement_count = len(df_simple[df_simple['has_data_statement'] == 'yes'])
    category_counts = df_simple['statement_category'].value_counts().to_dict()
    
    return dg.MaterializeResult(
        metadata={
            "total_rows": total_rows,
            "has_data_statement_count": has_statement_count,
            "has_data_statement_percentage": round((has_statement_count / total_rows) * 100, 1) if total_rows > 0 else 0,
            "statement_categories": category_counts,
            "columns": list(df_simple.columns)
        }
    )


@dg.asset(kinds={"duckdb"}, key=["target", "main", "mcguinness_descriptive_2021"], group_name="literature_review")
def mcguinness_descriptive_2021() -> dg.MaterializeResult:
    df = pd.read_csv('src/backend/defs/data/lit_review/mcguinness_descriptive_2021/data-avail-published.csv', encoding='ISO-8859-1')
    
    # Simple schema
    df_simple = pd.DataFrame({
        'source': 'mcguinness_descriptive_2021',
        'text': df['data_avail'],
        'venue': df['journal'], 
        'publication_year': pd.to_datetime(df['date'], format='%Y%m%d').dt.year,
        'has_data_statement': df['data_avail'].notna().map({True: 'yes', False: 'no'}),
        'statement_category': df['data_avail'].apply(lambda x: 
            'upon_request' if pd.notna(x) and ('upon request' in str(x).lower() or 'on reasonable request' in str(x).lower()) else
            'available_paper_si' if pd.notna(x) and ('supplementary' in str(x).lower() or 'within the article' in str(x).lower() or 'included within' in str(x).lower()) else
            'available_repository' if pd.notna(x) and ('repository' in str(x).lower() or 'publicly available' in str(x).lower() or 'public domain' in str(x).lower()) else
            'not_applicable' if pd.notna(x) and 'not applicable' in str(x).lower() else
            'not_available' if pd.notna(x) and ('not available' in str(x).lower() or 'no empirical data' in str(x).lower()) else
            'other' if pd.notna(x) else 'no_statement'
        )
    })
    
    create_table_from_df("/tmp/data_luminosity.duckdb", "main.mcguinness_descriptive_2021", df_simple)
    
    # Calculate metadata
    total_rows = len(df_simple)
    has_statement_count = len(df_simple[df_simple['has_data_statement'] == 'yes'])
    category_counts = df_simple['statement_category'].value_counts().to_dict()
    
    return dg.MaterializeResult(
        metadata={
            "total_rows": total_rows,
            "has_data_statement_count": has_statement_count,
            "has_data_statement_percentage": round((has_statement_count / total_rows) * 100, 1) if total_rows > 0 else 0,
            "statement_categories": category_counts,
            "columns": list(df_simple.columns)
        }
    )


@dg.asset(kinds={"duckdb"}, key=["target", "main", "karcher_replication_2025"], group_name="literature_review")
def karcher_replication_2025() -> dg.MaterializeResult:
    df = pd.read_csv('src/backend/defs/data/lit_review/karcher_replication_2025/dataverse_files/data/raw/plos_fully_coded.csv')
    
    # Simple schema
    df_simple = pd.DataFrame({
        'source': 'karcher_replication_2025',
        'text': df['data_availability'],
        'venue': df['journal'],
        'publication_year': pd.to_datetime(df['pub_date']).dt.year,
        'has_data_statement': df['data_availability'].notna().map({True: 'yes', False: 'no'}),
        'statement_category': df['data_availability'].apply(lambda x:
            'available_paper_si' if pd.notna(x) and ('within the paper' in str(x).lower() or 'within the manuscript' in str(x).lower() or 'supporting information' in str(x).lower()) else
            'available_repository' if pd.notna(x) and ('publicly available' in str(x).lower() or 'repository' in str(x).lower() or 'webpages' in str(x).lower()) else
            'upon_request' if pd.notna(x) and ('upon request' in str(x).lower() or 'on request' in str(x).lower()) else
            'not_available' if pd.notna(x) and ('cannot be shared' in str(x).lower() or 'not available' in str(x).lower()) else
            'other' if pd.notna(x) else 'no_statement'
        )
    })
    
    create_table_from_df("/tmp/data_luminosity.duckdb", "main.karcher_replication_2025", df_simple)
    
    # Calculate metadata
    total_rows = len(df_simple)
    has_statement_count = len(df_simple[df_simple['has_data_statement'] == 'yes'])
    category_counts = df_simple['statement_category'].value_counts().to_dict()
    
    return dg.MaterializeResult(
        metadata={
            "total_rows": total_rows,
            "has_data_statement_count": has_statement_count,
            "has_data_statement_percentage": round((has_statement_count / total_rows) * 100, 1) if total_rows > 0 else 0,
            "statement_categories": category_counts,
            "columns": list(df_simple.columns)
        }
    )

