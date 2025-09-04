
"""
S2ORC dark data collection asset.

Processes S2ORC scientific papers to extract text paragraphs containing specific keywords
and loads them into MongoDB for annotation tasks.
"""

import dagster as dg
from dagster_duckdb import DuckDBResource
from backend.defs.resources import SemanticScholarResource
import pandas as pd


@dg.asset(
    kinds={"mongodb"}, 
    deps=["gscholar_venues"],
    group_name="ingestion"
)
def retrieval(
    duckdb: DuckDBResource, s2_resource: SemanticScholarResource
) -> dg.MaterializeResult:
    """Filter works_oa collection to get venues of interest and papers with s2orc_parsed=True"""


    df = pd.read_csv("src/backend/defs/data/metadata_venue.csv")

    # s2_resource = SemanticScholarResource()
    s2_client = s2_resource.get_client()

    snippet = s2_client.get_snippet(
        query = "data available on reasonable request",
        year = "2000-2025",
        venue = df.display_name[166],
        minCitationCount = 3,
        )

    len(snippet)