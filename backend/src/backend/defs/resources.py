from dagster_duckdb import DuckDBResource
import dagster as dg
from typing import Optional

from backend.clients.label_studio import LabelStudioClient

database_resource = DuckDBResource(database="/tmp/data_luminosity.duckdb")

class LabelStudioResource(dg.ConfigurableResource):
    api_key: Optional[str] = None
    
    def get_client(self) -> LabelStudioClient:
        return LabelStudioClient()
    
# Update your resources definition
@dg.definitions  
def resources():
    return dg.Definitions(
        resources={
            "duckdb": database_resource,
            "ls_client": LabelStudioClient()
        }
    )

