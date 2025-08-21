from dagster_duckdb import DuckDBResource
import dagster as dg
from typing import Optional
from pymongo import MongoClient

from backend.clients.label_studio import LabelStudioClient

# We use duckdb as our working space
database_resource = DuckDBResource(database=dg.EnvVar("DUCKDB_PATH"))

class MongoDBResource(dg.ConfigurableResource):
    """MongoDB connection resource for papersDB"""
    host: str = dg.EnvVar("MONGODB_HOST")
    port: int = dg.EnvVar.int("MONGODB_PORT")
    username: str = dg.EnvVar("MONGODB_USERNAME")
    password: str = dg.EnvVar("MONGODB_PASSWORD")
    database_name: str = dg.EnvVar("MONGODB_DATABASE")
    auth_source: str = dg.EnvVar("MONGODB_AUTH_SOURCE")
    
    def get_client(self) -> MongoClient:
        """Get MongoDB client instance"""
        uri = f"mongodb://{self.username}:{self.password}@{self.host}:{self.port}/?authSource={self.auth_source}&readPreference=primary&appname=MongoDB%20Compass&directConnection=true&ssl=false"
        return MongoClient(uri)
    
    def get_database(self):
        """Get database instance"""
        client = self.get_client()
        return client[self.database_name]

class LabelStudioResource(dg.ConfigurableResource):
    """Label Studio client resource with MongoDB dependency"""
    api_token: str = dg.EnvVar("LS_TOK")
    mongodb: dg.ResourceDependency[MongoDBResource]
    
    def get_client(self) -> LabelStudioClient:
        return LabelStudioClient(
            api_token=self.api_token,
            mongodb_resource=self.mongodb
        )
    
@dg.definitions  
def resources():
    return dg.Definitions(
        resources={
            "duckdb": database_resource,
            "mongodb": MongoDBResource(),
            "ls_client": LabelStudioResource()
        }
    )

