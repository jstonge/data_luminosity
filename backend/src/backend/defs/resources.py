from dagster_duckdb import DuckDBResource
import dagster as dg
from typing import Optional
from pymongo import MongoClient
import os

from backend.clients.label_studio import LabelStudioClient
from backend.clients.semantic_scholar import SemanticScholarClient

# We use duckdb as our working space
database_resource = DuckDBResource(database='data/data_luminosity.duckdb')

class MongoDBResource(dg.ConfigurableResource):
    """MongoDB connection resource for papersDB"""
    password: str = os.environ.get("MONGODB_PASSWORD")
    
    def get_client(self) -> MongoClient:
        """Get MongoDB client instance"""
        uri = f"mongodb://cwward:{self.password}@wranglerdb01a.uvm.edu:27017/?authSource=admin&readPreference=primary&appname=MongoDB%20Compass&directConnection=true&ssl=false"
        return MongoClient(uri)
    
    def get_database(self):
        """Get database instance"""
        client = self.get_client()
        return client['papersDB']

class LabelStudioResource(dg.ConfigurableResource):
    """Label Studio client resource with MongoDB dependency"""
    api_token: str = os.environ.get("LS_TOK")
    mongodb: dg.ResourceDependency[MongoDBResource]
    
    def get_client(self) -> LabelStudioClient:
        return LabelStudioClient(
            api_token=self.api_token,
            mongodb_resource=self.mongodb
        )

class SemanticScholarResource(dg.ConfigurableResource):
    """Semantic scholar client resource """
    api_key: str = os.environ.get("S2")
    
    def get_client(self) -> LabelStudioClient:
        return SemanticScholarClient(
            api_key=self.api_key,
        )
    
@dg.definitions  
def resources():
    return dg.Definitions(
        resources={
            "duckdb": database_resource,
            "mongodb": MongoDBResource(),
            "s2_resource": SemanticScholarResource(),
            "ls_resource": LabelStudioResource()
        }
    )

