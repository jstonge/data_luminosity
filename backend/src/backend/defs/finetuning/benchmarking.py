import httpx
import os
import pandas as pd
from pydantic import BaseModel
from tqdm import tqdm
import json
from backend.defs.resources import DuckDBResource
import dagster as dg
from pydantic import BaseModel, Field
from pathlib import Path
import ollama

from sklearn.metrics import confusion_matrix, classification_report

from label_studio_sdk.client import LabelStudio
from label_studio_sdk.label_interface import LabelInterface
from label_studio_sdk.label_interface.create import choices
from label_studio_sdk.data_manager import Filters, Column, Type, Operator

from transformers import AutoModelForCausalLM, AutoTokenizer

import outlines
from outlines import Generator, from_transformers, Template


def load_data(duckdb: DuckDBResource) -> dg.MaterializeResult:
    with duckdb.get_connection() as conn:
        df = conn.execute("SELECT * FROM main.deduplicated_annotations").df()
    return df

def initialize_project(ls) -> None:
    # Define labeling interface
    label_config = LabelInterface.create({
        'text': 'Text',
        'label': choices(['yes', 'no'])
    })

    project = ls.projects.create(
        title='Data availability statements 2.0',
        label_config=label_config
    )

    df = load_data()

    df = df[(df.has_data_statement == 'yes') | (df.has_data_statement == 'no')]
    df_yes = df[df.has_data_statement=='yes']
    df_no = df[df.has_data_statement=='no']

    ls.projects.import_tasks(
        id=project.id,
        request=pd.concat([df_yes.sample(50), df_no.sample(50)], axis=0).to_dict('records'),
        preannotated_from_fields=['has_data_statement']
    )

    return project

def sklearn():
    """predict based on sklearn methods"""
    pass

def format_prediction(result, prompt_path):
    return [
            {
                "id": "abc",
                "from_name": "label",
                "to_name": "text",
                "type": "choices",
                "prompt_filename": str(prompt_path),
                "value": { "choices": [ 
                    'yes' if result['has_data_availability_statement'] else 'no'               
                ]}
            }
        ]

def zero_to_many_shots(input_text, client, model_name, prompt_path):
    
    # define structure and produce generator
    class Classification(BaseModel):
        has_data_availability_statement: bool 
        justification: str = Field(description="Why?")

    model = outlines.from_ollama(client, model_name)
    generator = Generator(model, Classification)
    
    # load template
    stance_template = Template.from_file(prompt_path)
    prompt = stance_template(text=input_text)

    # get result
    result = generator(prompt)

    return result

@dg.asset(
    kinds={"duckdb"}, 
    deps=["semantic_deduplication"],
    group_name="finetuning"
)
def benchmarking():
    LABEL_STUDIO_URL = 'https://cclabel.uvm.edu/'
    API_KEY = os.environ['LS_TOK']
    ls = LabelStudio(base_url=LABEL_STUDIO_URL, api_key=API_KEY, httpx_client=httpx.Client(verify=False))
    
    projects = ls.projects.list()
    projects
    project = ls.projects.get(id=60)
    # initialize model
    client = ollama.Client()

    project = initialize_project(ls)
    
    tasks = ls.tasks.list(project=project.id)

    timing = {}
    for model_name in ["llama3.2:3b", "qwen3:32b"]:
        # model_name="llama3.2:3b"
        prompt_path = Path("./templates/zero_shot_01.txt")
        
        for task in tqdm(tasks, total=len(tasks.items)):
            try:
                result = json.loads(zero_to_many_shots(task.data['text'], client, model_name, prompt_path))

                prediction = format_prediction(result, prompt_path)

                ls.predictions.create(
                    task=task.id, 
                    model_version=f"{prompt_path.stem}-{model_name}", 
                    score=0.5, 
                    result=prediction
                    )
            except:
                print(f"{task.id} has failed.")
    
    # timing["qwen3:32b"] = {"time":'24:31', "it_by_second":"14.71"}
    # timing[model_name] = {"time":'24:31', "it_by_second":"14.71"}