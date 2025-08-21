"""
LABEL STUDIO CLIENT
 - interact with our MONGODB
 - assisted LLM preannotations
"""
import requests
import json
import gc
from typing import List, Any, Union, Dict, ClassVar, Set
from pathlib import Path

from transformers import GenerationConfig, TextStreamer, pipeline
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import GenerationConfig, TextStreamer, pipeline
# from unsloth import FastLanguageModel

from outlines import Generator, from_transformers, Template

import pandas as pd
import numpy as np
from inspect import cleandoc
from tqdm import tqdm
from typing import Literal

try:
    import torch
    is_cuda = torch.cuda.is_available()
    print(f"runs on CUDA: {is_cuda}")
except:
    is_cuda = False

chosen_journals = set(['Journal of Business Research','Technological Forecasting and Social Change',
                      'Journal of Business Ethics','Advanced Materials', 'Angewandte Chemie', 
                      'Advanced Energy Materials', 'Neural Information Processing Systems', 
                      'International Conference on Learning Representations', 'Journal of Cleaner Production', 
                      'International Journal of Molecular Sciences', 'Nature Medicine', 
                      'BMJ', 'Synthese', 'Digital journalism', 
                      'Media, Culture & Society',  'Science of The Total Environment',  
                      'Nucleic Acids Research',  'International Journal of Molecular Sciences', 
                      'The astrophysical journal',  'Light-Science & Applications',  'Journal of High Energy Physics', 
                      'Nature Human Behaviour', 'Social Science & Medicine', 'Cities'])

class LabelStudioClient:
    model = None
    tokenizer = None
    model_path = None

    @classmethod
    def load_llama3(cls):
        if cls.model is None and cls.tokenizer is None and is_cuda:
            try:                
                # run on the VACC
                cls.model_path = "/gpfs1/llm/llama-3.2-hf/Meta-Llama-3.2-3B-Instruct"

                cls.model = from_transformers(
                    AutoModelForCausalLM.from_pretrained(cls.model_path, device_map="auto"),
                    AutoTokenizer.from_pretrained(cls.model_path)
                )
            except ImportError:
                cls.model = None
                cls.tokenizer = None
                raise ImportError("model and tokenizer not available") 

    #! TODO: fix with TRL
    # @classmethod
    # def load_llama3_finetuned(cls):
    #     if cls.model is None and cls.tokenizer is None and is_cuda:
    #         try:                
    #             cls.model_id = "jstonge1/dark-data-lora-balanced"

    #             cls.model, cls.tokenizer = FastLanguageModel.from_pretrained(
    #                     model_name = cls.model_id, 
    #                     max_seq_length = 2048,
    #                     dtype = None,
    #                     load_in_4bit = False,
    #                 )
                
    #             FastLanguageModel.for_inference(cls.model) 

    #         except ImportError:
    #             cls.model = None
    #             cls.tokenizer = None
    #             raise ImportError("model and tokenizer not available") 

    @classmethod
    def free_memory(cls):
        if cls.model is not None:
            del cls.model
            cls.model = None
        if cls.tokenizer is not None:
            del cls.tokenizer
            cls.tokenizer = None
        if cls.model_id is not None:
            del cls.model_id
            cls.model_id = None
        torch.cuda.empty_cache()
        gc.collect()

    def __init__(self, api_token: str, mongodb_resource):
        """
        Initialize LabelStudioClient with MongoDB connection and API token.
        
        Args:
            api_token: Label Studio API token (required)
            mongodb_resource: MongoDBResource instance (required)
        """
        # Configure MongoDB connection
        self.db = mongodb_resource.get_database()
        
        # Configure API token
        self.LS_TOK = api_token
        self.cache = Path("./cache")
        self.is_cuda = is_cuda
        self.annotators = {'juniper.lovato@uvm.edu': 19456, 'achawla1@uvm.edu': 17284, 'CW': 23575, 'JZ': 23576, 'jonathan.st-onge@uvm.edu': 16904}
        # self.active_annotators = {'juniper.lovato@uvm.edu':19456, 'achawla1@uvm.edu': 17284, 'jonathan.st-onge@uvm.edu': 16904}
        self.active_annotators = {'jonathan.st-onge@uvm.edu': 16904}

        if self.cache.exists() == False:
            self.cache.mkdir()

        print('accessing project status...')
        project_status = self.is_project_exists('Dark-Data')
        
        if project_status is None:
            self.proj_id = self.create_dark_data_project()
        else:
            self.proj_id = project_status
        
    # LABEL STUDIO HELPERS

    def get_annotations_LS(self, proj_id, only_annots=True):
        """Get annotations of a given project id."""
        headers = { "Authorization": f"Token {self.LS_TOK}" }
        
        url = f"https://cclabel.uvm.edu/api/projects/{proj_id}/export?exportType=JSON&download_all_tasks=true"
        print("requesting the annots...")
        response = requests.get(url, headers=headers, verify=False)

        if response.status_code == 200:
            json_data = json.loads(response.text)
            
            if only_annots:
                return [_ for _ in json_data if len(_['annotations']) > 0]
            else:
                return json_data
            
        else:
            print(f"Failed to fetch data: {response.status_code}")
            return []

    def post_LS(self, proj_id:int, data: List) -> None:
        """Import data to label studio"""
        response = requests.post(f'https://app.heartex.com/api/projects/{proj_id}/import', 
                                headers={'Content-Type': 'application/json', 'Authorization': f"Token {self.LS_TOK}"}, 
                                data=json.dumps(data), verify=False)
        print(response.status_code)

    # PROJECT HELPERS

    def is_project_exists(self, title: str) -> int:
        """if it exists, return project id, else return None"""
        headers = { "Authorization": f"Token {self.LS_TOK}"}
        response = requests.get("https://cclabel.uvm.edu/api/projects", headers=headers, verify=False)
        if response.status_code == 200:
            all_projects = json.loads(response.text)['results']
            proj_id = [_['id'] for _ in all_projects if _['title'] == title]
            if len(proj_id) > 0:
                return proj_id[0]

    def create_dark_data_project(self) -> int:
        """create dark data project"""
        
        project_config = """\
            {"title": "Dark-Data","label_config": "<View>\
            <View>\
            <Text name='text' value='$text'/>\
            <Choices name='sentiment' toName='text'>\
                <Choice value='yes'/>\
                <Choice value='no'/>\
                <Choice value='maybe'/>\
            </Choices>\
            </View>\
            </View>"}
            """
    
        response = requests.post(f'https://cclabel.uvm.edu/projects', 
                         headers={'Content-Type': 'application/json', 'Authorization': f"Token {self.LS_TOK}"}, 
                         data=project_config, verify=False)
        
        if response.text == 200:
            print("project created")
            return json.loads(response.text)['id']
    
        response = requests.post(f'https://cclabel.uvm.edu/projects', 
                         headers={'Content-Type': 'application/json', 'Authorization': f"Token {self.LS_TOK}"}, 
                         data=project_config, verify=False)

        print(response.status_code)

        if response.text == 200:
            print("project created")
            return json.loads(response.text)['id']

    def preannotate_with_llama3(self, annot: str, proj_id: int = 42):
        """Use Llama3 with structured generation for data availability statement classification""" 
        self.load_llama3()
        if self.model is None or self.tokenizer is None:
            raise ImportError("llama3 not available")

        if proj_id == 42:
            # Create outlines model
            model = self.model

            # Define the structured generation template
            template = Template.from_string(
                """You are an expert at analyzing scientific text for data availability statements.

A data availability statement describes where research data can be found, accessed, or obtained. This includes:
- References to data repositories or databases
- Statements about data sharing policies  
- Information about how to access the research data
- Contact information for data requests

# Examples

TEXT: We thank A. Sachraida, C. Gould and P. J. Kelly for providing us with the experimental data and helpful comments, and S. Tomsovic for a critical discussion.
RESULT: yes

TEXT: River discharge data for the Tully River were obtained from the Queensland Bureau of Meteorology (http://www.bom.gov.au). Data were obtained from the box centered on 17.5°S and 146°E from the SODA reanalysis project.
RESULT: yes

TEXT: The current results should be considered relative to a few study limitations. Officers may engage in proactive work that is not captured in these data.
RESULT: no

# Task

Analyze the following text and determine if it contains a data availability statement.
Answer with "yes", "no", or "maybe".

TEXT: {{ text }}
RESULT: """)

            # Create structured generator
            generator = Generator(model, Literal["yes", "no", "maybe"])
            
            # Generate structured response
            prompt = template(text=annot)
            result = generator(prompt, max_new_tokens=400, temperature=0.0, do_sample=False)
            
            return result
        else:
            raise NotImplementedError("Only project 42 (data availability) is implemented")
    
    #! TODO: fix with TRL
    # def run_llama3_finetuned(self, annot: str, proj_id:int = 42) -> List[Union[str, int]]:
    #     """run llama3 using few shot learning on the annotations for a given project""" 
    #     self.load_llama3_finetuned()
    #     # annot=row['text']
    #     if self.model is not None and self.tokenizer is not None:
    #         # set the configs
            
    #         # datas stuff
    #         alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

    #         ### Instruction:
    #         {}

    #         ### Inputs:
    #         {}

    #         ### Response:
    #         {}"""

    #         def parse_reply(x):
    #             if isinstance(x, list):
    #                 x=x[0]
    #             return 'yes' if x.split("Response:\n")[-1].replace("<|eot_id|>", "").strip() == '1' else 'no'
    #         if proj_id == 42:
                
    #             input = self.tokenizer(alpaca_prompt.format(
    #                     'Is data availability statement', # instruction
    #                     annot, # input
    #                     "", 
    #                 ), return_tensors = "pt").to("cuda")
    #             output = self.model.generate(**input, max_new_tokens = 64, use_cache = True)
    #             reply = self.tokenizer.batch_decode(output)
                
    #             # quirky, im sure there is a better way
    #             return parse_reply(reply)
                
    #         else:
    #             print("Not implemented yet")
    #             y_pred = None
        
    #         return y_pred
    #     else:
    #         raise ImportError("llama3 finetuned not available")

    # MONGODB HELPERS

    def read_cache_venue(self, keyword: str, N=500) -> pd.DataFrame:
        """read cache of venues with a certain threshold"""
        cache_f = self.cache / f"venues_{keyword}_{N}.csv"
        if cache_f.exists():
            return pd.read_csv(cache_f)

    def get_venues_more_than_N(self, keywords: List[str] = ["data"], N:int = 500) -> List[str]:
        """return venues with more than N occurences of a given keyword in text."""
        # the pipeline below takes time. We use caching when we can. 
        list_venues = self.read_cache_venue('_'.join(keywords), N=N)
        
        if list_venues is None:
            print("no cached filed found, querying mongoDB...")
            pipeline = [
                { "$match": {"text": {"$in": keywords}}  },
                {
                    '$group': {
                        '_id': { 'venue': '$venue' }, 
                        'count': { '$sum': 1 }
                    }
                },
                {
                    "$match": {
                        "count": {"$gt": N}
                } 
                },
                {
                    "$project" : {
                        "_id" : '$_id.venue'
                }
                }
            ]
            
            list_venues = pd.DataFrame(list(self.db.s2orc_dark_data.aggregate(pipeline)))
            
            #write to cache
            list_venues.to_csv(self.cache / f"venues_{'_'.join(keywords)}_{N}.csv", index=False)

        return list(set(list_venues['_id'].sort_values().tolist()))
        
    def get_sample_jn(self, keywords:str, jn:str, size:int, done_ids:Set = set()) -> pd.DataFrame:
        """get a sample of size from a given journal. Exclude done_ids."""
        pipeline = [
                { "$match":  { "venue": jn, "text": {"$in": keywords} }   } ,
                { "$sample": { "size": size } }
        ]
        print("querying DB for more pubs...")
        hits = pd.DataFrame(list(self.db.s2orc_dark_data.aggregate(pipeline)))
        
        if len(hits) > 0:
            return hits[~hits['corpusid'].isin(done_ids)]
        else:
            print("no more hits")
        
    # WORKING WITH LABEL STUDIO
    
    def more_annotations(self, proj_id: int = 42, only_annots=False, min_wc:int = 5, max_wc:int = 1000, sample_by_field:int = 1000) -> pd.DataFrame:
        """
        get more annotations for a given keyword.

        Parameters:
         - only_annots: if True, filter out annotated tasks else all tasks on Label Studio
         - min_wc: minimum word count of the text
         - max_wc: maximum word count of the text  
         - sample_by_field: number of samples to get per venue
        """
        print("get more annotations...")
        
        done_annots = self.get_annotations_LS(proj_id, only_annots=only_annots)
        done_corpusids = set([_['data']['corpusid'] for _ in done_annots]) if len(done_annots) > 0 else set()
        
        overshoot_sample = sample_by_field*2 # there must be a better way to do this. Rn we sample way more for DB to make sure we have
                                             # enough hits by keyword. This is a bit wasteful. 

        # We first get venues for which we have overshoot samples
        keywords = ['data'] if proj_id == 42 else ['code', 'software']
        print(f"getting venues with more than 2000 occurences of {' '.join(keywords)}...")
        all_jns_gt_thresh = self.get_venues_more_than_N(keywords=keywords, N=overshoot_sample)
        
        # Then we get more annotations for each venue
        out = []
        for jn in all_jns_gt_thresh:
            out.append(self.get_sample_jn(keywords, jn, size=overshoot_sample, done_ids=done_corpusids))
        new_annots = pd.concat(out, axis=0)

        # just making sure we don't have any overlap
        assert len(set(new_annots.corpusid.unique()) & done_corpusids) == 0
        
        # We filter by word count

        new_annots['text'] = new_annots['text'].map(lambda x: ' '.join(x))
        new_annots['wc'] = new_annots.text.str.split(" ").map(len)

        subset_d = new_annots.sample(frac=1, random_state=42)\
                    .groupby('venue').head(sample_by_field)\
                    .query(f'wc > {min_wc} & wc < {max_wc}').reset_index(drop=True)
        
        subset_d['corpusid_unique'] = subset_d.corpusid.astype(str) + '_' + subset_d.par_id.astype(str)
        
        return subset_d

    def create_annotation_task(self, row, annot_id: int, email: str, proj_id: int = 42, preannotate: str = None) -> tuple:
        """Create a complete annotation task with optional pre-annotation"""
        
        # Base task structure
        data_dict = {
            "data": {
                'corpusid': row['corpusid'],
                'corpusid_unique': row['corpusid_unique'],
                'par_id': row['par_id'],
                'wc': row['wc'],
                'text': row['text']
            },
            "annotations": [{ 
                "ground_truth": False,
                'completed_by': {
                    "id": annot_id,
                    "first_name": "",
                    "last_name": "",
                    "avatar": None,
                    "email": email,
                    "initials": email[:2]
                },
                'result': [{
                    'value': {'choices': []},
                    'id': "",
                    "from_name": 'sentiment',
                    'to_name': 'text',
                    'type': 'choices',
                    'origin': 'manual',
                }]
            }]
        }
        
        # Add pre-annotation if requested
        if preannotate == 'llama3':
            try:
                result = self.preannotate_with_llama3(row['text'], proj_id=proj_id)
                
                data_dict['predictions'] = [{
                    "model_version": "llama3-8B-structured",
                    "result": [{
                        "id": row.name if hasattr(row, 'name') else 0,
                        "from_name": 'sentiment',
                        'to_name': 'text',
                        "type": "labels",
                        'value': {'choices': [result]}
                    }]
                }]
                
                return data_dict, True  # Return tuple: (task, prediction_generated)
                
            except Exception as e:
                print(f"Pre-annotation failed for task: {e}")
                return data_dict, False
        
        return data_dict, False
    
