"""
LABEL STUDIO CLIENT
 - interact with our MONGODB
 - assisted LLM preannotations
"""
import os
import requests
import json
import gc
from typing import List, Any, Union, Dict, ClassVar, Set
from pathlib import Path

from transformers import GenerationConfig, TextStreamer, pipeline
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import GenerationConfig, TextStreamer, pipeline
# from unsloth import FastLanguageModel

from pymongo import MongoClient
import pandas as pd
import numpy as np
from inspect import cleandoc
from tqdm import tqdm

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
    model_id = None

    # @classmethod
    # def load_llama3(cls):
    #     if cls.model is None and cls.tokenizer is None and is_cuda:
    #         try:                
    #             cls.model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

    #             cls.tokenizer = AutoTokenizer.from_pretrained(cls.model_id)

    #             cls.model = AutoModelForCausalLM.from_pretrained(
    #                 cls.model_id,
    #                 torch_dtype=torch.float16,
    #                 device_map = 'auto'
    #             )
    #         except ImportError:
    #             cls.model = None
    #             cls.tokenizer = None
    #             raise ImportError("model and tokenizer not available") 

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

    def __init__(self, client: MongoClient = None):
        if client is None:
            uri="mongodb://cwward:password@wranglerdb01a.uvm.edu:27017/?authSource=admin&readPreference=primary&appname=MongoDB%20Compass&directConnection=true&ssl=false"
            client = MongoClient(uri)
        
        self.LS_TOK = os.getenv('LS_TOK')
        self.db = client['papersDB']
        self.cache = Path("./cache")
        self.is_cuda = is_cuda
        self.annotators = {'juniper.lovato@uvm.edu': 19456, 'achawla1@uvm.edu': 17284, 'CW': 23575, 'JZ': 23576, 'jonathan.st-onge@uvm.edu': 16904}
        # self.active_annotators = {'juniper.lovato@uvm.edu':19456, 'achawla1@uvm.edu': 17284, 'jonathan.st-onge@uvm.edu': 16904}
        self.active_annotators = {'jonathan.st-onge@uvm.edu': 16904}

        if self.cache.exists() == False:
            self.cache.mkdir()

        # print('accessing project status...')
        # project_status = self.is_project_exists('Dark-Data')
        # code_proj_status = self.is_project_exists('Code-Statement')
        
        # if project_status is None:
        #     self.proj_id = self.create_dark_data_project()
        # else:
        #     self.proj_id = project_status
        
        # if code_proj_status is None:
        #     self.code_proj_id = self.create_code_project()
        # else:
        #     self.code_proj_id = code_proj_status

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
        if self.is_dark_data_project_exists():
            return "project already exists"
        
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
    
    def create_code_project(self) -> int:
        """create code project. Based on https://codalab.lisn.upsaclay.fr/competitions/16935"""
        if self.is_project_exists('Code-Statement'):
            return "project already exists"
        
        project_config = """\
            {"title": "Code-Statement","label_config": "<View>\
            <View>\
            <Text name='text' value='$text'/>\
            <Choices name='sentiment' toName='text'>\
                <Choice value='usage'/>\
                <Choice value='creation'/>\
                <Choice value='mention'/>\
                <Choice value='unclear'/>\
                <Choice value='None'/>\
            </Choices>\
            </View>\
            </View>"}
            """
    
        response = requests.post(f'https://cclabel.uvm.edu/projects', 
                         headers={'Content-Type': 'application/json', 'Authorization': f"Token {self.LS_TOK}"}, 
                         data=project_config, verify=False)

        print(response.status_code)

        if response.text == 200:
            print("project created")
            return json.loads(response.text)['id']

    def run_llama3(self, annot: str, proj_id:int = 42, config: GenerationConfig = None) -> List[Union[str, int]]:
        """run llama3 using few shot learning on the annotations for a given project""" 
        self.load_llama3()
        if self.model is not None and self.tokenizer is not None:
            # set default config if none provided
            if config is None:
                generation_config = GenerationConfig.from_pretrained(self.model_id)
                generation_config.max_new_tokens = 512
                generation_config.temperature = 0.0001
                generation_config.do_sample = True

            streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
            stop_token_ids = [self.tokenizer.eos_token_id, self.tokenizer.convert_tokens_to_ids("<|eot_id|>")]

            llm = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                return_full_text=False,
                generation_config=generation_config,
                num_return_sequences=1,
                eos_token_id=stop_token_ids,
                streamer=streamer,
            )

            if proj_id == 42:
                ex1 = """We thank A. Sachraida, C. Gould and P. J. Kelly for providing us with the experimental data and helpful comments, and S. Tomsovic for a critical discussion."""
                ex2 = """River discharge data for the Tully River were obtained from the Queensland Bureau of Meteorology (http://www.bom.gov. au). No long-term in situ salinity data are available from King Reef; therefore, data from the Carton-Giese Simple Ocean Data Assimilation (SODA) reanalysis project were chosen as a longterm monthly resolution SSS dataset. This consists of a combination of observed and modeled data (Carton et al., 2000). Data were obtained from the box centered on 17.5°S (17.25°-17.75°) and 146°E (145.75°-146.25°). SODA version 1.4.2 extends from 1958 to 2001 and uses surface wind products from the European Center for Medium-Range Weather Forecasts 40-year reanalysis (ECMWF ERA 40), which may contain inaccuracies in tropical regions (Cahyarini et al., 2008). The most recent version of SODA (1.4.3) now uses wind data from the Quick-Scat scatterometer, thus providing more accurate data for the tropics (Cahyarini et al., 2008;Carton and Giese, 2008"""
                ex3 = """The current results should be considered relative to a few study limitations. The CFS data did not specify the nature of proactive activities that patrol, DRT officers, or investigators were engaged in. Furthermore, although the coding of the ten call categories analyzed were informed by prior research (Wu & Lum, 2017), idiosyncrasies associated with the study departments' method of cataloging and recording call information did not always allow for direct comparisons to prior research on COVID-19's impact on police services. Similarly, measuring proactivity solely through self-initiated activities from CFS data is not a flawless indicator. Officers may engage in proactive work that is not captured in these **data** (Lum, Koper, et al., 2020). However, this method has been established as a reasonable way to distinguish proactivity from reactivity (Lum, Koper, et al., 2020;Wu & Lum, 2017;Zhang and Zhao, 2021)."""

                return llm([{
                    "role": "user",
                    "content": cleandoc(f"""
                        Text: {ex1}
                        is_data_availability_statement: yes

                        Text: {ex2}
                        is_data_availability_statement: yes

                        Text: {ex3}
                        is_data_availability_statement: no

                        Text: {annot}
                        is_data_availability_statement: ?

                        Give a one word response.
                        """
                    )}])
            else:
                print("Not implemented yet")
                y_pred = None
        
            return y_pred
        else:
            raise ImportError("llama3 not available")
    
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
    
    def dispatch_annots(self, proj_id:int = 42, N:int = 200, preannotate: str = 'llama3'):
        """dispatch N annotations to each (active) annotator."""
        annots_to_dispatch = self.more_annotations(proj_id=proj_id)

        for email, annot_id in self.active_annotators.items():
            print(f"dispatching to {email}")
            next_corpus_id = np.random.choice(annots_to_dispatch.corpusid_unique, N)
            next_sample_df = annots_to_dispatch[annots_to_dispatch.corpusid_unique.isin(next_corpus_id)]

            data2dump = []
            for i, row in next_sample_df.iterrows():
                
                data_dict = {
                    "data": {
                        'corpusid': row['corpusid'],
                        'corpusid_unique': row['corpusid_unique'],
                        'par_id': row['par_id'],
                        'wc': row['wc'],
                        'text': row['text']
                    },
                    # we add empty annotations as a way to assign the task to the annotator. Not ideal.
                    "annotations": [{ 
                        "ground_truth": False,
                        'completed_by' : {
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

                if preannotate:
                    # slow because we don't batch. Makes it simpler.
                    y_pred = self.run_llama3(row['text'], proj_id=proj_id)
                    y_pred = y_pred[0]['generated_text'].lower()
                    # y_pred = self.run_llama3_finetuned(row['text'], proj_id=self.proj_id)

                    data_dict['predictions'] = [{
                        "model_version": "llama3-8B-few-shots",
                        "score": 0.5, # could be ammended to have uncertainty score from llama3
                        "result": [
                            {
                                "id": i,
                                "from_name": 'sentiment',
                                'to_name': 'text',
                                "type": "labels",
                                'value': { 'choices': [y_pred] }
                            }
                        ]
                    }]
                
                data2dump.append(data_dict)     
            
            self.post_LS(proj_id, data_dict)

        annots_to_dispatch = annots_to_dispatch[~annots_to_dispatch.corpusid_unique.isin(next_corpus_id)]

def format_data_dict_LS(row, annot_id, email):
    return {
            "data": {
                'corpusid': row['corpusid'],
                'corpusid_unique': row['corpusid_unique'],
                'par_id': row['par_id'],
                'wc': row['wc'],
                'text': row['text']
            },

            # we add empty annotations as a way to assign the task to the annotator. Not ideal.
            "annotations": [{ 
                "ground_truth": False,
                'completed_by' : {
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

def format_prediction_LS(y_pred:str, model_version:str) -> List:
    return [{
            "model_version": model_version,
            "score": 0.5, # could be ammended to have uncertainty score from llama3
            "result": [
                {
                    "id": i,
                    "from_name": 'sentiment',
                    'to_name': 'text',
                    "type": "labels",
                    'value': { 'choices': [y_pred] }
                }
            ]
        }]

