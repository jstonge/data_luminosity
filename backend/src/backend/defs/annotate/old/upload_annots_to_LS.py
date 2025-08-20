
# import dagster as dg
# from backend.defs.resources import LabelStudioResource
# from dagster_duckdb import DuckDBResource
# from typing import List, Tuple, Optional
# from tqdm import tqdm
# import numpy as np


# def format_data_dict_LS(row, annot_id, email):
#     return {
#             "data": {
#                 'corpusid': row['corpusid'],
#                 'corpusid_unique': row['corpusid_unique'],
#                 'par_id': row['par_id'],
#                 'wc': row['wc'],
#                 'text': row['text']
#             },

#             # we add empty annotations as a way to assign the task to the annotator. Not ideal.
#             "annotations": [{ 
#                 "ground_truth": False,
#                 'completed_by' : {
#                     "id": annot_id,
#                     "first_name": "",
#                     "last_name": "",
#                     "avatar": None,
#                     "email": email,
#                     "initials": email[:2]
#                 },
#                 'result': [{
#                     'value': {'choices': []},
#                     'id': "",
#                     "from_name": 'sentiment',
#                     'to_name': 'text',
#                     'type': 'choices',
#                     'origin': 'manual',
#                 }]
#                 }]
#         }

# def format_prediction_LS(y_pred:str, model_version:str) -> List:
#     return [{
#             "model_version": model_version,
#             "score": 0.5, # could be ammended to have uncertainty score from llama3
#             "result": [
#                 {
#                     "id": i,
#                     "from_name": 'sentiment',
#                     'to_name': 'text',
#                     "type": "labels",
#                     'value': { 'choices': [y_pred] }
#                 }
#             ]
#         }]



# def dispacth(LS: LabelStudioResource, author_id: str) -> Tuple[Optional[int], Optional[int], int]:
#     new_annots = LS.more_annotations(42)

#     for email, annot_id in LS.active_annotators.items():
#         print(f"dispatching to {email}")
#         next_corpus_id = np.random.choice(new_annots.corpusid_unique, 200)
#         next_sample_df = new_annots[new_annots.corpusid_unique.isin(next_corpus_id)]
#         data2dump = []
#         for i, row in tqdm(next_sample_df.iterrows()):
#             data_dict = format_data_dict_LS(row, annot_id, email)
#             y_pred = LS.run_llama3(row['text'], proj_id=LS.proj_id)
#             y_pred = y_pred[0]['generated_text'].lower()
#             data_dict['predictions'] = format_prediction_LS(y_pred, "llama3-8B")
#             data2dump.append(data_dict)     

#     LS.free_memory()
#     LS.model, LS.model_id, LS.tokenizer

#     for i in range(len(data2dump)):
#         annot = data2dump[i]['data']['text']
#         y_pred = LS.run_llama3_finetuned(annot, proj_id=LS.proj_id)
#         data2dump[i]['predictions'] += format_prediction_LS(y_pred, "llama3-8B-finetuned")

#     LS.free_memory()


#     data2dump_pos = [
#         _ for _ in data2dump if
#         _['predictions'][1]['result'][0]['value']['choices'][0] == 'yes'
#     ]

#     LS.post_LS(42, data2dump_pos)

#     # annot = "The current results should be considered relative to a few study limitations. The CFS **data** did not specify the nature of proactive activities that patrol, DRT officers, or investigators were engaged in. Furthermore, although the coding of the ten call categories analyzed were informed by prior research (Wu & Lum, 2017), idiosyncrasies associated with the study departments' method of cataloging and recording call information did not always allow for direct comparisons to prior research on COVID-19's impact on police services. Similarly, measuring proactivity solely through self-initiated activities from CFS **data** is not a flawless indicator. Officers may engage in proactive work that is not captured in these **data** (Lum, Koper, et al., 2020). However, this method has been established as a reasonable way to distinguish proactivity from reactivity (Lum, Koper, et al., 2020;Wu & Lum, 2017;Zhang and Zhao, 2021)."
#     # annot = "to the same scales as the soluble silica. 1 In what follows numerical **data** relating to the Chalk, unless otherwise stated, have been obtained from A. J. Jukes-Browne, Cretaceous Rocks of Britain, Mem. Geol. Surv. ; and from the papers on the White Chalk of the Knglish Coast in the Proc. Geol. Assoc. (1899-1903), by A. B. Rowe and CD. Sherborn.2 B. Moore, Trans. Geol. Physics Soc, 1917, p. 1.3 W. J. Sollas, The Age of the Earth, London, 1905, pp. 132-65.r-"
#     # annot = "The metagenomes are deposited in European Bioinformatics Institute European Nucleotide Archive under accession no . PRJEB39223 . The non - metagenomic data used for analysis in this study are held by the Department of Twin Research at King 's College London . The data can be released to bona fide researchers using our normal procedures overseen by the Wellcome Trust and its guidelines as part of our core funding . We receive around 100 requests per year for our datasets and have three meetings per month with independent members to assess proposals . The application can be found at https://twinsuk.ac.uk/resources-for-researchers/ access - our - data/. This means that data need to be anonymized and conform to GDPR standards ."

#     # reply = LS.run_llama3_finetuned(annot, proj_id=LS.proj_id)

#     annots_to_dispatch = new_annots[~new_annots.corpusid_unique.isin(next_corpus_id)]