# """
# Description: Upload old/excel annotations on label studio, as if
#              they were done on the platform.
# Author: JSO
# """
# from pathlib import Path
# import pandas as pd
# import json
# import numpy as np

# from backend.clients.label_studio import LabelStudioClient

# # Read annotator data

# def read_round_1_annotations(user):
#   input_dir = Path("../../data/annots")
  
#   data2dump = []
  
#   if user == 'JL':
#     data = pd.read_csv("../../data/annots/2023-03-07/annotated/data2lab_2023-03-07_JL_labelled.csv")
#     data['sentiment'] = np.where(data.Yes == 1, 'yes', data.Yes)
#     data['sentiment'] = np.where(data.No == 1, 'no', data.sentiment)
#     data['sentiment'] = np.where(data.Maybe == 1, 'maybe', data.sentiment)
    
#     for i, row in data.iterrows():
#       data2dump.append({
#          "data": {
#             'corpusid': row['corpusid'],
#             'section': row['section'],
#             'subsection': row['subsection'],
#             'subsection_pos_in_pct': None,
#             'text': row['text']
#          },
#          "annotations": [{
#           'completed_by' : {
#               "id": 19456,
#               "first_name": "Juniper",
#               "last_name": "Lovato",
#               "avatar": None,
#               "email": "juniper.lovato@uvm.edu",
#               "initials": "ju"
#           },
#           'result': [{
#              'value': {'choices': [row['sentiment']]},
#              'id': "",
#              "from_name": 'sentiment',
#              'to_name': 'text',
#              'type': 'choices',
#               'origin': 'manual',
#           }]
#          }]
#       })     
  
#   elif user == 'AC':
#     with open(input_dir / '2023-03-07' / 'annotated' / f'data2lab_2023-03-07_AC_labelled.json')  as f:
#         data=json.loads(f.read())

#     old2new = {
#       'data availability statement': 'yes',
#       'not': 'no'
#     }
    
#     for annot in data:
#         old_annot = annot['annotations'][0]['result'][0]['value']['choices'][0]
#         new_annot = old2new[old_annot] if old_annot in old2new else old_annot
#         annot['annotations'][0]['result'][0]['value']['choices'] = [new_annot]
    
#         data2dump.append({
#         "data": annot['data'],
#         "annotations": [{
#           'completed_by' : {
#               "id": 17284,
#               "first_name": "Aviral",
#               "last_name": "Chawla",
#               "avatar": None,
#               "email": "achawla1@uvm.edu",
#               "initials": "AC"
#           },
#           'result': annot['annotations'][0]['result']
#         }],
#       })
        
#   elif user == 'CW':
#     with open(input_dir / '2023-03-07' / 'annotated' / f'data2lab_2023-03-07_CW_labelled.json')  as f:
#         data=json.loads(f.read())

#     old2new = {
#         "data availability statement": "yes",
#         "not": "no"
#      }
    
#     for annot in data:
#       if annot.get('sentiment') == None:
#          continue
#       old_annot = annot['sentiment']
#       new_annot = old2new[old_annot] if old_annot in old2new else old_annot
#       annot['sentiment'] = [new_annot]
      
#       data2dump.append({
#          "data": {
#             'corpusid': annot['corpusid'],
#             'section': annot['section'],
#             'subsection': annot['subsection'],
#             'section_pos_in_pct': annot['section_pos_in_pct'],
#             'text': annot['text']
#          },
#          "annotations": [{
#           'completed_by' : {
#               "id": 23575,
#               "first_name": "Carter",
#               "last_name": "Ward",
#               "avatar": None,
#               "email": "carterward4@gmail.com",
#               "initials": "CA"
#           },
#           'result': [{
#              'value': {'choices': [new_annot]},
#              'id': annot['id'],
#              "from_name": 'sentiment',
#              'to_name': 'text',
#              'type': 'choices',
#               'origin': 'manual',
#           }]
#          }]
#       })

#   elif user == 'JZ':
#       with open(input_dir / '2023-03-07' / 'annotated' / f'data2lab_2023-03-07_JZ_labelled.json')  as f:
#         data=json.loads(f.read())

      
#       old2new = {
#         'Maybe': 'maybe',
#         'Yes': 'yes',
#         'No': 'no'
#       }

#       for annot in data:
#         old_annot = annot['annotations'][0]['result'][0]['value']['choices'][0]
#         new_annot = old2new[old_annot] if old_annot in old2new else old_annot
#         annot['annotations'][0]['result'][0]['value']['choices'] = [new_annot]
    
#         data2dump.append({
#         "data": annot['data'],
#         "annotations": [{
#            'completed_by' : {
#               "id": 23576,
#               "first_name": "Julia",
#               "last_name": "Zimmerman",
#               "avatar": None,
#               "email": "jwzimmer1990@gmail.com",
#               "initials": "JZ"
#           },
#           'result': annot['annotations'][0]['result']
#         }],
#         })
      
#   return data2dump

# def read_round_2_annotations(user):
#   input_dir = Path("../../data/annots")
  
#   data2dump = []
  
#   if user == 'JL':
#     with open("../../data/annots/2023-09-21/annotated/sample_jn_JL_0921_labelled.json")  as f:
#         data=json.loads(f.read())
    
#     old2new = {
#         "Yes": "yes",
#         "No": "no",
#         "Maybe": "maybe",
#      }
    
#     for annot in data:
#       if annot.get('sentiment') == None:
#          continue
    
#       old_annot = annot['sentiment']
#       new_annot = old2new[old_annot] if old_annot in old2new else old_annot
#       annot['sentiment'] = [new_annot]
      
#       data2dump.append({
#          "data": {
#             'corpusid': annot['corpusid'],
#             'section': None,
#             'subsection': None,
#             'section_pos_in_pct': None,
#             'text': annot['text']
#          },
#          "annotations": [{
#           'completed_by' : {
#               "id": 19456,
#               "first_name": "Juniper",
#               "last_name": "Lovato",
#               "avatar": None,
#               "email": "juniper.lovato@uvm.edu",
#               "initials": "ju"
#           },
#           'result': [{
#              'value': {'choices': [new_annot]},
#              'id': annot['id'],
#              "from_name": 'sentiment',
#              'to_name': 'text',
#              'type': 'choices',
#               'origin': 'manual',
#           }]
#          }]
#       })
  
#   if user == 'AC':
#     with open("../../data/annots/2023-09-21/annotated/sample_jn_AC_0921_labelled.json")  as f:
#         data=json.loads(f.read())
    
#     for annot in data:
#       if annot.get('sentiment') == None:
#          continue
    
#       data2dump.append({
#          "data": {
#             'corpusid': annot['corpusid'],
#             'section': None,
#             'subsection': None,
#             'section_pos_in_pct': None,
#             'text': annot['text']
#          },
#          "annotations": [{
#           'completed_by' : {
#               "id": 17284,
#               "first_name": "Aviral",
#               "last_name": "Chawla",
#               "avatar": None,
#               "email": "achawla1@uvm.edu",
#               "initials": "AC"
#           },
#           'result': [{
#              'value': {'choices': [annot['sentiment']]},
#              'id': annot['id'],
#              "from_name": 'sentiment',
#              'to_name': 'text',
#              'type': 'choices',
#               'origin': 'manual',
#           }]
#          }]
#       })
  
#   if user == 'JSO':
#     with open("../../data/annots/2023-09-21/annotated/sample_jn_JSO_0921_labelled.json")  as f:
#         data=json.loads(f.read())
    
#     for annot in data:
#       if annot.get('sentiment') == None:
#          continue
    
      
#       data2dump.append({
#          "data": {
#             'corpusid': annot['corpusid'],
#             'section': None,
#             'subsection': None,
#             'section_pos_in_pct': None,
#             'text': annot['text']
#          },
#          "annotations": [{
#           'result': [{
#              'value': {'choices': annot['sentiment']},
#              'id': annot['id'],
#              "from_name": 'sentiment',
#              'to_name': 'text',
#              'type': 'choices',
#               'origin': 'manual',
#           }]
#          }]
#       })
  
#   if user == 'JZ':
#     data = pd.read_csv("../../data/annots/2023-09-21/annotated/sample_jn_JZ_0921_labelled.csv")
#     data = data[~data.is_data_availability.isna()] # 1na
#     data = data[['text', 'corpusid', 'is_data_availability', 'par_id', 'wc']]
#     data['lead_time'] = 0.
#     data.rename(columns={'is_data_availability': 'sentiment'}, inplace=True)
#     data['sentiment'] = data.sentiment.replace(1., "yes")
#     data['sentiment'] = data.sentiment.replace(2., "no")
#     data['sentiment'] = data.sentiment.replace(3., "maybe")
    
#     for i, row in data.iterrows():
#       data2dump.append({
#          "data": {
#             'corpusid': row['corpusid'],
#             'section': row['par_id'],
#             'subsection': None,
#             'subsection_pos_in_pct': None,
#             'text': row['text']
#          },
#          "annotations": [{
#             'completed_by' : {
#               "id": 23576,
#               "first_name": "Julia",
#               "last_name": "Zimmerman",
#               "avatar": None,
#               "email": "jwzimmer1990@gmail.com",
#               "initials": "JZ"
#           },
#           'result': [{
#              'value': {'choices': [row['sentiment']]},
#              'id': "",
#              "from_name": 'sentiment',
#              'to_name': 'text',
#              'type': 'choices',
#               'origin': 'manual',
#           }]
#          }]
#       })
    
#   return data2dump


# def main():
#   LS = LabelStudioClient()
  
#   for annot in ['AC', 'JL', 'CW', 'JZ', 'JSO']:
#     if annot != 'JSO':
#       data_round1 = read_round_1_annotations(annot)
      
#     if annot != 'CW':
#       data_round2 = read_round_2_annotations('JSO')

#     LS.post_LS(data_round1)
#     LS.post_LS(data_round2)

# if __name__ == '__main__':
#   main()