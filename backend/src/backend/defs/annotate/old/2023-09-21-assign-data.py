# # import jsonlines
# import pandas as pd
# import re
# import json

# from pathlib import Path
# from creds import client
# from tqdm import tqdm


# ROOT_DIR = Path("..")
# OUTPUT_DIR = ROOT_DIR / "output"
# MISC_DIR = ROOT_DIR / "etc"

# db = client['papersDB']

# list(db.s2orc_dark_data.aggregate([
#     { "$match": {"text": "data"}  },
#     { "$limit": 1 }
#     ])
# )

# pipeline = [
#     { "$match": {"text": "data"}  },
#     {
#          '$group': {
#                '_id': { 'venue': '$venue' }, 
#                'count': { '$sum': 1 }
#          }
#     },
#     {
#           "$match": {
#             "count": {"$gt": 500}
#       } 
#     },
#     {
#         "$project" : {
#             "_id" : '$_id.venue'
#       }
#     }
#    ]

# list_venues = pd.DataFrame(list(db.s2orc_dark_data.aggregate(pipeline)))
# all_jns_gt_thresh = set(list_venues['_id'].sort_values().tolist())

# chosen_journals = set(['Journal of Business Research','Technological Forecasting and Social Change',
#                   'Journal of Business Ethics','Advanced Materials', 'Angewandte Chemie', 
#                   'Advanced Energy Materials', 'Neural Information Processing Systems', 
#                   'International Conference on Learning Representations', 'Journal of Cleaner Production', 
#                   'International Journal of Molecular Sciences', 'Nature Medicine', 
#                   'BMJ', 'Synthese', 'Digital journalism', 
#                   'Media, Culture & Society',  'Science of The Total Environment',  
#                   'Nucleic Acids Research',  'International Journal of Molecular Sciences', 
#                    'The astrophysical journal',  'Light-Science & Applications',  'Journal of High Energy Physics', 
#                    'Nature Human Behaviour', 'Social Science & Medicine', 'Cities'])

# chosen_journals - all_jns_gt_thresh


# out = []
# for jn in chosen_journals:
#     pipeline2 = [
#         { "$match":  { "venue": jn, "text": "data" }   } ,
#         { "$sample": { "size": 2000 } }
#     ]

#     out.append(pd.DataFrame(list(db.s2orc_dark_data.aggregate(pipeline2))))

# out = pd.concat(out, axis=0)

# out['text'] = out['text'].map(lambda x: ' '.join(x))

# out = out.reset_index(drop=True)

# MIN_WC, MAX_WC = 5, 1000
# SAMPLE_BY_FIELD = 1000

# out['wc'] = out.text.str.split(" ").map(len)

# subset_d = out.sample(frac=1, random_state=42)\
#                 .groupby('venue').head(SAMPLE_BY_FIELD)\
#                 .query(f'wc > {MIN_WC} & wc < {MAX_WC}').reset_index(drop=True)
    
  
# subset_d['corpusid_unique'] = subset_d.corpusid.astype(str) + '_' + subset_d.par_id.astype(str)

# import numpy as np
# corpusid_JSO = np.random.choice(subset_d.corpusid_unique, 3000)
# sample_JSO = subset_d[subset_d.corpusid_unique.isin(corpusid_JSO)]

# subset_d = subset_d[~subset_d.corpusid_unique.isin(corpusid_JSO)]

# corpusid_JL = np.random.choice(subset_d.corpusid_unique, 3000)
# sample_JL = subset_d[subset_d.corpusid_unique.isin(corpusid_JL)]

# subset_d = subset_d[~subset_d.corpusid_unique.isin(corpusid_JL)]

# corpusid_AC = np.random.choice(subset_d.corpusid_unique, 3000)
# sample_AC = subset_d[subset_d.corpusid_unique.isin(corpusid_AC)]

# subset_d = subset_d[~subset_d.corpusid_unique.isin(corpusid_AC)]
  

# corpusid_CW = np.random.choice(subset_d.corpusid_unique, 3000)
# sample_CW = subset_d[subset_d.corpusid_unique.isin(corpusid_CW)]

# subset_d = subset_d[~subset_d.corpusid_unique.isin(corpusid_CW)]
  
# sample_JSO.to_csv("sample_jn_JSO_0921.csv", index=False)
# sample_JL.to_csv("sample_jn_JL_0921.csv", index=False)
# sample_AC.to_csv("sample_jn_AC_0921.csv", index=False)
# sample_CW.to_csv("sample_jn_CW_0921.csv", index=False)