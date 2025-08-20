# from datetime import date
# from pathlib import Path
# import numpy as np
# import pandas as pd
# import argparse

# def parse_args():
#     parser = argparse.ArgumentParser("Data Downloader")
#     parser.add_argument(
#         "-i", "--input", type=Path, help="input directory", required=True
#     )
#     parser.add_argument(
#         "-o", "--output", type=Path, help="output directory", required=True
#     )
#     return parser.parse_args()



# def get_prop(df, cat):
#     assert 'sentiment' in df.columns, 'missing sent col'
#     assert len(set(['yes', 'no', 'maybe']) - set(df['sentiment'])) == 0, 'not the right col values'
#     return round(sum(df['sentiment'] == cat) / len(df) * 100, 2)

# def read_labelled_data():
#     sel_cols = ['text', 'corpusid', 'sentiment', 'section', 'subsection', 'section_pos_in_pct', 'lead_time']
    
#     ## ROUND 1

#     d_ac = pd.read_json(labelled_dat_dir / 'data2lab_2023-03-07_AC_labelled.json')
#     d_ac['annotator'] = 'ac'
#     d_ac['sentiment'] = [_[0]['result'][0]['value']['choices'][0] for _ in d_ac['annotations']]
#     d_ac['sentiment'] = d_ac['sentiment'].str.lower()
#     d_ac['sentiment'] = d_ac.sentiment.replace("data availability statement", "yes")
#     d_ac['sentiment'] = d_ac.sentiment.replace("not", "no")
#     d_ac['corpusid'] = d_ac['data'].map(lambda x: x['corpusid'])
#     d_ac['section'] = d_ac['data'].map(lambda x: x['section'])
#     d_ac['subsection'] = d_ac['data'].map(lambda x: x['subsection'])
#     d_ac['text'] = [_['text'] for _ in d_ac['data']]
    
#     d_cw = pd.read_json(labelled_dat_dir / 'data2lab_2023-03-07_CW_labelled.json')
#     d_cw['annotator'] = 'cw'
#     d_cw = d_cw[~d_cw.sentiment.isna()] # 1na
#     d_cw = d_cw[sel_cols]
#     d_cw['sentiment'] = d_cw.sentiment.replace("data availability statement", "yes")
#     d_cw['sentiment'] = d_cw.sentiment.replace("not", "no")
    
#     d_jz = pd.read_json(labelled_dat_dir / 'data2lab_2023-03-07_JZ_labelled.json')
#     d_jz['annotator'] = 'jz'
#     d_jz['sentiment'] = [_[0]['result'][0]['value']['choices'][0] for _ in d_jz['annotations']]
#     d_jz['text'] = [_['text'] for _ in d_jz['data']]
#     d_jz['section'] =  d_jz['data'].map(lambda x: x['section'])
#     d_jz['subsection'] =  d_jz['data'].map(lambda x: x['subsection'])
#     d_jz['sentiment'] = d_jz['sentiment'].str.lower()
#     d_jz['corpusid'] = d_jz['data'].map(lambda x: x['corpusid'])

#     d_jl = pd.read_csv(labelled_dat_dir / "data2lab_2023-03-07_JL_labelled.csv")
#     d_jl['annotator'] = 'jl'
#     d_jl['sentiment'] = np.where(d_jl.Yes == 1, 'yes', d_jl.Yes)
#     d_jl['sentiment'] = np.where(d_jl.No == 1, 'no', d_jl.sentiment)
#     d_jl['sentiment'] = np.where(d_jl.Maybe == 1, 'maybe', d_jl.sentiment)

#     all_true_pos = pd.concat([d_ac, d_cw, d_jz, d_jl], axis=0)[['text', 'sentiment', 'corpusid', 'section', 'subsection']].query("sentiment == 'yes'")
#     all_true_pos['corpusid_unique'] = all_true_pos.corpusid.astype(int).astype(str) + '_' + all_true_pos.section + '_' + all_true_pos.subsection.astype(int).astype(str)
#     all_true_pos.to_csv("pos_example_id.csv", index=False)

#     ### ROUND 2 

#     sel_cols = ['text', 'corpusid', 'sentiment', 'lead_time']

#     d_jso = pd.read_json(labelled_dat_dir / 'sample_jn_JSO_0921_labelled.json')
#     d_jso['annotator'] = 'jso'
#     d_jso = d_jso[~d_jso.sentiment.isna()] # 1na
#     d_jso = d_jso[sel_cols]
#     d_jso['sentiment'] = d_jso.sentiment.replace("data availability statement", "yes")
#     d_jso['sentiment'] = d_jso.sentiment.replace("not", "no")

#     d_ac = pd.read_json(labelled_dat_dir / 'sample_jn_AC_0921_labelled.json')
#     d_ac['annotator'] = 'ac'
#     d_ac = d_ac[~d_ac.sentiment.isna()] # 1na
#     d_ac = d_ac[sel_cols]
#     d_ac['sentiment'] = d_ac.sentiment.replace("data availability statement", "yes")
#     d_ac['sentiment'] = d_ac.sentiment.replace("not", "no")

#     d_jl = pd.read_json(labelled_dat_dir / 'sample_jn_JL_0921_labelled.json')
#     d_jl['annotator'] = 'jl'
#     d_jl = d_jl[~d_jl.sentiment.isna()] # 1na
#     d_jl = d_jl[sel_cols]
#     d_jl['sentiment'] = d_jl.sentiment.replace("data availability statement", "yes")
#     d_jl['sentiment'] = d_jl.sentiment.replace("not", "no")

#     d_jz = pd.read_csv(labelled_dat_dir / 'sample_jn_JZ_0921_labelled.csv')
#     d_jz['annotator'] = 'jz'
#     d_jz = d_jz[~d_jz.is_data_availability.isna()] # 1na
    
#     d_jz = d_jz[['text', 'corpusid', 'is_data_availability']]
#     d_jz['lead_time'] = 0.
#     d_jz.rename(columns={'is_data_availability': 'sentiment'}, inplace=True)
#     d_jz['sentiment'] = d_jz.sentiment.replace(1., "yes")
#     d_jz['sentiment'] = d_jz.sentiment.replace(2., "no")
#     d_jz['sentiment'] = d_jz.sentiment.replace(3., "maybe")

#     all_together = pd.concat([d_jso, d_ac, d_jl, d_jz])
#     all_together['sentiment'] = all_together['sentiment'].str.lower()
#     all_together.drop('lead_time', axis=1, inplace=True)
    
#     all_together.value_counts('sentiment')
    
#     all_together.to_parquet(labelled_dat_dir / "training_data_2023-12-07.parquet")


#     collective_truth = pd.read_csv("../output/collective_truth.csv")

#     collective_truth = collective_truth[['text', 'collective_score', 'corpusid']]
#     collective_truth.rename(columns={'collective_score': 'sentiment'}, inplace=True)
#     collective_truth['sentiment'] = np.where(collective_truth.sentiment == 1, 'yes', 'no')
    
#     collective_truth.to_parquet(labelled_dat_dir / "test_data_2023-12-07.parquet")



#     return d_cw, d_jz, d_jl, d_ac

# def read_collective_pos_labelled_data():
#     df = pd.read_csv(labelled_dat_dir / "collective sentiment labeling - pos_example_id.csv")
#     df['collective_score'] = df[['jso', 'jl', 'cw', 'cw', 'jz']].sum(axis=1) / 5
#     df['collective_score'] = np.where((df.collective_score <= 0.2) | (df.collective_score >= 8.), 1, 0)
#     embedding_df = pd.read_parquet("data2lab.parquet")
#     df.query('collective_score == 1').merge(embedding_df, on='corpusid_unique', how='left').to_parquet("pos_lab_data.parquet", index=False)
#     # checking consensus
#     len(df) - sum((df.collective_score > 0.) & (df.collective_score < 1.))
#     df[(df.collective_score > 0.) & (df.collective_score < 1.)]
#     df[(df.collective_score != 0.) & (df.collective_score != 1.)][['text', 'jso', 'jl', 'ac', 'cw', 'jz', 'collective_score']]

# def print_labels():
#     d_cw, d_jz, d_jl, d_ac = read_labelled_data()

#     print("Carter")
#     print(f"yes: {get_prop(d_cw, 'yes')}%")
#     print(f"no: {get_prop(d_cw, 'no')}%")
#     print(f"maybe: {get_prop(d_cw, 'maybe')}%")

#     print("Julia")
#     print(f"yes: {get_prop(d_jz, 'yes')}%")
#     print(f"no: {get_prop(d_jz, 'no')}%")
#     print(f"maybe: {get_prop(d_jz, 'maybe')}%")
    
#     print("Juniper")
#     print(f"yes: {get_prop(d_jl, 'yes')}%")
#     print(f"no: {get_prop(d_jl, 'no')}%")
#     print(f"maybe: {get_prop(d_jl, 'maybe')}%")

#     print("Avi")
#     print(f"yes: {get_prop(d_ac, 'yes')}%")
#     print(f"no: {get_prop(d_ac, 'no')}%")
#     print(f"maybe: {get_prop(d_ac, 'maybe')}%")
    
#     tot_yes = sum(d_cw.sentiment == 'yes') + sum(d_jl.sentiment == 'yes') + sum(d_jz.sentiment == 'yes') + sum(d_ac.sentiment == 'yes')
#     tot_no = sum(d_cw.sentiment == 'no') + sum(d_jl.sentiment == 'no') + sum(d_jz.sentiment == 'no') + sum(d_ac.sentiment == 'no')
#     tot_maybe = sum(d_cw.sentiment == 'maybe') + sum(d_jl.sentiment == 'maybe') + sum(d_jz.sentiment == 'maybe') + + sum(d_ac.sentiment == 'maybe')

#     print(f'yes/no/maybe: {tot_yes}/{tot_no}/{tot_maybe} (total labelled: {tot_yes+tot_no+tot_maybe})')

    
# def main():
#     """
#     Sampling procedure
#     ===================

#     1. Take a random sample from paper's collection in mongoDB, e.g.
#     ```{python}
#     `subset = list(db.papers.aggregate([{'$sample': {'size': 1_000_000}}]))`
#     ```
#     2. Take enough data by field (n=10_000)  so that we end up with end with many sections
#     containing `data` and find them in the s2orc's collection
#     ```{python}
#     df_subset = df_subset.groupby('field').head(10_000)
#     3. Keep the sections that contain at least one occurence of `data`
#     ```{python}
#     df = df[df.text.str.contains(' data ')]
#     ```
#     4. Keep 500 random sections by field
#     ```{python}
#     d.sample(frac=1, random_state=42).groupby('field').head(500)
#     ```
#     """
#     args = parse_args()
#     # INPUT_DIR = Path("../../data/processed")
#     INPUT_DIR = args.input
#     # OUTPUT_DIR = Path("../../data/annots/")
#     OUTPUT_DIR = args.output
    
#     # I don't know where clusteredDataStatements.csv comes from, but
#     # it must be derived from the `s2orc_dark_data` collection in the mongoDB.
#     # This collection was created by `import/create_s2orc_dark_data.py`.
#     d = pd.read_csv(INPUT_DIR / "clusteredDataStatements.csv")
    
#     MIN_WC, MAX_WC = 5, 400
#     SAMPLE_BY_FIELD = 500

#     subset_d = d.sample(frac=1, random_state=42)\
#                 .groupby('field').head(SAMPLE_BY_FIELD)\
#                 .query(f'wc > {MIN_WC} & wc < {MAX_WC}').reset_index(drop=True)
    

#     # subset_d2 = d.loc[~d.corpusid.isin(subset_d.corpusid.tolist()), :]\
#     #              .sample(frac=1, random_state=42)\
#     #              .groupby('field').head(SAMPLE_BY_FIELD)\
#     #              .query(f'wc > {MIN_WC} & wc < {MAX_WC}').reset_index(drop=True)
    

#     subset_d['corpusid_unique'] = subset_d.corpusid.astype(str) + '_' + subset_d.section + '_' + subset_d.subsection.astype(int).astype(str)

#     subset_d['section_pos_in_pct'] = subset_d.section.str.split("/")\
#                                              .map(lambda x: round(float(x[0]) / float(x[1]), 3))

#     subset_d['x_centered'] = subset_d.x - subset_d.x.mean()
#     subset_d['y_centered'] = subset_d.y - subset_d.y.mean()

#     subset_d.to_parquet(OUTPUT_DIR / "2023-03-07-data2lab.parquet")

#     available_corpusids = set(subset_d.corpusid_unique.tolist())
#     sel_cols = ['corpusid', 'section', 'subsection', 'section_pos_in_pct', 'text']
#     team  = ['JL', 'AC', 'CW', 'JZ']
    
#     for quadrant, ind in zip(range(4), team):
        
#         # little experiment
#         # each team member get a quadrant
#         if quadrant == 0:
#             subsubset = subset_d.query('x_centered > 0 & y_centered > 0')
#         elif quadrant == 1:
#             subsubset = subset_d.query('x_centered <= 0 & y_centered > 0')
#         elif quadrant == 2:
#             subsubset = subset_d.query('x_centered <= 0 & y_centered <= 0')
#         else:
#             subsubset = subset_d.query('x_centered > 0 & y_centered <= 0')
        
#         # We check if corpus id is available from general repository of unique corpus ids.
#         # It should
#         corpusids = subsubset[subsubset.corpusid_unique.isin(available_corpusids)].corpusid_unique
#         # Pick a list of 500
#         given_subset = np.random.choice(list(corpusids), 500)

#         # Grab these 500. For some reason we don't always get 500 exactly.
#         ind_subset = subset_d.loc[subset_d.corpusid_unique.isin(given_subset), sel_cols]
        
#         # write to disk
#         ind_subset.to_csv(OUTPUT_DIR / "to_annotate" / f'data2lab_{str(date.today())}_{ind}.csv', index=False)
        
#         # no more available
#         available_corpusids = available_corpusids - set(given_subset)

#     # We increase validation by giving same docs to different people
#     for ind in team:
#         rest_team = set(team) - set([ind])
#         other_dfs = []
#         for i in rest_team:
#             other_dfs.append(pd.read_csv(OUTPUT_DIR / "to_annotate" / f'data2lab_{str(date.today())}_{i}.csv').sample(15))
        
#         other_dfs = pd.concat(other_dfs, axis=0)
#         target_df = pd.read_csv(OUTPUT_DIR / "to_annotate" /f'data2lab_{str(date.today())}_{ind}.csv')
        
#         pd.concat([target_df, other_dfs], axis=0)\
#           .to_csv(OUTPUT_DIR / "to_annotate" / f'data2lab_{str(date.today())}_{ind}.csv', index=False)

# if __name__ == '__main__':
#     main()
