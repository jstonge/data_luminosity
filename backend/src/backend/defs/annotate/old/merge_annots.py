# import argparse
# from pathlib import Path
# import pandas as pd
# import numpy as np

# def parse_args():
#     parser = argparse.ArgumentParser("Data Downloader")
#     parser.add_argument(
#         "-i", "--input", type=Path, help="annotated data directory", required=True
#     )
#     parser.add_argument(
#         "-o", "--output", type=Path, help="output directory", required=True
#     )
#     return parser.parse_args()

# def merge_round_1():
#     sel_cols = ['text', 'corpusid', 'sentiment', 'section', 'subsection', 'section_pos_in_pct', 'lead_time']

#     # Avi
#     d_ac = pd.read_json(input_dir / '2023-03-07' / 'annotated' / 'data2lab_2023-03-07_AC_labelled.json')
#     d_ac['annotator'] = 'ac'
#     d_ac['sentiment'] = [_[0]['result'][0]['value']['choices'][0] for _ in d_ac['annotations']]
#     d_ac['sentiment'] = d_ac['sentiment'].str.lower()
#     d_ac['sentiment'] = d_ac.sentiment.replace("data availability statement", "yes")
#     d_ac['sentiment'] = d_ac.sentiment.replace("not", "no")
#     d_ac['corpusid'] = d_ac['data'].map(lambda x: x['corpusid'])
#     d_ac['section'] = d_ac['data'].map(lambda x: x['section'])
#     d_ac['subsection'] = d_ac['data'].map(lambda x: x['subsection'])
#     d_ac['text'] = [_['text'] for _ in d_ac['data']]

#     # Carter    
#     d_cw = pd.read_json(input_dir / '2023-03-07' / 'annotated' / 'data2lab_2023-03-07_CW_labelled.json')
#     d_cw['annotator'] = 'cw'
#     d_cw = d_cw[~d_cw.sentiment.isna()] # 1na
#     d_cw = d_cw[sel_cols]
#     d_cw['sentiment'] = d_cw.sentiment.replace("data availability statement", "yes")
#     d_cw['sentiment'] = d_cw.sentiment.replace("not", "no")
    
#     # Julia
#     d_jz = pd.read_json(input_dir / '2023-03-07' / 'annotated' / 'data2lab_2023-03-07_JZ_labelled.json')
#     d_jz['annotator'] = 'jz'
#     d_jz['sentiment'] = [_[0]['result'][0]['value']['choices'][0] for _ in d_jz['annotations']]
#     d_jz['text'] = [_['text'] for _ in d_jz['data']]
#     d_jz['section'] =  d_jz['data'].map(lambda x: x['section'])
#     d_jz['subsection'] =  d_jz['data'].map(lambda x: x['subsection'])
#     d_jz['sentiment'] = d_jz['sentiment'].str.lower()
#     d_jz['corpusid'] = d_jz['data'].map(lambda x: x['corpusid'])
    
#     # Juni
#     d_jl = pd.read_csv(input_dir / '2023-03-07' / 'annotated' / "data2lab_2023-03-07_JL_labelled.csv")
#     d_jl['annotator'] = 'jl'
#     d_jl['sentiment'] = np.where(d_jl.Yes == 1, 'yes', d_jl.Yes)
#     d_jl['sentiment'] = np.where(d_jl.No == 1, 'no', d_jl.sentiment)
#     d_jl['sentiment'] = np.where(d_jl.Maybe == 1, 'maybe', d_jl.sentiment)

#     return pd.concat([d_ac, d_cw, d_jz, d_jl], axis=0)[['text', 'sentiment', 'corpusid', 'section', 'subsection']]

# def merge_round_2():
#     sel_cols = ['text', 'corpusid', 'sentiment', 'lead_time']

#     # Jso
#     d_jso = pd.read_json(input_dir / '2023-09-21' / 'annotated' / 'sample_jn_JSO_0921_labelled.json')
#     d_jso['annotator'] = 'jso'
#     d_jso = d_jso[~d_jso.sentiment.isna()] # 1na
#     d_jso = d_jso[sel_cols]
#     d_jso['sentiment'] = d_jso.sentiment.replace("data availability statement", "yes")
#     d_jso['sentiment'] = d_jso.sentiment.replace("not", "no")

#     # Avi
#     d_ac = pd.read_json(input_dir / '2023-09-21' / 'annotated' / 'sample_jn_AC_0921_labelled.json')
#     d_ac['annotator'] = 'ac'
#     d_ac = d_ac[~d_ac.sentiment.isna()] # 1na
#     d_ac = d_ac[sel_cols]
#     d_ac['sentiment'] = d_ac.sentiment.replace("data availability statement", "yes")
#     d_ac['sentiment'] = d_ac.sentiment.replace("not", "no")

#     # Juni
#     d_jl = pd.read_json(input_dir / '2023-09-21' / 'annotated' / 'sample_jn_JL_0921_labelled.json')
#     d_jl['annotator'] = 'jl'
#     d_jl = d_jl[~d_jl.sentiment.isna()] # 1na
#     d_jl = d_jl[sel_cols]
#     d_jl['sentiment'] = d_jl.sentiment.replace("data availability statement", "yes")
#     d_jl['sentiment'] = d_jl.sentiment.replace("not", "no")

#     # Julia
#     d_jz = pd.read_csv(input_dir / '2023-09-21' / 'annotated' / 'sample_jn_JZ_0921_labelled.csv')
#     d_jz['annotator'] = 'jz'
#     d_jz = d_jz[~d_jz.is_data_availability.isna()] # 1na
#     d_jz = d_jz[['text', 'corpusid', 'is_data_availability']]
#     d_jz['lead_time'] = 0.
#     d_jz.rename(columns={'is_data_availability': 'sentiment'}, inplace=True)
#     d_jz['sentiment'] = d_jz.sentiment.replace(1., "yes")
#     d_jz['sentiment'] = d_jz.sentiment.replace(2., "no")
#     d_jz['sentiment'] = d_jz.sentiment.replace(3., "maybe")

#     return pd.concat([d_jso, d_ac, d_jl, d_jz])


# def main():
#     round1_merged = merge_round_1()
#     round1_merged.value_counts('sentiment')
    
#     # all_true_pos = round1_merged.query("sentiment == 'yes'")
#     # all_true_pos['corpusid_unique'] = all_true_pos.corpusid.astype(int).astype(str) + '_' + all_true_pos.section + '_' + all_true_pos.subsection.astype(int).astype(str)
#     # all_true_pos.to_csv("pos_example_id.csv", index=False)
    
#     round2_merged = merge_round_2()
#     round2_merged['sentiment'] = round2_merged['sentiment'].str.lower()
#     round2_merged.drop('lead_time', axis=1, inplace=True)

#     round2_merged.value_counts('sentiment')
    
#     pd.concat([round1_merged,round2_merged], axis=0)\
#       .reset_index(drop=True) \
#       .to_parquet(output_dir / "annotated_data.parquet")



# if __name__ == "__main__":
#     args = parse_args()
#     # input_dir = Path("../../data/annots")
#     input_dir = args.input
#     # output_dir = Path("../../data/annots")
#     output_dir = args.output

#     main()
