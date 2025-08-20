"""
S2ORC dark data collection asset.

Processes S2ORC scientific papers to extract text paragraphs containing specific keywords
and loads them into MongoDB for annotation tasks.
"""

import pandas as pd
import json
import spacy
import dagster as dg
from tqdm import tqdm
from pymongo import MongoClient
from pathlib import Path
import duckdb
import filelock


def get_journals_display_name(db, df_google):
    df_oa = pd.DataFrame(list(db.venues_oa.find( {}, 
        {"display_name": 1, "alternate_titles":1, "abbreviated_title": 1, "ids": 1}
    )))
    
    oa_display_name2googlepub = {
        'Review of Financial Studies': 'the review of financial studies',
        'Applied Catalysis B-environmental': 'applied catalysis b: environmental',
        'Journal of energy & environmental sciences': 'energy & environmental science',
        'Journal of materials chemistry. A, Materials for energy and sustainability': 'journal of materials chemistry a',
        'The American Economic Review': 'american economic review',
        'Quarterly Journal of Economics': 'the quarterly journal of economics',
        'Journal of Finance': 'the journal of finance',
        'Physical Review X': 'physical review. x',
        'European Physical Journal C': 'the european physical journal c',
        'Nature Reviews Molecular Cell Biology': 'nature reviews. molecular cell biology',
        'Journal of Religion & Health': 'journal of religion and health',
        'Lancet Oncology': 'the lancet oncology',
        'Lancet Infectious Diseases': 'the lancet infectious diseases',
        'Astronomy and Astrophysics': 'astronomy & astrophysics',
        'Light-Science & Applications': 'light: science & applications',
        'Energy research and social science': 'energy research & social science',
        'Global Environmental Change-human and Policy Dimensions': 'global environmental change',
        'Journalism: Theory, Practice & Criticism': 'journalism',
    }

    # remove duplicated? Why do we have duplicated in our db.
    df_oa = df_oa[~df_oa.display_name.duplicated()]

    # get everything lower case wihtout changing display_name.
    def lower_display_name(x):
        return [oa_display_name2googlepub[x].lower() 
                if oa_display_name2googlepub.get(x) 
                else x.lower() 
                for x in x.display_name]
    
    df_oa = df_oa.assign(display_name_lower = lower_display_name)

    df_oa['issn'] = df_oa.ids.map(lambda x: x['issn'][0] if x.get('issn') else None)

    return df_google.merge(df_oa, how="left", left_on="Publication", right_on="display_name_lower")

def add_journals_via_issnl(db, metadata_venue, df_google):
    # missing venues, if anyone wants to keep looking for them in the db.
    # mostly coming from computer science journals.
    missing_venues = metadata_venue[metadata_venue.display_name.isna()]
    metadata_venue = metadata_venue[~metadata_venue.display_name.isna()]
    
    list_issnl = ['1063-6919', '1364-0321', '2159-5399', '1520-6149']
    
    df_oa_issnl = pd.DataFrame(list(db.venues_oa.find(
        {"issn_l": {"$in": list_issnl}}, 
        {"display_name": 1, "alternate_titles":1, "abbreviated_title": 1, "ids": 1, 'issn_l': 1}
    )))
    
    missing_metadata_venue = df_google.loc[(df_google.Publication.isin(missing_venues.Publication.tolist())) & (df_google.Publication != 'ieee/cvf international conference on computer vision'),:]\
             .assign(issn_l = list_issnl)\
             .merge(df_oa_issnl, how="left", on="issn_l")

    return pd.concat([metadata_venue, missing_metadata_venue], axis=0).reset_index(drop=True)

def serialize_duckdb_query(duckdb_path: str, sql: str):
    """Execute SQL statement with file lock to guarantee cross-process concurrency."""
    lock_path = f"{duckdb_path}.lock"
    with filelock.FileLock(lock_path):
        conn = duckdb.connect(duckdb_path)
        try:
            result = conn.execute(sql)
            # For SELECT queries, fetch the results before closing connection
            if sql.strip().upper().startswith('SELECT'):
                return result.fetchall()
            return result
        finally:
            conn.close()


@dg.asset(
    kinds={"mongodb"}, 
    key=["target", "main", "s2orc_dark_data"],
    deps=["gscholar_venues"]
)
def s2orc_dark_data() -> dg.MaterializeResult:
    """Process S2ORC scientific papers to extract text paragraphs and load into MongoDB"""
    
    # MongoDB connection
    pw = "password"  # TODO: Move to environment variable
    uri = f"mongodb://cwward:{pw}@wranglerdb01a.uvm.edu:27017/?authSource=admin&readPreference=primary&appname=MongoDB%20Compass&directConnection=true&ssl=false"
    client = MongoClient(uri)
    db = client['papersDB']
    
    # Load Google Scholar venues data from DuckDB
    venues_query = "SELECT venue as Publication, h5_index, h5_median FROM main.gscholar_venues"
    venues_data = serialize_duckdb_query("/tmp/data_luminosity.duckdb", venues_query)
    
    df_google = pd.DataFrame(venues_data, columns=['Publication', 'h5_index', 'h5_median'])\
                  .assign(Publication = lambda x: x.Publication.str.lower())
    
    # Get venue metadata
    meta_f = Path("data/metadata_venue.csv")
    metadata_venue = pd.read_csv(meta_f) if meta_f.exists() else get_journals_display_name(db, df_google)
    metadata_venue = add_journals_via_issnl(db, metadata_venue, df_google)
    
    # Take care of duplication
    metadata_venue = metadata_venue[~metadata_venue.display_name.duplicated()]
    
    tot_venue_set = set(metadata_venue.display_name)
    
    # Load spaCy model for text processing
    nlp = spacy.load("en_core_web_trf", enable=["tok2vec"])

    # Process venues and extract text data
    total_processed = 0
    total_paragraphs = 0
    venues_processed = 0
    
    for i, venue in enumerate(tot_venue_set):
        print(f"Processing {venue} ({i+1}/{len(tot_venue_set)})")
        
        # Find all corpus IDs of parsed papers for this venue
        venue_df = pd.DataFrame(list(db.papers.aggregate([
            { "$match": { 
                'works_oa.host_venue.display_name': venue, 
                's2orc_parsed': True } 
            },
            { "$project": { "corpusid": 1 }}
            ])))
        
        if len(venue_df) > 0:
            
            corpusIds_we_want = venue_df.corpusid.tolist()
            total_processed += len(corpusIds_we_want)

            # Extract text paragraphs from S2ORC data
            text_subset = []
            for cid in tqdm(corpusIds_we_want, desc=f"Processing {venue}"):
                text = db.s2orc.find_one({'corpusid': cid})
                
                # Extract paragraphs using character locations
                if text is not None and text.get('content', {}).get('annotations', {}).get('paragraph') is not None:
                    current_text = text['content']['text']
                    section_headers_raw = text['content']['annotations'].get('sectionheader')
                    par_raw = text['content']['annotations']['paragraph']
                    
                    header_lookup = {}
                    if section_headers_raw:
                        headers_start_end = [(int(_['start']), int(_['end'])) for _ in json.loads(section_headers_raw)]
                        headers_title = [current_text[start:end] for start, end in headers_start_end]
                        header_lookup = {loc[0]: title for loc, title in zip(headers_start_end, headers_title)}
                        
                    par_start_end = [(int(_['start']), int(_['end'])) for _ in json.loads(par_raw)]
                    
                    new_text = []
                    for j, p in enumerate(par_start_end):
                        start_par, end_par = p[0], p[1]
                        if section_headers_raw:
                            for start_section, _ in header_lookup.items():
                                if start_par < start_section:
                                    new_text.append(current_text[start_par:end_par])
                                    break
                        else:
                            new_text.append(current_text[start_par:end_par])

                    # Process text with spaCy
                    if new_text:
                        docs = list(nlp.pipe(new_text))
                        
                        for j, doc in enumerate(docs):
                            tok_text = [w.text for w in doc]
                            text_subset.append({
                                'corpusid': cid, 
                                'venue': venue, 
                                'par_id': j, 
                                'text': tok_text
                            })
                                                   
            if len(text_subset) > 0:
                db.s2orc_dark_data.insert_many(text_subset)
                total_paragraphs += len(text_subset)
                
            venues_processed += 1
    
    client.close()
    
    return dg.MaterializeResult(
        metadata={
            "total_venues_processed": venues_processed,
            "total_papers_processed": total_processed,
            "total_paragraphs_extracted": total_paragraphs,
            "mongodb_collection": "s2orc_dark_data",
            "spacy_model": "en_core_web_trf",
            "processing_complete": True
        }
    )
