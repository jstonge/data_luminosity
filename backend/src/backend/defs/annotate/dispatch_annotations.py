"""
Annotation dispatch asset for Label Studio.

Dispatches new annotation tasks from S2ORC dark data to Label Studio annotators,
with optional pre-annotation using Llama3.
"""

import dagster as dg
from backend.defs.resources import LabelStudioResource


@dg.asset(
    kinds={"label_studio"}, 
    deps=["s2orc_dark_data"],
    group_name="annotate"
)
def dispatch_annotations(ls_client: LabelStudioResource) -> dg.MaterializeResult:
    """
    Dispatch data availability annotation tasks to Label Studio annotators.
    
    Equivalent to the Makefile command:
    cd src && python -c "from label_studio import labelStudio; LS = labelStudio(); LS.dispatch_annots(keyword='data')"
    
    Configuration:
    - proj_id: 42 (Dark Data project)
    - N: 200 (annotations per annotator) 
    - preannotate: 'llama3'
    """
    
    # Configuration - these could be made configurable via Dagster config if needed
    proj_id = 42
    N = 200
    preannotate = 'llama3'
    
    # Initialize Label Studio client
    ls_client.LS_TOK = '8ecf272719351405d5c1dee84c97a9c304a9e96e'
    
    try:
        # Get new annotations to dispatch
        annots_to_dispatch = ls_client.more_annotations(proj_id=proj_id)
        
        total_dispatched = 0
        dispatch_results = {}
        
        # Dispatch to each active annotator
        for email, annot_id in ls_client.active_annotators.items():
            print(f"Dispatching {N} annotations to {email}")
            
            #!TODO We could add more sampling strategies, e.g. uncertainty sampling, based on prob. tokens. 
            # Sample N unique annotations for this annotator (random sampling)
            import numpy as np
            next_corpus_id = np.random.choice(annots_to_dispatch.corpusid_unique, N, replace=False)
            next_sample_df = annots_to_dispatch[annots_to_dispatch.corpusid_unique.isin(next_corpus_id)]
            
            data2dump = []
            predictions_count = 0
            
            for i, row in next_sample_df.iterrows():
                # Create annotation task with optional pre-annotation
                data_dict, prediction_generated = ls_client.create_annotation_task(
                    row=row, 
                    annot_id=annot_id, 
                    email=email, 
                    proj_id=proj_id, 
                    preannotate=preannotate
                )
                
                if prediction_generated:
                    predictions_count += 1
                
                data2dump.append(data_dict)
            
            # Post to Label Studio
            try:
                ls_client.post_LS(proj_id, data2dump)
                dispatch_results[email] = {
                    "annotations_sent": len(data2dump),
                    "predictions_generated": predictions_count
                }
                total_dispatched += len(data2dump)
                print(f"Successfully dispatched {len(data2dump)} annotations to {email}")
            except Exception as e:
                print(f"Failed to dispatch annotations to {email}: {e}")
                dispatch_results[email] = {"error": str(e)}
            
            # Remove dispatched annotations from pool
            annots_to_dispatch = annots_to_dispatch[~annots_to_dispatch.corpusid_unique.isin(next_corpus_id)]
        
        return dg.MaterializeResult(
            metadata={
                "project_id": proj_id,
                "total_annotations_dispatched": total_dispatched,
                "annotators_count": len(ls_client.active_annotators),
                "annotations_per_annotator": N,
                "preannotation_method": preannotate,
                "dispatch_results": dispatch_results,
                "remaining_annotations": len(annots_to_dispatch)
            }
        )
        
    except Exception as e:
        print(f"Annotation dispatch failed: {e}")
        return dg.MaterializeResult(
            metadata={
                "project_id": proj_id,
                "error": str(e),
                "dispatch_failed": True
            }
        )
