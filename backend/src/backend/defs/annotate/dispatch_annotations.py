"""
Annotation dispatch asset for Label Studio.

Dispatches new annotation tasks from S2ORC dark data to Label Studio annotators,
with optional pre-annotation using Llama3.
"""

import dagster as dg
from backend.clients.label_studio import LabelStudioClient


@dg.asset(
    kinds={"label_studio"}, 
    key=["target", "main", "dispatch_annotations"],
    deps=["s2orc_dark_data"]
)
def dispatch_annotations(context: dg.AssetExecutionContext) -> dg.MaterializeResult:
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
    ls_client = LabelStudioClient()
    ls_client.LS_TOK = '8ecf272719351405d5c1dee84c97a9c304a9e96e'
    
    context.log.info(f"Dispatching annotations for project {proj_id}")
    context.log.info(f"Active annotators: {list(ls_client.active_annotators.keys())}")
    
    try:
        # Get new annotations to dispatch
        context.log.info("Getting new annotations from MongoDB...")
        annots_to_dispatch = ls_client.more_annotations(proj_id=proj_id)
        
        total_dispatched = 0
        dispatch_results = {}
        
        # Dispatch to each active annotator
        for email, annot_id in ls_client.active_annotators.items():
            context.log.info(f"Dispatching {N} annotations to {email}")
            
            # Sample N unique annotations for this annotator
            import numpy as np
            next_corpus_id = np.random.choice(annots_to_dispatch.corpusid_unique, N, replace=False)
            next_sample_df = annots_to_dispatch[annots_to_dispatch.corpusid_unique.isin(next_corpus_id)]
            
            data2dump = []
            predictions_count = 0
            
            for i, row in next_sample_df.iterrows():
                data_dict = {
                    "data": {
                        'corpusid': row['corpusid'],
                        'corpusid_unique': row['corpusid_unique'],
                        'par_id': row['par_id'],
                        'wc': row['wc'],
                        'text': row['text']
                    },
                    # Add empty annotations to assign task to annotator
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
                        y_pred = ls_client.run_llama3(row['text'], proj_id=proj_id)
                        y_pred_text = y_pred[0]['generated_text'].lower()
                        
                        data_dict['predictions'] = [{
                            "model_version": "llama3-8B-few-shots",
                            "score": 0.5,
                            "result": [{
                                "id": i,
                                "from_name": 'sentiment',
                                'to_name': 'text',
                                "type": "labels",
                                'value': {'choices': [y_pred_text]}
                            }]
                        }]
                        predictions_count += 1
                    except Exception as e:
                        context.log.warning(f"Pre-annotation failed for text {i}: {e}")
                
                data2dump.append(data_dict)
            
            # Post to Label Studio
            try:
                ls_client.post_LS(proj_id, data2dump)
                dispatch_results[email] = {
                    "annotations_sent": len(data2dump),
                    "predictions_generated": predictions_count
                }
                total_dispatched += len(data2dump)
                context.log.info(f"Successfully dispatched {len(data2dump)} annotations to {email}")
            except Exception as e:
                context.log.error(f"Failed to dispatch annotations to {email}: {e}")
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
        context.log.error(f"Annotation dispatch failed: {e}")
        return dg.MaterializeResult(
            metadata={
                "project_id": proj_id,
                "error": str(e),
                "dispatch_failed": True
            }
        )
