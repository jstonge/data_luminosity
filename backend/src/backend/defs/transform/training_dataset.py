"""
Training dataset preparation asset.

Prepares and formats annotation data for LLM fine-tuning on data availability detection.
Creates and shares datasets on Hugging Face Hub with proper chat templates for different models.
"""

import pandas as pd
import dagster as dg
from datasets import Dataset, DatasetDict
from pathlib import Path
import duckdb
import filelock
from datetime import datetime
import json


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


def get_chat_template(model_name="llama3.2"):
    """Get chat template for different models"""
    templates = {
        "llama3.2": {
            "system": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|>",
            "user": "<|start_header_id|>user<|end_header_id|>\n\n{user}<|eot_id|>",
            "assistant": "<|start_header_id|>assistant<|end_header_id|>\n\n{assistant}<|eot_id|>",
            "format": "{system}{user}{assistant}"
        },
        "qwen": {
            "system": "<|im_start|>system\n{system}<|im_end|>\n",
            "user": "<|im_start|>user\n{user}<|im_end|>\n",
            "assistant": "<|im_start|>assistant\n{assistant}<|im_end|>",
            "format": "{system}{user}{assistant}"
        },
        "gemma": {
            "system": "",  # Gemma doesn't use system messages
            "user": "<start_of_turn>user\n{user}<end_of_turn>\n",
            "assistant": "<start_of_turn>model\n{assistant}<end_of_turn>",
            "format": "{user}{assistant}"
        },
        "gpt-oss": {
            "system": "<|start|>system<|message|>{system}<|end|>",
            "user": "<|start|>user<|message|>{user}<|end|>", 
            "assistant": "<|start|>assistant<|channel|>final<|message|>{assistant}<|end|>",
            "format": "{system}{user}{assistant}"
        },
        "generic": {  # Fallback for other models
            "system": "<|system|>\n{system}<|end|>\n",
            "user": "<|user|>\n{user}<|end|>\n",
            "assistant": "<|assistant|>\n{assistant}<|end|>",
            "format": "{system}{user}{assistant}"
        }
    }
    return templates.get(model_name, templates["generic"])


@dg.asset(
    kinds={"python"}, 
    deps=["deduplicated_annotations"],
    group_name="transform"
)
def training_dataset(context: dg.AssetExecutionContext) -> dg.MaterializeResult:
    """
    Prepare training dataset from deduplicated annotations and share on Hugging Face Hub.
    
    Loads deduplicated annotation data and formats it into train/test splits using proper
    chat templates, then pushes the dataset to Hugging Face Hub for easy sharing.
    """
    
    # Configuration
    target_model = "llama3.2"  # Can be changed to "qwen", "gemma", "gpt-oss", etc.
    
    context.log.info(f"Formatting data for model: {target_model}")
    
    context.log.info("Loading deduplicated annotations from DuckDB...")
    
    # Get deduplicated annotations
    query = """
        SELECT 
            text,
            has_data_statement,
            statement_category,
            source,
            sentiment
        FROM main.deduplicated_annotations 
        WHERE text IS NOT NULL 
        AND has_data_statement IS NOT NULL
    """
    
    annotations_data = serialize_duckdb_query("/tmp/data_luminosity.duckdb", query)
    
    if not annotations_data:
        context.log.warning("No deduplicated annotation data found")
        return dg.MaterializeResult(
            metadata={
                "error": "No deduplicated annotation data found",
                "total_samples": 0
            }
        )
    
    # Convert to DataFrame
    df = pd.DataFrame(annotations_data, columns=['text', 'has_data_statement', 'statement_category', 'source', 'sentiment'])
    
    context.log.info(f"Loaded {len(df)} deduplicated annotations")
    
    # Encode sentiment for ML
    df['sentiment_encoded'] = df['sentiment'].map({'yes': 1, 'no': 0})
    
    context.log.info("Creating train/test split and formatting for SFT...")
    
    # Create Hugging Face dataset and split
    dataset_df = df[['text', 'sentiment_encoded', 'sentiment', 'source']].copy()
    ds = Dataset.from_pandas(dataset_df, preserve_index=False).train_test_split(test_size=0.2, seed=42)
    
    # Format data for SFTTrainer using proper chat templates
    def format_for_sft(examples, model_name="llama3.2"):
        formatted_texts = []
        template = get_chat_template(model_name)
        
        system_message = "You are an expert at identifying data availability statements in scientific text. A data availability statement describes where research data can be found or how it can be accessed."
        instruction = "Determine if the following text contains a data availability statement. Answer with 'yes' or 'no'."
        
        for i in range(len(examples['text'])):
            answer = "yes" if examples["sentiment_encoded"][i] == 1 else "no"
            
            user_message = f"{instruction}\n\nText: {examples['text'][i]}"
            
            # Build the formatted conversation
            if template["system"] and model_name not in ["gemma"]:  # Some models don't use system messages
                system_part = template["system"].format(system=system_message)
            else:
                system_part = ""
                
            user_part = template["user"].format(user=user_message)
            assistant_part = template["assistant"].format(assistant=answer)
            
            # Combine according to format
            if model_name == "gemma":
                formatted_text = template["format"].format(
                    user=user_message,
                    assistant=answer
                )
            else:
                formatted_text = template["format"].format(
                    system=system_part,
                    user=user_part, 
                    assistant=assistant_part
                )
            
            formatted_texts.append(formatted_text)
        
        return {"text": formatted_texts}
    
    # Apply formatting
    train_dataset = ds['train'].map(
        lambda x: format_for_sft(x, target_model), 
        batched=True, 
        remove_columns=ds['train'].column_names
    )
    eval_dataset = ds['test'].map(
        lambda x: format_for_sft(x, target_model),
        batched=True, 
        remove_columns=ds['test'].column_names
    )
    
    # Calculate split statistics
    train_size = len(ds['train'])
    test_size = len(ds['test'])
    train_pos = sum(ds['train']['sentiment_encoded'])
    train_neg = train_size - train_pos
    test_pos = sum(ds['test']['sentiment_encoded'])
    test_neg = test_size - test_pos
    
    context.log.info(f"Train set: {train_size} samples ({train_pos} positive, {train_neg} negative)")
    context.log.info(f"Test set: {test_size} samples ({test_pos} positive, {test_neg} negative)")
    
    # Create DatasetDict for Hugging Face Hub
    today = datetime.now().strftime("%Y%m%d")
    dataset_name = f"jstonge1/data-availability-statements-{target_model}-{today}"
    
    dataset_dict = DatasetDict({
        "train": train_dataset,
        "test": eval_dataset
    })
    
    # Get example formatted text for documentation
    example_template = get_chat_template(target_model)
    if target_model == "gemma":
        example_format = example_template["format"].format(
            user="Determine if the following text contains a data availability statement. Answer with 'yes' or 'no'.\n\nText: [SAMPLE_TEXT]",
            assistant="[yes|no]"
        )
    else:
        system_part = example_template["system"].format(system="You are an expert at identifying data availability statements in scientific text...")
        user_part = example_template["user"].format(user="Determine if the following text contains a data availability statement. Answer with 'yes' or 'no'.\n\nText: [SAMPLE_TEXT]")
        assistant_part = example_template["assistant"].format(assistant="[yes|no]")
        example_format = example_template["format"].format(
            system=system_part,
            user=user_part,
            assistant=assistant_part
        )
    
    # Add dataset card
    dataset_card = f"""# Data Availability Statements Dataset

This dataset contains deduplicated and annotated text for training models to detect data availability statements in scientific literature.

## Dataset Information
- **Total samples**: {len(df):,}
- **Train samples**: {train_size:,}
- **Test samples**: {test_size:,}
- **Train positive ratio**: {round(train_pos / train_size, 3) if train_size > 0 else 0}
- **Test positive ratio**: {round(test_pos / test_size, 3) if test_size > 0 else 0}
- **Created**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Target model**: {target_model.upper()}

## Data Sources
{chr(10).join([f'- **{source}**: {count:,} samples' for source, count in df['source'].value_counts().to_dict().items()])}

## Data Quality
- **Exact duplicates removed**: Eliminated identical texts
- **Semantic duplicates removed**: Used embeddings to remove similar texts
- **Clean binary labels**: Only 'yes' and 'no' responses included

## Format
Each sample is formatted for supervised fine-tuning using **{target_model.upper()}** chat template with the instruction:
"Determine if the following text contains a data availability statement. Answer with 'yes' or 'no'."

The format follows {target_model.upper()}'s chat template:
```
{example_format}
```

## Usage
```python
from datasets import load_dataset

dataset = load_dataset("{dataset_name}")
print(dataset)

# For training with SFTTrainer
from trl import SFTTrainer
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset['train'],
    eval_dataset=dataset['test'],
    dataset_text_field="text",
    # ... other args
)
```

## Labels
- **yes**: Text contains a data availability statement
- **no**: Text does not contain a data availability statement

## Compatible Models
This dataset is formatted specifically for **{target_model}** but the preprocessing pipeline supports:
- Llama 3.2 (and other Llama models)
- Qwen series
- Gemma series
- GPT-OSS (20B and other variants)
- Generic chat format

## Citation
If you use this dataset, please cite the original data sources and this dataset.
"""
    
    # Save locally and push to Hub
    local_save_success = False
    hub_push_success = False
    
    try:
        # Save locally as backup
        output_dir = Path(f"../data/training/data-availability-{target_model}-{today}")
        output_dir.mkdir(parents=True, exist_ok=True)
        dataset_dict.save_to_disk(str(output_dir))
        
        # Save metadata
        metadata_local = {
            "dataset_date": today,
            "target_model": target_model,
            "total_samples": len(df),
            "train_samples": train_size,
            "test_samples": test_size,
            "source_distribution": df['source'].value_counts().to_dict(),
            "format": f"sft_{target_model}_chat_format",
            "dataset_name": dataset_name,
            "uses_deduplicated_data": True
        }
        
        with open(output_dir / "metadata.json", 'w') as f:
            json.dump(metadata_local, f, indent=2)
        
        context.log.info(f"Saved dataset locally to {output_dir}")
        local_save_success = True
        
    except Exception as e:
        context.log.error(f"Failed to save dataset locally: {e}")
    
    try:
        # Push to Hugging Face Hub
        context.log.info(f"Pushing dataset to Hugging Face Hub: {dataset_name}")
        
        dataset_dict.push_to_hub(dataset_name)
        
        context.log.info(f"Successfully pushed dataset to Hugging Face Hub: {dataset_name}")
        hub_push_success = True
        
    except Exception as e:
        context.log.error(f"Failed to push to Hugging Face Hub: {e}")
        dataset_name = None
    
    # Calculate source distribution
    source_distribution = df['source'].value_counts().to_dict()
    
    return dg.MaterializeResult(
        metadata={
            "dataset_date": today,
            "target_model": target_model,
            "total_samples": len(df),
            "train_samples": train_size,
            "test_samples": test_size,
            "train_positive": train_pos,
            "train_negative": train_neg,
            "test_positive": test_pos, 
            "test_negative": test_neg,
            "train_balance_ratio": round(train_pos / train_size, 3) if train_size > 0 else 0,
            "test_balance_ratio": round(test_pos / test_size, 3) if test_size > 0 else 0,
            "source_distribution": source_distribution,
            "local_save_success": local_save_success,
            "hub_push_success": hub_push_success,
            "dataset_name": dataset_name,
            "dataset_url": f"https://huggingface.co/datasets/{dataset_name}" if hub_push_success else None,
            "format": f"sft_{target_model}_chat_format",
            "uses_deduplicated_data": True,
            "output_directory": str(output_dir) if local_save_success else None
        }
    )