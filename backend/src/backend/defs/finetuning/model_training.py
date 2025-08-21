"""
LLM fine-tuning asset using SFTTrainer.

Loads prepared training dataset from Hugging Face Hub and fine-tunes a language model 
for data availability statement detection.
"""

import dagster as dg
from datasets import load_dataset
from pathlib import Path
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import SFTTrainer
from peft import LoraConfig
import torch
from datetime import datetime


@dg.asset(
    kinds={"huggingface"}, 
    deps=["training_dataset"],
    group_name="fit"
)
def fine_tuned_model(context: dg.AssetExecutionContext) -> dg.MaterializeResult:
    """
    Fine-tune an LLM using SFTTrainer for data availability statement detection.
    
    Loads the prepared training dataset from Hugging Face Hub and fine-tunes a 
    language model using TRL's SFTTrainer with LoRA for efficient parameter tuning.
    """
    
    # Model configuration
    model_name = "meta-llama/Meta-Llama-3.2-3B-Instruct"  # Can be changed to qwen, gemma, etc.
    model_short_name = "llama3.2"  # For dataset naming
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    output_dir = f"../models/data-availability-{model_short_name}-{timestamp}"
    
    context.log.info(f"Starting fine-tuning with model: {model_name}")
    
    # Get dataset name from training_dataset asset metadata
    context.log.info("Loading dataset from Hugging Face Hub...")
    
    # Use today's date and model name to find the most recent dataset
    today = datetime.now().strftime("%Y%m%d")
    dataset_name = f"jstonge1/data-availability-statements-{model_short_name}-{today}"
    
    # Load dataset from Hub
    try:
        dataset = load_dataset(dataset_name)
        train_dataset = dataset['train']
        eval_dataset = dataset['test']
        
        context.log.info(f"Successfully loaded dataset from Hub: {dataset_name}")
        context.log.info(f"Train samples: {len(train_dataset)}, Test samples: {len(eval_dataset)}")
        
        # Get dataset info
        dataset_metadata = {
            "dataset_name": dataset_name,
            "train_samples": len(train_dataset),
            "test_samples": len(eval_dataset),
            "source": "huggingface_hub"
        }
        
    except Exception as e:
        context.log.error(f"Failed to load dataset from Hub: {e}")
        context.log.info("Attempting to load from local backup...")
        
        # Fallback to local loading
        data_dir = Path("../data/training")
        if data_dir.exists():
            dataset_dirs = [d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith("data-availability-")]
            if dataset_dirs:
                latest_dataset_dir = max(dataset_dirs, key=lambda x: x.name)
                try:
                    from datasets import load_from_disk
                    dataset_local = load_from_disk(str(latest_dataset_dir))
                    train_dataset = dataset_local['train']
                    eval_dataset = dataset_local['test']
                    
                    context.log.info(f"Loaded from local backup: {latest_dataset_dir}")
                    dataset_metadata = {
                        "dataset_path": str(latest_dataset_dir),
                        "train_samples": len(train_dataset),
                        "test_samples": len(eval_dataset),
                        "source": "local_backup"
                    }
                except Exception as local_e:
                    context.log.error(f"Local fallback also failed: {local_e}")
                    return dg.MaterializeResult(
                        metadata={
                            "error": f"Both Hub and local loading failed. Hub: {e}, Local: {local_e}",
                            "attempted_dataset": dataset_name
                        }
                    )
            else:
                return dg.MaterializeResult(
                    metadata={
                        "error": f"No datasets found in Hub or locally. Hub error: {e}",
                        "attempted_dataset": dataset_name
                    }
                )
        else:
            return dg.MaterializeResult(
                metadata={
                    "error": f"Dataset loading failed from Hub and no local backup found: {e}",
                    "attempted_dataset": dataset_name
                }
            )
    
    # Load model and tokenizer
    context.log.info("Loading model and tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        
        # Add padding token if not present
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        context.log.info("Successfully loaded model and tokenizer")
        
    except Exception as e:
        context.log.error(f"Failed to load model: {e}")
        return dg.MaterializeResult(
            metadata={
                "error": f"Model loading failed: {e}",
                "model_name": model_name,
                "dataset_metadata": dataset_metadata
            }
        )
    
    # Configure LoRA for efficient fine-tuning
    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=2,
        optim="paged_adamw_32bit",
        save_steps=100,
        logging_steps=25,
        learning_rate=2e-4,
        weight_decay=0.001,
        fp16=torch.cuda.is_available(),
        bf16=False,
        max_grad_norm=0.3,
        max_steps=-1,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="constant",
        report_to="none",  # Disable wandb/tensorboard for now
        evaluation_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        load_best_model_at_end=True,
    )
    
    # Initialize SFTTrainer
    context.log.info("Initializing SFTTrainer...")
    try:
        trainer = SFTTrainer(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            peft_config=peft_config,
            dataset_text_field="text",
            max_seq_length=512,
            tokenizer=tokenizer,
            args=training_args,
        )
        
        context.log.info("Successfully initialized SFTTrainer")
        
    except Exception as e:
        context.log.error(f"Failed to initialize trainer: {e}")
        return dg.MaterializeResult(
            metadata={
                "error": f"Trainer initialization failed: {e}",
                "model_name": model_name,
                "dataset_metadata": dataset_metadata
            }
        )
    
    # Start training
    context.log.info("Starting model fine-tuning...")
    training_success = False
    training_metrics = {}
    
    try:
        # Train the model
        trainer.train()
        
        # Save the model
        trainer.save_model()
        tokenizer.save_pretrained(output_dir)
        
        # Get training metrics
        training_metrics = trainer.state.log_history[-1] if trainer.state.log_history else {}
        
        context.log.info(f"Training completed successfully. Model saved to {output_dir}")
        training_success = True
        
    except Exception as e:
        context.log.error(f"Training failed: {e}")
        training_success = False
        training_metrics = {"error": str(e)}
    
    # Optionally push to Hugging Face Hub
    hub_push_success = False
    model_hub_name = None
    try:
        if training_success:
            model_hub_name = f"jstonge1/data-availability-detector-{model_short_name}-{timestamp}"
            trainer.push_to_hub(model_hub_name)
            context.log.info(f"Model pushed to Hugging Face Hub: {model_hub_name}")
            hub_push_success = True
    except Exception as e:
        context.log.warning(f"Failed to push model to Hugging Face Hub: {e}")
    
    return dg.MaterializeResult(
        metadata={
            "training_date": datetime.now().isoformat(),
            "model_name": model_name,
            "base_model": model_name,
            "output_directory": output_dir,
            "dataset_used": dataset_name if dataset_metadata.get("source") == "huggingface_hub" else dataset_metadata.get("dataset_path"),
            "dataset_metadata": dataset_metadata,
            "training_success": training_success,
            "hub_push_success": hub_push_success,
            "training_metrics": training_metrics,
            "model_path": output_dir,
            "model_hub_name": model_hub_name,
            "model_hub_url": f"https://huggingface.co/{model_hub_name}" if hub_push_success else None,
            "cuda_available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "lora_config": {
                "lora_alpha": 16,
                "lora_dropout": 0.1,
                "r": 64,
                "task_type": "CAUSAL_LM"
            },
            "training_args": {
                "num_train_epochs": 3,
                "per_device_train_batch_size": 4,
                "learning_rate": 2e-4,
                "max_seq_length": 512
            }
        }
    )