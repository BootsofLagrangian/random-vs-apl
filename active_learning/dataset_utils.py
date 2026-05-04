from datasets import load_dataset, Dataset
import logging
import os

logger = logging.getLogger(__name__)

def load_datasets(dataset_name, dataset_config, split = "train"):
    cache_dir = os.environ["HF_DATASETS_CACHE"]
    if os.environ["HF_HUB_OFFLINE"] == '1':
        local_path = os.path.join(cache_dir, f'datasets--{dataset_name.replace("/", "--")}')
        print(f'Loading dataset {dataset_name} from local path = {local_path}')

        dataset = load_dataset(path=local_path, split=split, cache_dir=cache_dir, trust_remote_code=True)
    else:
        dataset = load_dataset(path=dataset_name, name=dataset_config, split=split, cache_dir=cache_dir, trust_remote_code=True)
    
    return dataset

def prepare_dataset_for_method(dataset_name: str, dataset_config, alignment_method: str, tokenizer=None, split="train", max_samples=None):
    """Load and format datasets based on alignment method - accepts full HF dataset names."""
    
    try:
        # Load dataset directly using the full HF name
        print(f"Using custom dataset preprocessing for {dataset_name} with alignment method {alignment_method}")
        print(f"Loading dataset: {dataset_name}")
        
        dataset = load_datasets(dataset_name, dataset_config, split=split)
        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
        
        # Print one example BEFORE preprocessing
        # print("\n" + "="*50)
        # print("EXAMPLE BEFORE PREPROCESSING:")
        # print("="*50)
        # print(f"Dataset columns: {dataset.column_names}")
        # if len(dataset) > 0:
        #     example = dataset[0]
        #     for key, value in example.items():
        #         print(f"{key}: {str(value)[:200]}{'...' if len(str(value)) > 200 else ''}")
        # print("="*50 + "\n")
        
        # Format based on alignment method
        if alignment_method.lower() in ['online_dpo', 'xpo', 'sea']:
            # Convert to prompt-only format
            print(f"Converting to prompt-only format for {alignment_method}")
            processed_dataset = convert_to_prompt_only(dataset, tokenizer)
        
        elif alignment_method.lower() == 'dpo':
            # Keep preference format, just ensure proper columns
            print(f"Keeping preference format for {alignment_method}")
            processed_dataset = ensure_preference_format(dataset)
        
        else:
            logger.warning(f"Unknown alignment method: {alignment_method}, returning dataset as-is")
            processed_dataset = dataset
        
        # Print one example AFTER preprocessing
        # print("\n" + "="*50)
        # print("EXAMPLE AFTER PREPROCESSING:")
        # print("="*50)
        # print(f"Dataset columns: {processed_dataset.column_names}")
        # if len(processed_dataset) > 0:
        #     example = processed_dataset[0]
        #     for key, value in example.items():
        #         print(f"{key}: {str(value)[:200]}{'...' if len(str(value)) > 200 else ''}")
        # print("="*50 + "\n")
        
        return processed_dataset
            
    except Exception as e:
        print(f"Error in dataset preprocessing: {e}")
        # Fallback to loading normally
        dataset = load_dataset(dataset_name, dataset_config, split=split)
        return dataset

def convert_to_prompt_only(dataset, tokenizer=None):
    """Convert preference dataset to prompt-only format."""
    
    def extract_prompt(example):
        # Handle different dataset structures
        if 'prompt' in example:
            return {'prompt': example['prompt']}
        elif 'context' in example:
            return {'prompt': example['context']}
        elif 'summaries' in example and 'choice' in example and 'info' in example:
            # TL;DR preference dataset - construct prompt from info field
            if isinstance(example['info'], dict):
                subreddit = example['info'].get('subreddit', 'Unknown')
                title = example['info'].get('title', '')
                post = example['info'].get('post', '')
                prompt = f"SUBREDDIT: r/{subreddit}\nTITLE: {title}\nPOST: {post}\nTL;DR:"
                return {'prompt': prompt}
            else:
                raise ValueError(f"TL;DR dataset should have 'info' field with subreddit, title, and post. Available fields: {list(example.keys())}")
        elif 'chosen' in example and 'rejected' in example:
            # Handle ultrafeedback format (list of messages) or standard string format
            if isinstance(example.get('chosen'), list):
                # Ultrafeedback format: extract from the first message (user message)
                if len(example['chosen']) > 0 and isinstance(example['chosen'][0], dict) and 'content' in example['chosen'][0]:
                    return {'prompt': example['chosen'][0]['content']}
                else:
                    raise ValueError(f"Expected 'chosen' field to contain message format but got: {example['chosen']}")
            elif isinstance(example.get('chosen'), str):
                # Simple extraction - take everything before first Assistant response
                text = example['chosen']
                if "Human:" in text and "Assistant:" in text:
                    # keep the Assistant: tag so the model can continue right after it
                    prompt_core = text.split("Assistant:")[0]
                    prompt_core = prompt_core.replace("Human:", "").rstrip()
                    prompt = f"{prompt_core}\nAssistant:"          # <- added tag back
                    return {"prompt": prompt}
                else:
                    raise ValueError(f"Unable to extract prompt from chosen text. Expected 'Human:' and 'Assistant:' markers but found: {text[:100]}...")
            else:
                raise ValueError(f"Expected 'chosen' field to be a string or list, but got: {type(example.get('chosen'))}")
        else:
            # Check for any suitable text field
            for key in example.keys():
                if isinstance(example[key], str) and len(example[key]) > 10:
                    raise ValueError(f"Cannot extract prompt from dataset. Found text field '{key}' but no standard prompt format. Dataset structure: {list(example.keys())}")
            raise ValueError(f"No suitable text field found in dataset example. Available fields: {list(example.keys())}")
    
    return dataset.map(extract_prompt, remove_columns=dataset.column_names)

def ensure_preference_format(dataset):
    """Ensure dataset has proper preference format columns."""
    
    required_columns = ['prompt', 'chosen', 'rejected']
    
    # Check if dataset already has the right format
    if all(col in dataset.column_names for col in required_columns):
        return dataset
    
    # Try to create the format from existing columns
    def format_preference(example):
        result = {}
        
        # Handle different dataset formats for chosen/rejected first
        if 'summaries' in example and 'choice' in example:
            # TL;DR preference dataset - construct prompt from info field
            if 'info' in example and isinstance(example['info'], dict):
                # Construct prompt from info similar to CarperAI/openai_summarize_tldr format
                subreddit = example['info'].get('subreddit', 'Unknown')
                title = example['info'].get('title', '')
                post = example['info'].get('post', '')
                result['prompt'] = f"SUBREDDIT: r/{subreddit}\nTITLE: {title}\nPOST: {post}\nTL;DR:"
            else:
                raise ValueError(f"TL;DR dataset should have 'info' field with subreddit, title, and post. Available fields: {list(example.keys())}")
            
            choice_idx = int(example['choice'])
            # Extract just the text from each summary dict
            if isinstance(example['summaries'][choice_idx], dict):
                result['chosen'] = example['summaries'][choice_idx]['text']
            else:
                result['chosen'] = example['summaries'][choice_idx]
                
            if isinstance(example['summaries'][1 - choice_idx], dict):
                result['rejected'] = example['summaries'][1 - choice_idx]['text']
            else:
                result['rejected'] = example['summaries'][1 - choice_idx]
            
        elif 'chosen' in example and 'rejected' in example:
            # Extract prompt
            if 'prompt' in example:
                result['prompt'] = example['prompt']
            elif 'context' in example:
                result['prompt'] = example['context']
            elif isinstance(example['chosen'], list) and len(example['chosen']) > 0:
                # For ultrafeedback format: extract from the first message (user message)
                if isinstance(example['chosen'][0], dict) and 'content' in example['chosen'][0]:
                    result['prompt'] = example['chosen'][0]['content']
                else:
                    raise ValueError(f"Expected 'chosen' field to contain message format but got: {type(example['chosen'][0])}")
            else:
                # Try to extract from chosen field with Human:/Assistant: format
                if "Human:" in str(example["chosen"]) and "Assistant:" in str(example["chosen"]):
                    prompt_core = str(example["chosen"]).split("Assistant:")[0]
                    prompt_core = prompt_core.replace("Human:", "").rstrip()
                    result["prompt"] = f"{prompt_core}\nAssistant:"
                else:
                    raise ValueError(f"Cannot extract prompt from dataset. No 'prompt' or 'context' field found, and unable to extract from 'chosen' field. Available fields: {list(example.keys())}")
                    
            # Check if this is ultrafeedback format (list of messages)
            if isinstance(example['chosen'], list) and isinstance(example['rejected'], list):
                # For ultrafeedback: extract the assistant response (second message)
                if len(example['chosen']) > 1 and isinstance(example['chosen'][1], dict) and 'content' in example['chosen'][1]:
                    result['chosen'] = example['chosen'][1]['content']
                else:
                    raise ValueError(f"Expected 'chosen' field to have assistant message at index 1 but got: {example['chosen']}")
                
                if len(example['rejected']) > 1 and isinstance(example['rejected'][1], dict) and 'content' in example['rejected'][1]:
                    result['rejected'] = example['rejected'][1]['content']
                else:
                    raise ValueError(f"Expected 'rejected' field to have assistant message at index 1 but got: {example['rejected']}")
            else:
                # Standard preference format (strings)
                result['chosen'] = example['chosen']
                result['rejected'] = example['rejected']
        else:
            # Standard format - extract prompt normally
            if 'prompt' in example:
                result['prompt'] = example['prompt']
            elif 'context' in example:
                result['prompt'] = example['context']
            else:
                raise ValueError(f"Cannot extract prompt from dataset. No 'prompt' or 'context' field found. Available fields: {list(example.keys())}")
                
            # Ensure chosen/rejected fields exist
            if 'chosen' not in example:
                raise ValueError(f"Required 'chosen' field not found in dataset. Available fields: {list(example.keys())}")
            if 'rejected' not in example:
                raise ValueError(f"Required 'rejected' field not found in dataset. Available fields: {list(example.keys())}")
                
            result['chosen'] = example['chosen']
            result['rejected'] = example['rejected']
        
        return result
    
    return dataset.map(format_preference, remove_columns=dataset.column_names)

def detect_alignment_method(trainer):
    """Auto-detect alignment method from trainer type."""
    trainer_name = trainer.__class__.__name__
    
    TRAINER_METHOD_MAP = {
        'DPOTrainer': 'DPO',
        'OnlineDPOTrainer': 'OnlineDPO', 
        'XPOTrainer': 'XPO',
        'CPOTrainer': 'CPO',
        'ORPOTrainer': 'ORPO',
        'GRPOTrainer': 'GRPO',
        'NashMDTrainer': 'NashMD',
    }
    
    method = TRAINER_METHOD_MAP.get(trainer_name, 'DPO')
    logger.info(f"Auto-detected alignment method: {method} from trainer: {trainer_name}")
    return method

def get_supported_datasets():
    """Return list of supported datasets with full HuggingFace names."""
    return [
        "Anthropic/hh-rlhf",
        "yuasosnin/imdb-dpo", 
        "UCL-DARK/openai-tldr-summarisation-preferences",
        "trl-lib/ultrafeedback_binarized"
    ]
