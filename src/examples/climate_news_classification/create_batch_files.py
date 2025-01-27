#%%
import os
import sys
import json
import pandas as pd
from typing import Literal
from pydantic import BaseModel
from pathlib import Path
from tqdm import tqdm
from prompts import long_fewshotcot_pt_2label
from openai.lib._parsing._completions import type_to_response_format_param

#%%

class ClimateClassification_2label(BaseModel):
    justification: str
    classification: Literal["favorable", "unfavorable"]

def create_batch_tasks(input_file: Path, output_format: BaseModel):
    """Create batch tasks from input file"""
    dataset = pd.read_csv(input_file)
    
    output_json_schema = type_to_response_format_param(ClimateClassification_2label)
    
    tasks = []
    for idx, row in enumerate(dataset.itertuples()):
        structured_prompt = long_fewshotcot_pt_2label.copy()
        structured_prompt['user'] = structured_prompt['user'].format(
            PARAGRAPH=row.body
        )
        
        task = {
            "custom_id": f"task-{input_file.stem}-{idx}",
            "method": "POST",
            "url": "/chat/completions",
            "body": {
                "model": "llama-3.1-8b-Instruct",
                "messages": [
                    {"role": "system", "content": structured_prompt["system"]},
                    {"role": "user", "content": structured_prompt["user"]}
                ],
                "response_format": {"type": "json_object"}, #output_json_schema, output format affects output speed
                "temperature": 0.1
            }
        }
        tasks.append(task)
    
    return tasks

#%%
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Create batch task files for climate news classification')
    parser.add_argument('--start-idx', type=int, help='Starting index of files to process (inclusive)')
    parser.add_argument('--end-idx', type=int, help='Ending index of files to process (exclusive)')
    parser.add_argument('--test-run', action='store_true', help='Run on a small subset of files for testing')
    args = parser.parse_args()
    #%%
    # Define directories
    data_dir = Path('/ephemeral/home/xiong/data/Fund/Climate')
    input_dir = data_dir / 'infer_res_2label'
    batch_dir = data_dir / 'batch_tasks'
    
    # Create batch directory if it doesn't exist
    batch_dir.mkdir(exist_ok=True)
    
    # Get input files
    if args.test_run:
        input_files = list(input_dir.glob('*.csv'))[:2]
    else:
        input_files = list(input_dir.glob('*.csv'))
    input_files = [f for f in input_files if not (f.name.startswith('results_') or f.name.startswith('.'))]
    
    # Apply file range filtering if specified
    if args.start_idx is not None or args.end_idx is not None:
        total_files = len(input_files)
        start_idx = args.start_idx if args.start_idx is not None else 0
        end_idx = args.end_idx if args.end_idx is not None else total_files
        input_files = input_files[start_idx:end_idx]
    
    print(f"Creating batch files for {len(input_files)} input files")
    
    #%%
    # Create batch files
    for input_file in tqdm(input_files, desc="Creating batch files"):
        batch_file_path = batch_dir / f"batch_tasks_{input_file.stem}.jsonl"
        
        # Skip if batch file already exists
        if batch_file_path.exists():
            print(f"Skipping {input_file.name} - batch file already exists")
            continue
            
        print(f"\nProcessing {input_file}")
        
        try:
            # Create batch tasks
            tasks = create_batch_tasks(input_file, ClimateClassification_2label)
            
            # Write tasks to JSONL file
            with open(batch_file_path, 'w') as f:
                for task in tasks:
                    f.write(json.dumps(task) + '\n')
                    
        except Exception as e:
            print(f"Error processing file {input_file.name}: {str(e)}")
            continue 
# %%
