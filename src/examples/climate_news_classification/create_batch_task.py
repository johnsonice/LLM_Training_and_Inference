#%%
import os
import sys
import json
import pandas as pd
from typing import Literal, List
from pydantic import BaseModel
from pathlib import Path
from tqdm import tqdm
import time
from openai import OpenAI

class ClimateClassification_2label(BaseModel):
    justification: str
    classification: Literal["favorable", "unfavorable"]

def get_batch_files(batch_dir: Path, output_dir: Path, test_run: bool = False, 
                    start_idx: int = None, end_idx: int = None) -> List[Path]:
    """Get list of batch files to process"""
    # Get batch files
    batch_files = list(batch_dir.glob('batch_tasks_*.jsonl'))
    # Filter out already processed files
    existing_outputs = {f.name.replace('results_', '') for f in output_dir.glob('*.jsonl')}
    batch_files = [f for f in batch_files if f.stem + '.jsonl' not in existing_outputs]
    
    if test_run:
        batch_files = batch_files[:5]
        
    # Apply file range filtering if specified
    if start_idx is not None or end_idx is not None:
        total_files = len(batch_files)
        start_idx = start_idx if start_idx is not None else 0
        end_idx = end_idx if end_idx is not None else total_files
        batch_files = batch_files[start_idx:end_idx]
    
    return batch_files

def process_single_batch(client: OpenAI, batch_file: Path, output_file: Path) -> None:
    """Process a single batch file and save results"""
    batch_file_obj = None
    result_file_id = None
    
    try:
        # Create batch file
        batch_file_obj = client.files.create(
            file=open(batch_file, "rb"),
            purpose="batch"
        )
        print(f"Created batch file {batch_file_obj.id}")
        # Create batch job
        batch_job = client.batches.create(
            input_file_id=batch_file_obj.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )
        
        print(f"Created batch job {batch_job.id}")
        
        # Wait for completion and get results
        while True:
            batch_job = client.batches.retrieve(batch_job.id)
            if batch_job.status == "completed":
                break
            print(f"Waiting for batch job completion... Status: {batch_job.status}", end='\r', flush=True)
            time.sleep(60)  # Check every minute
        
        # Get results
        result_file_id = batch_job.output_file_id
        result = client.files.content(result_file_id).content
        
        # Simply write the raw results to output file
        with open(output_file, 'wb') as f:
            f.write(result)
        
    except Exception as e:
        print(f"Error processing batch file {batch_file.name}: {str(e)}")
    finally:
        # Clean up files from the server
        try:
            if batch_file_obj:
                client.files.delete(batch_file_obj.id)
            if result_file_id:
                client.files.delete(result_file_id)
        except Exception as e:
            print(f"Error deleting files from server: {str(e)}")

def main(args):
    client = OpenAI(
        base_url=f'http://localhost:{args.port}/v1',
        api_key='abc'
    )
    
    # Define directories
    data_dir = Path('/ephemeral/home/xiong/data/Fund/Climate')
    batch_dir = data_dir / 'batch_tasks'#_formated_output'
    output_dir = data_dir / 'batch_tasks_results_v1_no_formated_output'
    output_dir.mkdir(exist_ok=True)
    
    # Get batch files to process
    batch_files = get_batch_files(
        batch_dir, 
        output_dir, 
        test_run=args.test_run,
        start_idx=args.start_idx,
        end_idx=args.end_idx
    )
    
    print(f"Processing {len(batch_files)} batch files")
    
    # Process each batch file
    for batch_file in tqdm(batch_files, desc="Processing batch files"):
        output_file = output_dir / f"results_{batch_file.stem}.jsonl"
        print(f"\nProcessing {batch_file}")
        try:
            process_single_batch(client, batch_file, output_file)
        except Exception:
            continue
#%%
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Process batch files for climate news classification')
    parser.add_argument('--start-idx', type=int, help='Starting index of batch files to process (inclusive)')
    parser.add_argument('--end-idx', type=int, help='Ending index of batch files to process (exclusive)')
    parser.add_argument('--test-run', action='store_true', help='Run on a small subset of files for testing')
    parser.add_argument('--port', type=int, default=8800, help='Port for the vLLM server (default: 8800)')
    args = parser.parse_args()
    
    main(args)

# %%
