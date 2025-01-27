import os
import sys
import json
import pandas as pd
from typing import Literal
from pydantic import BaseModel
from pathlib import Path
from tqdm import tqdm
from openai import OpenAI
from prompts import long_fewshotcot_pt_2label

class ClimateClassification_2label(BaseModel):
    justification: str
    classification: Literal["favorable", "unfavorable"]

def create_batch_tasks(input_file: Path, start_idx: int = None, end_idx: int = None):
    """Create batch tasks from input file"""
    dataset = pd.read_csv(input_file)
    
    # Apply index filtering if specified
    if start_idx is not None or end_idx is not None:
        start_idx = start_idx if start_idx is not None else 0
        end_idx = end_idx if end_idx is not None else len(dataset)
        dataset = dataset.iloc[start_idx:end_idx]
    
    tasks = []
    for idx, row in enumerate(dataset.itertuples()):
        structured_prompt = long_fewshotcot_pt_2label.copy()
        structured_prompt['user'] = structured_prompt['user'].format(
            PARAGRAPH=row.body
        )
        
        task = {
            "custom_id": f"task-{input_file.stem}-{idx}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "llama-3.1-8b-Instruct",
                "messages": [
                    {"role": "system", "content": structured_prompt["system"]},
                    {"role": "user", "content": structured_prompt["user"]}
                ],
                "response_format": {"type": "json_object"},
                "temperature": 0.1
            }
        }
        tasks.append(task)
    
    return tasks

def process_batch_results(results, output_file: Path, input_file: Path):
    """Process batch results and save to output file"""
    original_data = pd.read_csv(input_file)
    
    # Extract predictions from results
    predictions = []
    for result in results:
        try:
            content = json.loads(result['response']['body']['choices'][0]['message']['content'])
            predictions.append({
                'task_id': result['custom_id'],
                'predicted_label': content['classification'],
                'justification': content['justification']
            })
        except Exception as e:
            print(f"Error processing result {result['custom_id']}: {str(e)}")
            predictions.append({
                'task_id': result['custom_id'],
                'predicted_label': None,
                'justification': f"Error: {str(e)}"
            })
    
    # Create results DataFrame
    results_df = pd.DataFrame(predictions)
    results_df['original_idx'] = results_df['task_id'].apply(
        lambda x: int(x.split('-')[-1])
    )
    results_df = results_df.sort_values('original_idx')
    
    # Merge with original data and save
    original_data['llm_predicted_label'] = results_df['predicted_label'].values
    original_data['llm_justification'] = results_df['justification'].values
    original_data.to_csv(output_file, index=False)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Process climate news articles using OpenAI Batch API')
    parser.add_argument('--start-idx', type=int, help='Starting index of files to process (inclusive)')
    parser.add_argument('--end-idx', type=int, help='Ending index of files to process (exclusive)')
    parser.add_argument('--test-run', action='store_true', help='Run on a small subset of files for testing')
    args = parser.parse_args()
    
    client = OpenAI()
    
    # Define input and output directories
    data_dir = Path('/ephemeral/home/xiong/data/Fund/Climate')
    input_dir = data_dir / 'infer_res_2label'
    output_dir = data_dir / 'infer_res_2label_llama'
    batch_dir = data_dir / 'batch_tasks'  # New directory for batch task files
    
    # Create directories if they don't exist
    output_dir.mkdir(exist_ok=True)
    batch_dir.mkdir(exist_ok=True)
    
    # Get input files
    if args.test_run:
        input_files = list(input_dir.glob('*.csv'))[:2]
    else:
        input_files = list(input_dir.glob('*.csv'))
    input_files = [f for f in input_files if not (f.name.startswith('results_') or f.name.startswith('.'))]
    existing_outputs = {f.name.replace('results_', '') for f in output_dir.glob('*.csv')}
    input_files = [f for f in input_files if f.name not in existing_outputs]
    
    # Apply file range filtering if specified
    if args.start_idx is not None or args.end_idx is not None:
        total_files = len(input_files)
        start_idx = args.start_idx if args.start_idx is not None else 0
        end_idx = args.end_idx if args.end_idx is not None else total_files
        input_files = input_files[start_idx:end_idx]
    
    print(f"Processing {len(input_files)} files")
    
    for input_file in tqdm(input_files, desc="Processing files"):
        output_file = output_dir / f"results_{input_file.name}"
        batch_file_path = batch_dir / f"batch_tasks_{input_file.stem}.jsonl"
        print(f"\nProcessing {input_file}")
        
        # Create batch tasks
        tasks = create_batch_tasks(input_file)
        
        # Write tasks to JSONL file in batch directory
        with open(batch_file_path, 'w') as f:
            for task in tasks:
                f.write(json.dumps(task) + '\n')
        
        try:
            # Create batch file
            batch_file = client.files.create(
                file=open(batch_file_path, "rb"),
                purpose="batch"
            )
            
            # Create batch job
            batch_job = client.batches.create(
                input_file_id=batch_file.id,
                endpoint="/v1/chat/completions"
            )
            
            print(f"Created batch job {batch_job.id}")
            
            # Wait for completion and get results
            while True:
                batch_job = client.batches.retrieve(batch_job.id)
                if batch_job.status == "completed":
                    break
                print(f"Waiting for batch job completion... Status: {batch_job.status}")
                time.sleep(60)  # Check every minute
            
            # Get results
            result_file_id = batch_job.output_file_id
            result = client.files.content(result_file_id).content
            results = [json.loads(line) for line in result.decode().split('\n') if line]
            
            # Process and save results
            process_batch_results(results, output_file, input_file)
            
            # Note: We're not removing the batch file to keep a record
            # You can implement a cleanup script separately if needed
            
        except Exception as e:
            print(f"Error processing file {input_file.name}: {str(e)}")
            continue 