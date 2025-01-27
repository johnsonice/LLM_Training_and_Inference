## batch inference with vllm
## before run this script, you need to start the vllm server
## CUDA_VISIBLE_DEVICES=6,7 python -m vllm.entrypoints.openai.api_server --model /home/xiong/data/hf_cache/llama-3.1-8B-Instruct --dtype auto --served-model-name llama-3.1-8b-Instruct --tensor-parallel-size 2 --api-key abc
#%%
import os
import sys
import asyncio
import pandas as pd
from typing import Literal
from pydantic import BaseModel
import copy
from pathlib import Path
from tqdm.asyncio import tqdm_asyncio
import tqdm
import nest_asyncio
import asyncio
nest_asyncio.apply() 
sys.path.insert(0, '../../libs')
from llm_utils_async import AsyncBSAgent
from prompts import long_fewshotcot_pt_2label

#%%
# Define the response model
class ClimateClassification(BaseModel):
    justification: str
    classification: Literal["favorable", "unfavorable", "neutral"]

class ClimateClassification_2label(BaseModel):
    justification: str
    classification: Literal["favorable", "unfavorable"]

async def process_row(agent, row, prompt_template, file_name):
    """Process a single row of data"""
    structured_prompt = copy.deepcopy(prompt_template)
    structured_prompt['user'] = structured_prompt['user'].format(
        PARAGRAPH=row.body
    )
    try:
        response = await agent.get_response_content(
            prompt_template=structured_prompt, 
            response_format=ClimateClassification_2label
        )
        return {
            'paragraph': row.body,
            'predicted_label': response.classification,
            'justification': response.justification
        }
    except Exception as e:
        print(f"Error processing row in {file_name}: {str(e)}")
        return {
            'paragraph': row.body,
            'predicted_label': None,
            'justification': f"Error: {str(e)}"
        }

async def process_file(agent, input_file: Path, output_file: Path, prompt_template, batch_size=100):
    """Process a single CSV file in batches and save results"""
    print(f"Processing {input_file}")
    
    # Read input data
    dataset = pd.read_csv(input_file)
    total_rows = len(dataset)
    all_results = []
    
    # Process in batches with progress bar
    with tqdm.tqdm(total=total_rows, desc=f"Processing {input_file.name}") as pbar:
        for batch_start in range(0, total_rows, batch_size):
            batch_end = min(batch_start + batch_size, total_rows)
            
            # Process current batch
            batch_tasks = [
                process_row(agent, dataset.iloc[i], prompt_template, input_file.name) 
                for i in range(batch_start, batch_end)
            ]
            batch_results = await asyncio.gather(*batch_tasks)
            
            # Update progress and collect results
            all_results.extend(batch_results)
            pbar.update(len(batch_results))
            
            # Add wait period between batches to avoid overwhelming the server
            await asyncio.sleep(1)
    
    # Convert results to DataFrame and merge with original dataset
    results_df = pd.DataFrame(all_results)
    dataset['org_body'] = results_df['paragraph']
    dataset['llm_predicted_label'] = results_df['predicted_label'] 
    dataset['llm_justification'] = results_df['justification']
    dataset.to_csv(output_file, index=False)

#%%

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Process climate news articles for classification')
    parser.add_argument('--batch-size', type=int, default=128, help='Number of rows to process in parallel (default: 128)')
    parser.add_argument('--test-run', action='store_true', help='Run on a small subset of files for testing')
    parser.add_argument('--port', type=int, default=8800, help='Port for the vLLM server (default: 8800)')
    parser.add_argument('--start-idx', type=int, help='Starting index of files to process (inclusive)')
    parser.add_argument('--end-idx', type=int, help='Ending index of files to process (exclusive)')
    args = parser.parse_args()
    
    agent = AsyncBSAgent(
        model='llama-3.1-8b-Instruct',
        base_url=f'http://localhost:{args.port}/v1',
        api_key='abc'
    )

    # Define input and output directories
    data_dir = Path('/ephemeral/home/xiong/data/Fund/Climate')
    input_dir = data_dir / 'infer_res_2label'
    output_dir = data_dir / 'infer_res_2label_llama'
    output_dir.mkdir(exist_ok=True)
    
    # Get all CSV files to process and filter out already processed ones
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
        
        # Ensure indices are within valid range
        start_idx = max(0, min(start_idx, total_files))
        end_idx = max(0, min(end_idx, total_files))
        
        if start_idx >= end_idx:
            print("Warning: start_idx is greater than or equal to end_idx. No files will be processed.")
            input_files = []
        else:
            input_files = input_files[start_idx:end_idx]
            print(f"Processing files from index {start_idx} to {end_idx} (total files: {total_files})")
    else:
        print(f"Processing all files (total: {len(input_files)})")
    
    print(f"Found {len(input_files)} files to process")

    async def process_all_files():
        for input_file in tqdm.tqdm(input_files, desc="Processing files"):
            output_file = output_dir / f"results_{input_file.name}"
            try:
                await process_file(
                    agent, 
                    input_file, 
                    output_file, 
                    long_fewshotcot_pt_2label,
                    batch_size=args.batch_size
                )
            except Exception as e:
                print(f"Error processing file {input_file.name}: {str(e)}")
                continue

    ## async process all files 
    asyncio.run(process_all_files())

# %%
