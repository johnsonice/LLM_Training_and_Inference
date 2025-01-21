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

async def process_file(agent, input_file: Path, output_file: Path, prompt_template, batch_size=100):
    """Process a single CSV file in batches and save results"""
    print(f"Processing {input_file}")
    
    # Read input data
    dataset = pd.read_csv(input_file)
    total_rows = len(dataset)
    all_results = []
    
    async def process_row(i):
        structured_prompt = copy.deepcopy(prompt_template)
        structured_prompt['user'] = structured_prompt['user'].format(
            PARAGRAPH=dataset.iloc[i].body
        )
        try:
            response = await agent.get_response_content(
                prompt_template=structured_prompt, 
                response_format=ClimateClassification_2label
            )
            return {
                'paragraph': dataset.iloc[i].body,
                'predicted_label': response.classification,
                'justification': response.justification
            }
        except Exception as e:
            print(f"Error processing row {i} in {input_file.name}: {str(e)}")
            return {
                'paragraph': dataset.iloc[i].body,
                'predicted_label': None,
                'justification': f"Error: {str(e)}"
            }

    # Process in batches with progress bar
    with tqdm.tqdm(total=total_rows, desc=f"Processing {input_file.name}") as pbar:
        for batch_start in range(0, total_rows, batch_size):
            batch_end = min(batch_start + batch_size, total_rows)
            
            # Process current batch
            batch_tasks = [process_row(i) for i in range(batch_start, batch_end)]
            batch_results = await asyncio.gather(*batch_tasks)
            
            # Update progress and collect results
            all_results.extend(batch_results)
            pbar.update(len(batch_results))
    
    # Convert results to DataFrame and merge with original dataset
    results_df = pd.DataFrame(all_results)
    dataset['org_body'] = results_df['paragraph']
    dataset['llm_predicted_label'] = results_df['predicted_label'] 
    dataset['llm_justification'] = results_df['justification']
    dataset.to_csv(output_file, index=False)

#%%

if __name__ == "__main__":
    import nest_asyncio
    import asyncio
    nest_asyncio.apply() 
    test_run = False
    
    agent = AsyncBSAgent(
        model='llama-3.1-8b-Instruct',
        base_url='http://localhost:8000/v1',
        api_key='abc'
    )
    # Define input and output directories
    input_dir = Path('/home/xiong/data/Fund/Climate/infer_res_2label')
    output_dir = Path('/home/xiong/data/Fund/Climate/infer_res_2label_llama')
    output_dir.mkdir(exist_ok=True)
    
    # Get all CSV files to process
    if test_run:
        input_files = list(input_dir.glob('*.csv'))[:2]
    else:
        input_files = list(input_dir.glob('*.csv'))
    input_files = [f for f in input_files if not f.name.startswith('results_')]
    print(f"Found {len(input_files)} files to process")
    
    for input_file in tqdm.tqdm(input_files, desc="Processing files"):
        output_file = output_dir / f"results_{input_file.name}"
        asyncio.run(process_file(
            agent, 
            input_file, 
            output_file, 
            long_fewshotcot_pt_2label,
            batch_size=1000  # Process 50 rows at a time
        ))

# %%
