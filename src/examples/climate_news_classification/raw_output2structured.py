#%%
#### convert batch task results to formated output
import sys,os
sys.path.insert(0,'../../libs')
import pandas as pd
import argparse
from pathlib import Path
import json
import re
from typing import Dict, List, Optional, Union, Tuple, Any, Literal
from pydantic import BaseModel
import logging
from llm_utils import BSAgent
from langchain.output_parsers import PydanticOutputParser
from tqdm import tqdm
#%%

def parse_result(res, parser,verbose):
    try:
        return parser.parse(res).dict()
    except Exception as e:
        if verbose:
            print(f"Parser error: {e}")
        return None

def fix_trailing_comma(json_str):
    # Use regex to remove the last comma before the closing brace/bracket
    json_str = re.sub(r',\s*([\]}])', r'\1', json_str)
    return json_str

def custom_llm_result_parsing(llm_res, llm_agent, json_parse=True,parser=None, 
                              output_fixing_pt_temp=None,verbose=False):
    """
    Parses the response from an LLM agent using a specified parser. If the parser fails,
    it attempts to parse the response as JSON, and if that fails, it uses an optional
    output fixing prompt template to reformat the response.

    Args:
        llm_res (str): The response string from the LLM agent to be parsed.
        llm_agent (object): The LLM agent object which has methods for parsing and getting responses.
        json_parse (bool, optional): Flag to indicate if a basic JSON parsing should be attempted if the parser fails. Defaults to True.
        parser (object, optional): A custom parser object with a parse method. Defaults to None.
        output_fixing_pt_temp (str, optional): A prompt template string for the LLM agent to fix the output format. Defaults to None.
        verbose (bool, optional): Flag to print detailed error messages and steps. Defaults to False.

    Returns:
        dict or None: The parsed response as a dictionary, or None if all parsing attempts fail.
    """

    if parser:
        res_dict = parse_result(llm_res, parser,verbose)
        if res_dict is None:
            if json_parse:
                try:
                    res_dict_str = llm_agent.extract_json_string(llm_res)
                    res_dict_str = fix_trailing_comma(res_dict_str)
                    res_dict = json.loads(res_dict_str)
                    if verbose:
                        print('Use basic json parse to fix output ...')
                except Exception as e:
                    if verbose:
                        print(f"JSON parsing error: {e}")
                    res_dict = None
            if res_dict is None and output_fixing_pt_temp:
                try:
                    new_res = llm_agent.get_response_content(prompt_template=output_fixing_pt_temp, max_tokens=4096,temperature=0)
                    #print(res)
                    res_dict = custom_llm_result_parsing(new_res, llm_agent,json_parse,parser, False)
                    if verbose:
                        print('Use llm to fix output ...')
                except Exception as e:
                    if verbose:
                        print(f"Output fixing template error: {e}")
                    res_dict = None
        return res_dict
    else:
        return None
    
#%%
if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--input_file", type=str, required=True)
    # parser.add_argument("--output_file", type=str, required=True)
    # args = parser.parse_args()
    
    class ClimateClassification_2label(BaseModel):
        justification: str
        classification: Literal["favorable", "unfavorable"]
    
    #%%
    llm_agent = BSAgent(
        base_url='http://localhost:8100/v1',
        api_key='abc',
        temperature=0,
    )
    #res = llm_agent.connection_test()
    #%%
    data_dir = Path('/ephemeral/home/xiong/data/Fund/Climate')
    input_dir = data_dir / 'batch_tasks_results_v1_formated_output'
    input_files = [f for f in input_dir.glob('*.jsonl') if f.name.startswith('results_') and not f.name.startswith('.')]
    output_dir = data_dir / 'batch_tasks_results_v1_parsed_output_formated'
    output_dir.mkdir(parents=True, exist_ok=True)

    #%%
    input_file = input_files[0]
    parser = PydanticOutputParser(pydantic_object=ClimateClassification_2label)
    error_files = []
    for input_file in tqdm(input_files, desc="Processing input files"):
        with open(input_file, 'r') as f:
            raw_output = [json.loads(line) for line in f if line.strip()]
        parsed_results = []
        
        for response in raw_output:
            request_id = response['custom_id']
            if not response.get('error'):
                res = response['response']['body']['choices']['message']['content']
                parse_res = custom_llm_result_parsing(res, llm_agent, True, parser,verbose=False) 
                # did not pass in error fixing template no llm will be called to fix the output
                try:
                    parsed_results.append((request_id,parse_res['classification'],parse_res['justification']))
                except:
                    parsed_results.append((request_id,None,None))
            else:
                print(f"Error: {response['error']}")
                parsed_results.append((request_id,None,None))
        
        none_count = sum(1 for result in parsed_results if result[1] is None or result[2] is None)
        if none_count/len(parsed_results) > 0.01:
            error_files.append(input_file)
            print(f"Error rate: {(none_count/len(parsed_results))*100:.2f}%")
        else:
            parsed_df = pd.DataFrame(parsed_results, columns=['request_id', 'classification','justification'])
            parsed_df.to_csv(output_dir / f"{input_file.stem}.csv", index=False)
            
            print(f"Number of None results: {none_count}")
            print(f"Total results: {len(parsed_results)}")
            print(f"Percentage of None results: {(none_count/len(parsed_results))*100:.2f}%")

    #%%
    if error_files:
        print("\nFiles with high error rates (>1%):")
        for file in error_files:
            print(f"- {file}")
        
        if input("\nDelete these files? (yes/no): ").lower() == 'yes':
            for file in error_files:
                try:
                    file.unlink()
                    print(f"Deleted {file}")
                except Exception as e:
                    print(f"Error deleting {file}: {str(e)}")
        else:
            print("No files were deleted")
    else:
        print("\nNo files with high error rates found")


# %%
