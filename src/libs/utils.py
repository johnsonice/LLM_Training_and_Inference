## utils 
#%%
from huggingface_hub import hf_hub_download
from huggingface_hub import snapshot_download
from dotenv import load_dotenv
import os
env_path = '../../.env'
load_dotenv(dotenv_path=env_path)

#%%

def download_hf_model(REPO_ID, save_location, hf_token=None):
    # REPO_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
    # save_location = '/root/data/hf_cache/llama-3-8B-Instruct'
    if hf_token is None:
        hf_token = os.getenv('huggingface_token')
        if hf_token is None:
            hf_token = input("huggingface token:")
            
    snapshot_download(repo_id=REPO_ID,
                    local_dir=save_location, 
                    token=hf_token)
    
    return save_location
    ## you can also use hf cli 
    ## huggingface-cli download meta-llama/Meta-Llama-3-8B --include "original/*" --local-dir meta-llama/Meta-Llama-3-8B

    
#%%