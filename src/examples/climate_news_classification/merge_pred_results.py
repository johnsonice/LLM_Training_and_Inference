#%%
import pandas as pd
from pathlib import Path
from tqdm import tqdm

#%%

if __name__ == '__main__':
    data_dir = Path('/ephemeral/home/xiong/data/Fund/Climate')
    parsed_result_dir = data_dir / 'batch_tasks_results_v1_parsed_output_formated'
    parsed_result_files = [f for f in parsed_result_dir.glob('*.csv') if f.name.startswith('results_') and not f.name.startswith('.')]
    output_dir = data_dir / 'pred_results'
    output_dir.mkdir(parents=True, exist_ok=True)
    #%%
    # Create list of tuples matching parsed results with original inference files
    parid_files = [(data_dir / 'infer_res_2label' / f.name.replace('results_batch_tasks_',''),f) for f in parsed_result_files ]
    # Check if all original files exist
    missing_files = [f[0] for f in parid_files if not f[0].exists()]
    if missing_files:
        print("Missing original files:")
        for f in missing_files:
            print(f"- {f}")
        raise FileNotFoundError("Some original files are missing")
    
#%%
    for org_file, infer_file in tqdm(parid_files, desc="Merging prediction results"):
        org_df = pd.read_csv(org_file)
        infer_df = pd.read_csv(infer_file)
        assert org_df.shape[0] == infer_df.shape[0]

        org_df['llm_justification'] = infer_df['justification']
        org_df['llm_classification'] = infer_df['classification']

        # Create harmonized classification column
        org_df['harmonized_classification'] = org_df['llm_classification'].map({
            'neutral': 'favorable',
            'favorable': 'favorable',
            'unfavorable': 'unfavorable'
        })

        org_df.to_csv(output_dir / infer_file.name, index=False)
