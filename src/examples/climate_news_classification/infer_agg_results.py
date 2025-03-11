### infer_agg_results
# Created on: 2024-01-09
# Author: Chengyu
#%%
import os,sys
import pandas as pd
from tqdm.auto import tqdm
import numpy as np
from post_process_utils.country_dict_full import get_dict  
from post_process_utils.identify_countries import construct_country_group_rex,get_one_country_name,construct_country_rex
from pathlib import Path
#%%
#%%
climate_keys = ['policy', 'government', 'adaptation', 'mitigation',
    'climate adaptation', 'climate mitigation', 'non-market base', 'tax',
    'tax on fuel consumption', 'tariff', 'subsidies', 'consumption subsidy',
    'producer subsidy', 'fuel subsidy', 'preserve incentive',
    'electric vehicle incentive', 'tradable permit', 'liability rule',
    'deposit-refund', 'ban', 'performance standard',
    'environmental reporting', 'public adoption', 'investment', 'r & d',
    'energy', 'fiscal instrument', 'training', 'voluntary agreement',
    'risk', 'physical', 'physical risk', 'transition', 'transition risk',
    'financial', 'financial risk', 'liability', 'liability risk',
    'climate tag', 'policy tag', 'risk tag', 'climate policy',
    'climate policy risk', 'climate risk', 'monetary policy',
    'climate change', 'global warm', 'carbon', 'emissions',
    'carbon emission', 'greenhouse gas', 'sea-level rise', 'low carbon']

keep_cols = ['id','pub_date', 'source', 'language', 'title', 'body', 'Year', 'ym',
             'climate_count','climate_binary_org','climate_binary_org_no_adaptation', 'climate_binary_org_no_adaptation_esg',
             'climate_binary_org_no_adaptation_esg_canada','climate_policy_tag_no_adaptation','climate_policy_tag_no_adaptation_esg',
             'climate_policy_tag_no_adaptation_esg_canada',
             'positive', 'negative', 'label','harmonized_classification'] #'netural',

versions = ['climate_binary_org','climate_binary_org_no_adaptation', 'climate_binary_org_no_adaptation_esg',
             'climate_binary_org_no_adaptation_esg_canada','climate_policy_tag_no_adaptation','climate_policy_tag_no_adaptation_esg',
             'climate_policy_tag_no_adaptation_esg_canada']
        
def agg_by(df,group_cols:list,value_cols_dict:dict):
    """
    aggregate by group and agg values cols 

    """
    agg_df = df.groupby(group_cols).agg(value_dict)
    agg_df.columns = agg_df.columns.map("_".join)
    agg_df.reset_index(inplace=True)
    
    return agg_df

def tag_esg(input_txt,key='esg'):
    if isinstance(input_txt,str):
        content = input_txt.lower().split()
        if key in content:
            return 1
        else:
            return 0 
    else:
        return 0 
    
def tag_country_canada(text,country_rex):
    try:
        rc = country_rex.findall(text.lower())
        if len(rc)>0:
            return 1
        else:
            return 0 
    except:
        return 0 
    
#%%
if __name__ == "__main__":    
    
    data_dir = Path('/ephemeral/home/xiong/data/Fund/Climate')
    res_dir = data_dir / 'pred_results'
    out_agg_dir = data_dir / 'agg_results'
    out_agg_dir.mkdir(parents=True, exist_ok=True)
    res_files = os.listdir(res_dir)
    res_files = [f for f in res_files if '~' not in f]
    #%%
    # test_df = pd.read_csv(res_dir / res_files[0], encoding='utf8')
    #%%
    ## get country map 
    countr_dict = get_dict()
    country_rex = construct_country_rex(countr_dict['canada'])  ## only search for canada 
    #country_rex_dict = construct_country_group_rex(countr_dict)

    #%%
    for version in tqdm(versions):
        print('working on version : {}'.format(version))
        out_agg_path = out_agg_dir / 'final_agg_2label_{}.csv'.format(version)
        ## agg columns 
        agg_res_dfs = []
        group_cols = ['ym',version,'harmonized_classification'] # 'climate_binary' #'climate_binary_org' # 'climate_binary_no_adaptation','climate_policy_tag_no_adaptation'
        value_dict = {'harmonized_classification':['count']}
    
        for f in tqdm(res_files):
            df = pd.read_csv(res_dir / f, encoding='utf8')
            df['climate_count'] = df[climate_keys].sum(axis=1)
            df['esg'] = df['body'].apply(tag_esg)
            df['esg_only'] = df['body'].apply(tag_esg)
            df['canada'] = df['body'].apply(tag_country_canada,args=(country_rex,))
            #df['climate_binary'] = df['climate_count']>0
            
            #df['climate_count'] = df[climate_keys].sum(axis=1)
            df['climate_binary_org'] = df['climate_count']>0
            df['climate_binary_org_no_adaptation'] = np.where((df['climate_count'] >0) & (df['adaptation'] == 0), 1, 0)
            #df['climate_binary_org_no_adaptation'] = np.where((df['climate_count'] - df['adaptation'] > 0), 1, 0)
            df['climate_binary_org_no_adaptation_esg'] = np.where((df['climate_count'] >0) & (df['adaptation'] == 0) & (df['esg'] == 0), 1, 0)
            df['climate_binary_org_no_adaptation_esg_canada'] = np.where((df['climate_binary_org_no_adaptation_esg']  >0) & (df['canada']> 0), 1, 0)
            
            df['climate_policy_tag_no_adaptation'] = np.where((df['climate tag'] >0) & (df['policy tag'] - df['adaptation'] > 0), 1, 0)
            df['climate_policy_tag_no_adaptation_esg'] = np.where((df['climate tag'] >0) & (df['policy tag'] - df['adaptation'] > 0)  & (df['esg'] == 0), 1, 0)
            df['climate_policy_tag_no_adaptation_esg_canada'] = np.where((df['climate_policy_tag_no_adaptation_esg'] >0) & (df['canada']> 0), 1, 0)
            
            df = df[keep_cols]
            agg_df = agg_by(df,group_cols,value_dict)  ## aggregate on smaller chunks are quicker 
            agg_res_dfs.append(agg_df)
        
        final_agg_df = pd.concat(agg_res_dfs)
        final_agg_df = final_agg_df.groupby(group_cols)['harmonized_classification_count'].sum()
        #label_map = {0:'neutral',1:'positive',2:'negative'}
        #label_map = {0:'negative',1:'positive'}
        final_agg_df.name='harmonized_classification_count'
        final_agg_df = pd.DataFrame(final_agg_df)
        final_agg_df.reset_index(inplace=True)
        #final_agg_df['label'].replace(label_map,inplace=True)
        final_agg_df.rename(columns={'harmonized_classification': 'label', 'harmonized_classification_count': 'label_count'}, 
                            inplace=True)
        final_agg_df.to_csv(out_agg_path,encoding='utf8',index=False)
        print('save results to {}'.format(out_agg_path))