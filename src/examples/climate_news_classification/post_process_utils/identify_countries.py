# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 15:49:47 2023

@author: CHuang
"""

import os,sys
import pandas as pd 
from .country_dict_full import get_dict  
from .search_util import construct_rex, construct_rex_group
import re
from tqdm import tqdm

#%%
def construct_country_group_rex(country_dict):
    """
    construct a dictionary of regex patterns 
    """
    country_rex_dict = {}
    for c,v in country_dict.items():
        if c in ['united-states','united states of america']:
            rex = construct_rex(v,casing=True)
        else:
            rex = construct_rex(v)
        
        country_rex_dict[c] = rex
    
    return country_rex_dict

def construct_country_rex(key_list):
    rex = construct_rex(key_list)    
    return rex

def get_one_country_name(text,country_rex):
    """
    use regex patterns to match 
    """
    rc = country_rex.findall(text.lower(),re.IGNORECASE)
    
    return rc
    
def get_country_name(text,country_rex_dict):
    """
    use regex patterns to match 
    """
    for c,rex in country_rex_dict.items():
        rc = rex.findall(text)
        if len(rc)>0:
            yield c

def tag_country_canada(text,country_rex_dict,filter_country='canada'):
    try:
        res = list(get_country_name(text,country_rex_dict))
        if filter_country in res:
            return 1
    except:
        return 0 
  

# #%%
# if __name__ == "__main__":
    
    # in_data_folder = 'Data/Market_Non-Market_by_year_withSentiment'
    # out_data_folder = 'Data/Market_Non-Market_by_year_withSentiment_country'
    # keep_columns = ['created_at', 'lang', 'id', 'text_processed', 'negative','netural', 'positive', 'label','Country']
    # ## get all file names 
    # in_files = os.listdir(in_data_folder)
    # ## construct country search regex 
    # countr_dict = get_dict()
    # country_rex_dict = construct_country_group_rex(countr_dict)
    
    # #%%
    # for i in tqdm(in_files):
    #     in_fn = os.path.join(in_data_folder,i)
    #     out_fn = os.path.join(out_data_folder,i)
    #     df = pd.read_csv(in_fn,engine='python',error_bad_lines=False)
    #     df['Country'] = df['text_processed'].apply(tag_country,args=(country_rex_dict,))
    #     df = df[keep_columns]
    #     df.to_csv(out_fn,index=False)
