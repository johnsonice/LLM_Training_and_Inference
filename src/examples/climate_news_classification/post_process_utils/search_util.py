# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 11:21:13 2022

@author: CHuang
"""

import re
from collections import Counter
import pandas as pd 

def get_keywords_groups(key_path,clean=False,clean_keys=None,sheet_name=None):
    if sheet_name:
        key_df = pd.read_excel(key_path,sheet_name=sheet_name)
    else:
        key_df = pd.read_excel(key_path)
        
    key_group_dict = key_df.to_dict('list')
    for k in key_group_dict.keys():
        key_group_dict[k] = [i.strip('\xa0') for i in key_group_dict[k] if not pd.isna(i)]    
        if clean:
            ## if clean keys function was passed, process keywords 
            key_group_dict[k] = clean_keys(key_group_dict[k])
            
    return key_group_dict


def construct_rex(keywords,plural=True,casing=False):
    """
    construct regex for multiple match 
    """
    if plural:
        r_keywords = [r'\b' + re.escape(k) + r'(s|es)?\b'for k in keywords]    # tronsform keyWords list to a patten list, find both s and es 
    else:
        r_keywords = [r'\b' + re.escape(k) + r'\b'for k in keywords]
    
    if casing:
        rex = re.compile('|'.join(r_keywords)) 
    else:
        rex = re.compile('|'.join(r_keywords),flags=re.I)                       # use or to join all of them, ignore casing
        #match = [(m.start(),m.group()) for m in rex.finditer(content)]         # get the position and the word
    return rex
    

def construct_rex_group(key_group_dict):
    """
    construct a group of regular expression patterns 
    """
    reg_dict = {}
    for k in key_group_dict.keys():
        reg_dict[k] = construct_rex(key_group_dict[k])
        
    return reg_dict

def find_exact_keywords(content,keywords=None,content_clean=True,rex=None,return_count=True):
    if rex is None: 
        rex = construct_rex(keywords)
    
    if content_clean:
        content = content.replace('\n', '').replace('\r', '')#.replace('.',' .')
    
    match = Counter([m.group() for m in rex.finditer(content)])             # get all instances of matched words 
                                                                            # and turned them into a counter object, to see frequencies
    total_count = sum(match.values())
    
    if return_count:
        return match,total_count
    else:
        return match