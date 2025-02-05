# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 20:26:09 2021
@author: YZhang5
"""
#%%
import os 
import numpy 
import pandas 
from bs4 import BeautifulSoup
#%%
current_dir = os.path.dirname(__file__)

def find_item(c,category):
    try:
        item=c.find(category).get_text().lower().strip().split('\n')
        item=[i.replace('_',' ').strip() for i in item]  
    except:
        item=[]    
    return item

def find_president(c):
    try:
        item=c.find('leaders').find('prime').get_text().lower().strip().split('\n')
        item=[i.split('_\t')[0].replace('_',' ').strip() for i in item]  
    except:
        item=[]    
    return item

def get_dict(country_xml_path = os.path.join(current_dir,'CountryInfo.xml')): 
    infile=open(country_xml_path,'r')
    cotents=infile.read()
    soup=BeautifulSoup(cotents,'html.parser')
    country=soup.select('Country')       
    cdict={}
    for c in country:
        countryname=find_item(c, 'countryname')
        nationality=find_item(c, 'nationality')
        capital=find_item(c, 'capital')
        majorcities=find_item(c, 'majorcities')
        regions=find_item(c, 'regions')
        presidents = find_president(c)
        
        ccontent=[]
        ccontent+=countryname
        ccontent+=nationality
        ccontent+=capital
        ccontent+=majorcities
        ccontent+=regions
        ccontent+=presidents
        ccontent = list(filter(None, ccontent))
            
        cdict[''.join(countryname)]=ccontent
        
    return cdict

#%%
if __name__ == '__main__':
    country_dict=get_dict()
