import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import glob
import json
from pprint import pprint
from concurrent.futures import ProcessPoolExecutor
from itertools import chain
import nltk
import re
import os
from pathlib import Path

class FileReader:
    def __init__(self, file_path):
        with open(file_path) as file:
            content = json.load(file)
            self.paper_id = content['paper_id']
            self.abstract = []
            self.body_text = []
            # Abstract
            for entry in content['abstract']:
                self.abstract.append(entry['text'])
            # Body text
            for entry in content['body_text']:
                self.body_text.append(entry['text'])
            self.abstract = '\n'.join(self.abstract)
            self.body_text = '\n'.join(self.body_text)
    def __repr__(self):
        return f'{self.paper_id}: {self.abstract[:200]}... {self.body_text[:200]}...'


def normalize_text(text):
    # lowercase text
    text = str(text).lower()
    # remove non-UTF
    text = text.encode("utf-8", "ignore").decode()
    # remove punktuation symbols
    #text = " ".join(regex_tokenizer.tokenize(text))
    return text

def count_lines(filename):
    count = 0
    with open(filename) as fi:
        for line in fi:
            count += 1
    return count



if __name__ == '__main__':

	GLOBAL_PATH="../_data/CORD-19-research-challenge"
	#change to argument to pass to function 
	biorxiv = '/biorxiv_medrxiv'
	bioxiv_json = glob.glob(f'{GLOBAL_PATH+biorxiv}/**/*.json', recursive=True)

	
	metadata_path = f'{GLOBAL_PATH}/metadata.csv'
	meta_df = pd.read_csv(metadata_path, dtype={
	    'pubmed_id': str,
	    'Microsoft Academic Paper ID': str, 
	    'doi': str
	})
	
	
	dict_ = {'paper_id': [], 'abstract': [], 'body_text': [], 'title': []}
	for idx, entry in enumerate(bioxiv_json):
	    if idx % (len(bioxiv_json) // 10) == 0:
	        print(f'Processing index: {idx} of {len(bioxiv_json)}')
	    
	    try:
	        content = FileReader(entry)
	    except Exception as e:
	        continue  # invalid paper format, skip
	    
	    # get metadata information
	    meta_data = meta_df.loc[meta_df['sha'] == content.paper_id]
	    # no metadata, skip this paper
	    if len(meta_data) == 0:
	        continue
	    
	    dict_['abstract'].append(content.abstract)
	    dict_['paper_id'].append(content.paper_id)
	    dict_['body_text'].append(content.body_text)
	    
	    # get metadata information
	    meta_data = meta_df.loc[meta_df['sha'] == content.paper_id]
	    
	    
	    # add the title information, add breaks when needed
	    dict_['title'].append(meta_data['title'])

    
    
df_covid = pd.DataFrame(dict_, columns=['paper_id', 'abstract', 'body_text', 'title'])









	with open(PRC_DATA_FPATH, "w",encoding="utf-8") as fo:
	    for l in df_covid["body_text"]:
	        fo.write(normalize_text(l)+"\n")






