import ijson
import tensorflow_hub as hub
import numpy as np
from bs4 import BeautifulSoup
import sys
import re
import math
from sentence_transformers import SentenceTransformer

def calculate_stats():
    urlMap = {}
    with open('../../data/codeGraph/stackoverflow_questions_per_class_func_3M_filtered_new.json', 'r') as data:
        jsonCollect = ijson.items(data, 'results.bindings.item')
        for jsonObject in jsonCollect:
            objectType = jsonObject['class_func_type']['value'].replace('http://purl.org/twc/graph4code/ontology/', '')
            if objectType != 'Class':
                continue
            questionUrl = jsonObject['q']['value']
            stackQuestion = jsonObject['content_wo_code']
            urlMap[questionUrl] = stackQuestion
        

    embed = hub.load('https://tfhub.dev/google/universal-sentence-encoder/4')
    numPosts = 0
    numPostsWithLink = 0
    numPostsWithProperLink = 0
    for url in urlMap.keys():
        i += 1
    for url, question in urlMap.items():
        
'''    with open('../../data/codeGraph/stackoverflow_questions_per_class_func_3M_filtered_new.json', 'r') as data:
        jsonCollect = ijson.items(data, 'results.bindings.item')
        for jsonObject in jsonCollect:
            objectType = jsonObject['class_func_type']['value'].replace('http://purl.org/twc/graph4code/ontology/', '')
            if objectType != 'Class':
                continue
#            print("Parsing object")
            label = jsonObject['class_func_label']['value']
            stackQuestion = jsonObject['content_wo_code']
            soup = BeautifulSoup(stackQuestion, 'html.parser')
            foundLink = False
            foundProperLink = False
            for link in soup.find_all('a'):
                actualLink = link.get('href')
                if 'https://stackoverflow.com/questions/' in actualLink:
                    if not foundLink:
                        numPostsWithLink += 1
                        foundLink = True
                    properLink = re.search('https:\/\/stackoverflow\.com\/questions\/\d+', actualLink)
                    linkToSearch = properLink.group(0)
                    print('Searching for', linkToSearch)
                    if linkToSearch in urlMap:
                        if not foundProperLink:
                            foundProperLink = True
                            numPostsWithProperLink += 1
                    break
            numPosts += 1
#                print(actualLink)
    print(numPosts)
    print(numPostsWithLink)
    print(numPostsWithProperLink)
    print(numPostsWithLink/numPosts)'''

if __name__ == "__main__":
    calculate_stats()
