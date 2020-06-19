import ijson
from bs4 import BeautifulSoup
import re
import tensorflow_hub as hub
import faiss
import pickle
import numpy as np
import sys
def analyze_posts():
        classDocStrings={}
        sampleClassDocStrings=[]
        sampleStackOverFlow=[]
        with open('../../data/codeGraph/merge-15-22.2.format.json','rb') as data:
                    docStringObjects = ijson.items(data, 'item')
                    parsedLines = 0
                    for docString in docStringObjects:
                        parsedLines += 1
                        if parsedLines % 10000 == 0:
                            if 'class_docstring' in docString:
                                if 'class_docstring' != None:
                                    className = docString['klass']
                                    classDocStrings[className] = 1
                                    sampleClassDocStrings.append(className)
                            
                        if 'class_docstring' in docString:
                            if 'class_docstring' != None:
                                className = docString['klass']
                                classDocStrings[className] = 1
                            else:
                                pass
        with open('../../data/codeGraph/stackoverflow_questions_per_class_func_1M.json', 'r') as data:
                docStringObjects = ijson.items(data, 'results.bindings.item')
                countClassNotinDocstring=0
                countClassInStackOverflow=0

                with open('../../data/codeGraph/StackOverFlowDocstringSimilarityAnalysis.txt', 'w') as outputFile:
                        print("comparing with stackoverflow posts")
                        sys.stdout=outputFile
                        for docString in docStringObjects:
                                parsedLines += 1
                                classType = docString['type']['value'].replace('http://purl.org/twc/graph4code/ontology/', '')
                                if classType != 'Class':
                                        continue
                                classLabel = docString['class_func_label']['value'].lower() # this might not be needed
                                if classLabel not in classDocStrings:
                                    countClassNotinDocstring+=1
                                    countClassInStackOverflow+=1 
                                else:
                                    countClassInStackOverflow+=1 
                                if parsedLines % 5000 == 0:
                                    sampleStackOverFlow.append(classLabel)
                                    
                                    

                                
                        print("number of classes not in docstring:", countClassNotinDocstring)
                        print("number of classes in Stackoverflow:", countClassNotinDocstring)
                        print("number of unique docstrings",len(classDocStrings))
                        print("Class doc strings",sampleClassDocStrings)
                        print("Class stackoverflow strings",sampleStackOverFlow)
                        
                       
       
                                



if __name__ == '__main__':
    analyze_posts()
