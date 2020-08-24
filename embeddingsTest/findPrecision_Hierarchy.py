##simple calculation for precision and recall
import ijson
import tensorflow_hub as hub
import faiss
import numpy as np
from bs4 import BeautifulSoup
import sys
import re
from statistics import mean 
from statistics import pstdev
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
def findPrecision():
    embed = hub.load('https://tfhub.dev/google/universal-sentence-encoder/4')
    index = faiss.IndexFlatL2(512)
    docMessages = []
    embeddingtolabelmap = {}
    labeltotextmap = {}
    embedCollect = set()
    duplicateClassDocString=set()
    droppedClassWithLessLength=set()
    docStringLength_avg=[]
    docLabelToTextForSentenceTokenizationAndAnalysis= {}
    with open('../../data/codeGraph/stackoverflow_questions_per_class_func_3M_filtered_new.json', 'r') as data,open('./precisionAnalysisHierarchy.txt', 'w') as outputFile,open('../../data/codeGraph/classes2superclass.out', 'r') as class2superclass, open('class2top10neighbors_withScore.txt', 'r') as class2top10neighbors_withScore:
        outputArr=[]
        positive=False
        plus=0
        minus=0
        j=1
        for iterateInOut in class2top10neighbors_withScore:
            if j% 10 == 0:
                if positive== True:
                    plus=plus+1
                else:
                    minus=minus+1
             
                
                    
                positive=False
              
            
            splitArr=iterateInOut.split(" ")
            print(splitArr)
            if splitArr[2]!="":
                if int(splitArr[2]) > 0:
                    positive=True
                
            else:
                if int(splitArr[3]) > 0:
                    positive=True
            j=j+1
            
                
        print("Precision",plus/(plus+minus))
            
            
            
       




if __name__ == '__main__':
    dataTuple = findPrecision()
    print("Complete")
