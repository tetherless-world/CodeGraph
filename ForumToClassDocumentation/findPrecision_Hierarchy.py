##simple calculation for precision,given classes and their nearest neighbors computed with the help of embedDocstringsOnlyFlowDistantBasedHierarchy.py 
##which is the data located at https://ia601401.us.archive.org/30/items/classes2superclass/class2top10neighbors_withScore.txt


##Other data needed 
##https://archive.org/download/classes2superclass/classes2superclass.out
##run it: python findPrecision_Hierarchy.py  ../../data/codeGraph/classes2superclass.out class2top10neighbors_withScore.txt
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
def findPrecision(input_File1,input_File2):
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
    with open(input_File1, 'r') as class2superclass, open(input_File2, 'r') as class2top10neighbors_withScore:
        ##very basic computation of precision
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
            if splitArr[2]!="":
                if int(splitArr[2]) > 0:
                    positive=True
                
            else:
                if int(splitArr[3]) > 0:
                    positive=True
            j=j+1
            
                
        print("Precision",plus/(plus+minus))
        ##parses through and computes required precision
            
            
            
       




if __name__ == '__main__':
    dataTuple = findPrecision(sys.argv[1:][0],sys.argv[1:][1])
    print("Complete")
