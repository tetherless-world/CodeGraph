
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
def build_index():
    embed = hub.load('https://tfhub.dev/google/universal-sentence-encoder/4')
    index = faiss.IndexFlatL2(512)
    docMessages = []
    
    embeddingtolabelmap = {}
    labeltotextmap = {}
    embedCollect = set()
    duplicateClassDocString=set()
    droppedClassWithLessLength=set()
    docStringLength_avg=[]
    bucket_size=4
    docLabelToTextForSentenceTokenizationAndAnalysis= {}
    with open('../../data/codeGraph/stackoverflow_questions_per_class_func_3M_filtered_new.json', 'r') as data,open('./lengthAnalysisDocstrings_NewJson.txt', 'w') as outputFile:
        jsonCollect = ijson.items(data, 'results.bindings.item')
        i = 0
        originalout = sys.stdout
        sys.stdout = outputFile

        for jsonObject in jsonCollect:
            objectType = jsonObject['class_func_type']['value'].replace(
                'http://purl.org/twc/graph4code/ontology/', '')
            if objectType != 'Class':
                continue
            label = jsonObject['class_func_label']['value']
            docLabel = label
            docStringText = jsonObject['docstr']['value']# + ' ' + str(i)
            soup = BeautifulSoup(docStringText, 'html.parser')
            for code in soup.find_all('code'):
                code.decompose()
            docStringText = soup.get_text()
            
            if docStringText in embedCollect:
                ##had to include this because the same class was getting addded into the mapped array
                if docLabel in duplicateClassDocString:
                    pass
                    
                else:
#                     if len(docStringText) < 300:
#                         print("has less than 300 character,class:",docLabel,"docstring:",docStringText)
#                         droppedClassWithLessLength.add(docLabel)
#                         continue
                    duplicateClassDocString.add(docLabel)
                    print("doclabel",docLabel)
                    docLabelToTextForSentenceTokenizationAndAnalysis[docLabel]=docStringText
                    

                    for formbucket in range(0,len(sent_docStrings),bucket_size):
                            if formbucket >= len(sent_docStrings)-bucket_size :
                                text_combined=""
                                for t in sent_docStrings[formbucket:len(sent_docStrings)]:
                                    text_combined=text_combined+t
                                embeddedDocText = embed([text_combined])[0]
                                embeddingtolabelmap[tuple(
                                embeddedDocText.numpy().tolist())] = [docLabel]
            else:
#                 if len(docStringText) < 300:
#                         print("has less than 300 character,class:",docLabel,"docstring:",docStringText)
#                         droppedClassWithLessLength.add(docLabel)
#                         continue
                duplicateClassDocString.add(docLabel)
                embedCollect.add(docStringText)
                        ##had  to include this because the same class was getting addded into the mapped array

                docStringLength_avg.append(len(docStringText))
                docLabelToTextForSentenceTokenizationAndAnalysis[docLabel]=docStringText
                sent_docStrings=sent_tokenize(docStringText)
                print("doclabel",docLabel)
                for formbucket in range(0,len(sent_docStrings),bucket_size):
                            if formbucket >= len(sent_docStrings)-bucket_size :
                                text_combined=""
                                for t in sent_docStrings[formbucket:len(sent_docStrings)]:
                                    text_combined=text_combined+t
                                embeddedDocText = embed([text_combined])[0]
                                newText = np.asarray(
                                embeddedDocText, dtype=np.float32).reshape(1, -1)
                                docMessages.append(embeddedDocText.numpy().tolist())
                                index.add(newText)
                                embeddingtolabelmap[tuple(
                                embeddedDocText.numpy().tolist())] = [docLabel]
                                    
                            else:
                                text_combined=""
                                for t in sent_docStrings[formbucket:formbucket+bucket_size]:
                                    text_combined=text_combined+t
                                embeddedDocText = embed([text_combined])[0]
                                newText = np.asarray(
                                embeddedDocText, dtype=np.float32).reshape(1, -1)
                                docMessages.append(embeddedDocText.numpy().tolist())
                                index.add(newText)
                                embeddingtolabelmap[tuple(
                                embeddedDocText.numpy().tolist())] = [docLabel]

            i += 1
        sys.stdout=originalout

        return (index, docMessages, embeddingtolabelmap, docStringLength_avg,droppedClassWithLessLength,docLabelToTextForSentenceTokenizationAndAnalysis,bucket_size)


def evaluate_neighbors(index, docMessages, embeddingtolabelmap,docStringLength_avg,droppedClassWithLessLength,docLabelToTextForSentenceTokenizationAndAnalysis,bucket_size):
    k = 10*3
    fp=0
    fn=0
    tp=0
    tn=0
    efp=0
    efn=0
    etp=0
    etn=0
    positivepresent=False
    exactpositivepresent=False
    totaldocs=0
    embed = hub.load('https://tfhub.dev/google/universal-sentence-encoder/4')
    originalout = sys.stdout
    with open('../../data/codeGraph/stackoverflow_questions_per_class_func_3M_filtered_new.json', 'r') as data, open('./lengthAnalysisStackNewJson.txt', 'w') as outputFile:
        
        jsonCollect = ijson.items(data, 'results.bindings.item')
        sys.stdout = outputFile
        stack_overflow_length=[]
        for jsonObject in jsonCollect:
            totaldocs+=1
            objectType = jsonObject['class_func_type']['value'].replace(
                'http://purl.org/twc/graph4code/ontology/', '')
            if objectType != 'Class':
                continue
            title = jsonObject['title']['value']
            classLabel = jsonObject['class_func_label']['value']
            if classLabel in droppedClassWithLessLength:
                continue  

            stackText = jsonObject['content_wo_code']+ \
                " " + jsonObject['answer_wo_code']
            soup = BeautifulSoup(stackText, 'html.parser')

            for code in soup.find_all('code'):
                code.decompose()
            stackText = soup.get_text()
#             if len(stackText) < 50:
#                 continue
#              print('\nTitle of Stack Overflow Post:', title)
            print("---------------------------------------------------\n")
            print('Class associated with post:', classLabel)
            splitLabel = classLabel.lower().split('.')
            wholePattern = re.compile(classLabel.lower(), re.IGNORECASE)
            maskedText = wholePattern.sub(' ', stackText)
            for labelPart in splitLabel:
                    partPattern = re.compile(labelPart, re.IGNORECASE)
                    maskedText = partPattern.sub(' ', maskedText)#maskedText.replace(labelPart, ' ')

            embeddedText = embed([stackText])#[maskedText])

            embeddingVector = embeddedText[0]
            embeddingArray = np.asarray(
                embeddingVector, dtype=np.float32).reshape(1, -1)
            D, I = index.search(embeddingArray, k)
            distances = D[0]
            indices = I[0]
#             print("Distances of related vectors:", distances)
#             print("Indices of related vectors:", indices)
            positivepresent=False
            exactpositivepresent=False
            finallabels=[]
            actuallabelcount=0
            for p in range(0, k):
                if actuallabelcount >= (k / 3):
                    break

                properIndex = indices[p]
                embedding = docMessages[properIndex]
                adjustedembedding = tuple(embedding)
                label = embeddingtolabelmap[adjustedembedding]
                if label in finallabels:
                    continue
                actuallabelcount=actuallabelcount+1

                
                ##multiple docstrings associated with the same embedding mapped
                ##array of labels mapped
                j=0
                for l in label:
                    finallabels.append(l)
                    if l.startswith(classLabel.split(".")[0]):
                        positivepresent=True
                        if j == 0:
                             print("\n True positive label being contributed by \n",l)
                        else:
                            print("and \t",l)
                    else:
                        print("class not associated",l)
                    if l == classLabel:
                        exactpositivepresent=True
                        print("\n Exact positive label being contributed by \n",l)
                    j=j+1
                        
            if not positivepresent:
                fp=fp+1
                print("Loose False Positive Present \n")
                print("Investigating the reason with sentence tokenized docstring for:", classLabel,"\n")
                print(sent_tokenize(docLabelToTextForSentenceTokenizationAndAnalysis[classLabel]))
            else:
                tp=tp+1
#                 print("Loose True Positive Present -------------------------------------------------------- \n")
            if not exactpositivepresent:
                efp=efp+1
#                 print("match  False Positive Present ------------------------------------------------------- \n")
            else:
                etp=etp+1
            
#                 print("match True Positive Present -------------------------------------------------------- \n")
        print("--------------------------------------------- \n")
        

        print(tp/(tp+fp), " Loose Precision at 10 without masking  ")
        print(etp/(etp+efp), "Exact Precision at 10 without masking  and bag size=",bucket_size)

        sys.stdout=originalout

if __name__ == '__main__':
    dataTuple = build_index()
    print("Completed building index.")
    evaluate_neighbors(dataTuple[0], dataTuple[1], dataTuple[2], dataTuple[3],dataTuple[4],dataTuple[5], dataTuple[6])
