import ijson
import tensorflow_hub as hub
import numpy as np
from bs4 import BeautifulSoup
import sys
import re
import math
from sentence_transformers import SentenceTransformer

def sort_answers():
    totalQuestions = 0
    totalCorrectComparisons = 0
#    embed = hub.load('https://tfhub.dev/google/universal-sentence-encoder/4') 
    modelList = ['bert-base-nli-stsb-mean-tokens', 'bert-large-nli-stsb-mean-tokens',
     'roberta-base-nli-stsb-mean-tokens','roberta-large-nli-stsb-mean-tokens', 'distilbert-base-nli-stsb-mean-tokens']
    for model in modelList:
        print("Loading model")
        transformer = SentenceTransformer(model)
        print("Model loaded")
        with open('../../data/codeGraph/stackoverflow_questions_per_class_func_3M_filtered_new.json', 'r') as data:
            jsonCollect = ijson.items(data, 'results.bindings.item')
            f = 0
            coefficients = []
            print ("Beginning parsing json")
            for jsonObject in jsonCollect:
                objectType = jsonObject['class_func_type']['value'].replace('http://purl.org/twc/graph4code/ontology/', '')
                if objectType != 'Class':
                    continue
    #            print('parsing question')
                f += 1 
                label = jsonObject['class_func_label']['value']
                stackQuestion = jsonObject['content_wo_code']
#                embeddedQuestion = embed([stackQuestion])
                embeddedQuestion = transformer.encode([stackQuestion])
                embeddingVector = embeddedQuestion[0]
                embeddingQuestionArray = np.asarray(embeddingVector, dtype = np.float32).reshape(1,-1) 
                voteOrder = []
                distanceOrder = []
                voteStats = []
                distanceStats = []
                voteMap = {}
                i = 1
                valid = True
                while valid: 
                    try:
                        index = 'answer_' + str(i)
                        answer = jsonObject[index]['value']
                        answerVotes = jsonObject[index + '_votes']['value']
                        if answerVotes == '':
                            i += 1
                            continue
                        answerVotes = int(answerVotes)
                        soup = BeautifulSoup(answer, 'html.parser')
                        for code in soup.find_all('code'):
                            code.decompose()
                        answer = soup.get_text()
#                        answerEmbed = embed([answer])
                        answerEmbed = transformer.encode([answer])
                        answerVector = answerEmbed[0]
                        answerArray = np.asarray(answerVector, dtype=np.float32).reshape(1,-1)
                        dist = np.linalg.norm(answerArray - embeddingQuestionArray)**2
                        voteOrder.append((answerVotes, answer))
                        distanceOrder.append((dist, answer))
                        voteStats.append(answerVotes)
                        distanceStats.append(dist)
                        voteMap[answer] = answerVotes
                        i += 1
                    except KeyError as e:
                        valid = False
                if not voteOrder:
                    continue               
                voteOrder.sort()
                voteOrder.reverse()
                voteStats.sort()
                voteStats.reverse()
                distanceOrder.sort()
                distanceStats.sort()
    #            print(voteOrder)
    #            print(distanceOrder)
                matching = True
                if len(voteOrder) != 1:
                    # compute DCG
                    i = 1
                    workingDCG = 0
                    for distanceAnswer in distanceOrder:
                        rel = voteMap[distanceAnswer[1]]
                        normal = math.log2(i + 1)
                        totalAdd = rel/normal
                        workingDCG += totalAdd
                        i += 1
                    i = 1
                    workingIDCG = 0
                    for voteAnswer in voteOrder:
                        rel = voteAnswer[0]
                        normal = math.log2(i + 1)
                        totalAdd = rel/normal
                        workingIDCG += totalAdd
                        i += 1
                    nDCG = workingDCG/workingIDCG
                    coefficients.append(nDCG)
            fullNDCG = sum(coefficients)/len(coefficients)
            print("Average NDCG:", fullNDCG)

if __name__ == "__main__":
    sort_answers()
