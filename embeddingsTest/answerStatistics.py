import ijson
import tensorflow_hub as hub
import numpy as np
from bs4 import BeautifulSoup
import sys
import re
import scipy.stats as stat

def sort_answers():
    totalQuestions = 0
    totalCorrectComparisons = 0
    embed = hub.load('https://tfhub.dev/google/universal-sentence-encoder/4')
    with open('../../data/codeGraph/stackoverflow_questions_per_class_func_3M_filtered_new.json', 'r') as data:
        jsonCollect = ijson.items(data, 'results.bindings.item')
        f = 0
        coefficients = []
        for jsonObject in jsonCollect:
            objectType = jsonObject['class_func_type']['value'].replace('http://purl.org/twc/graph4code/ontology/', '')
            if objectType != 'Class':
                continue
#            print('parsing question')
            f += 1 
            label = jsonObject['class_func_label']['value']
            stackQuestion = jsonObject['content_wo_code']
            embeddedQuestion = embed([stackQuestion])
            embeddingVector = embeddedQuestion[0]
            embeddingQuestionArray = np.asarray(embeddingVector, dtype = np.float32).reshape(1,-1) 
            voteOrder = []
            distanceOrder = []
            voteStats = []
            distanceStats = []
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
                    answerEmbed = embed([answer])
                    answerVector = answerEmbed[0]
                    answerArray = np.asarray(answerVector, dtype=np.float32).reshape(1,-1)
                    dist = np.linalg.norm(answerArray - embeddingQuestionArray)**2
                    voteOrder.append((answerVotes, answer))
                    distanceOrder.append((dist, answer))
                    voteStats.append(float(answerVotes))
                    distanceStats.append(dist)
                    i += 1
                except KeyError as e:
                    valid = False
            if not voteOrder:
                continue               
            voteOrder.sort()
            voteOrder.reverse()
            distanceOrder.sort()
#            print(voteOrder)
#            print(distanceOrder)
            matching = True
            if len(voteStats) != 1:
                corr, p = stat.spearmanr(voteStats, distanceStats)
                if not np.isnan(corr):
                    coefficients.append(corr)
        coeffAvg = sum(coefficients)/len(coefficients)
        print("Average Spearman Coefficient is:", coeffAvg)
    

if __name__ == "__main__":
    sort_answers()
