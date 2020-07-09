import ijson
import tensorflow_hub as hub
import numpy as np
from bs4 import BeautifulSoup
import sys
import re

def sort_answers():
    totalQuestions = 0
    totalCorrectComparisons = 0
    embed = hub.load('https://tfhub.dev/google/universal-sentence-encoder/4')
    with open('../../data/codeGraph/stackoverflow_questions_per_class_func_3M_filtered_new.json', 'r') as data:
        jsonCollect = ijson.items(data, 'results.bindings.item')
        recipRanks = []
        f = 0
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
            if len(voteOrder) != 1: #change this to 1 for multi answer posts, and 0 for including singles
                correctAnswer = voteOrder[0][1]
                for x in range(0, len(distanceOrder)):
                    rank = x + 1
                    reciprocal = 1/rank
                    if distanceOrder[x][1] == correctAnswer:
                        recipRanks.append(reciprocal)
                        break
        meanRecipRank = sum(recipRanks)/len(recipRanks)
        print("Mean reciprocal rank is:", meanRecipRank)
    

if __name__ == "__main__":
    sort_answers()
