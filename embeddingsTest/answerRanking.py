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
            matching = True
            if len(voteOrder) != 0:         
                for i in range(0, len(voteOrder)):
                    voteAnswer = voteOrder[i][1]
                    distanceAnswer = distanceOrder[i][1]
                    if voteAnswer != distanceAnswer:
                        matching = False
                totalQuestions += 1
                if matching:
                    totalCorrectComparisons += 1
            '''voteValues = []
            distanceValues = []
            for p in range(0, len(voteOrder)):
                voteValues.append(voteOrder[p][0])
                distanceValues.append(distanceOrder[p][0])
            print(voteValues)
            print(distanceValues)'''
#            if f == 300:
#                break
    
        print('Total Questions:', totalQuestions)
        print('Total Number Of Correct Orders:', totalCorrectComparisons)

if __name__ == "__main__":
    sort_answers()
