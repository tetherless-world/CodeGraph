import ijson
import tensorflow_hub as hub
import numpy as np
from bs4 import BeautifulSoup
import sys
import re
import math
from sentence_transformers import SentenceTransformer

def beginAnalysis():
    with open('../../data/codeGraph/stackoverflow_questions_per_class_func_3M_filtered_new.json', 'r') as data:
        properJsonObjects = []
        encounteredPosts = set()
        jsonObjects = ijson.items(data, 'results.bindings.item')
        for jsonObject in jsonObjects:
            objectType = jsonObject['class_func_type']['value'].replace('http://purl.org/twc/graph4code/ontology/', '')
            if objectType != 'Class':
                continue
            stackUrl = jsonObject['q']['value']
            if stackUrl in encounteredPosts:
                continue
            else:
                encounteredPosts.add(stackUrl)
            properJsonObjects.append(jsonObject)
        USEList = ['https://tfhub.dev/google/universal-sentence-encoder/4']
        for USE in USEList:
            calculateMRR(properJsonObjects, USE, True)
        calculateMRR(properJsonObjects, 'bert-base-nli-mean-tokens', False)
        modelList = ['bert-base-nli-mean-tokens', 'bert-large-nli-mean-tokens',
        'roberta-base-nli-mean-tokens',
        'roberta-large-nli-mean-tokens', 'distilbert-base-nli-mean-tokens',
        'bert-base-nli-stsb-mean-tokens', 'bert-large-nli-stsb-mean-tokens',
        'roberta-base-nli-stsb-mean-tokens', 'roberta-large-nli-stsb-mean-tokens',
        'distilbert-base-nli-stsb-mean-tokens']
            
def calculateMRR(jsonCollect, model, isUSE):
    embed = None
    transformer = None
    recipRanks = []
    if isUSE:
        embed = hub.load('https://tfhub.dev/google/universal-sentence-encoder/4')
    else:
        transformer = SentenceTransformer(model)

    for jsonObject in jsonCollect:
        stackQuestion = jsonObject['content_wo_code']
        embeddedQuestion = None
        if isUSE:
            embeddedQuestion = embed([stackQuestion])
        else:
            embeddedQuestion = transformer.encode([stackQuestion])
        embeddingVector = embeddedQuestion[0]
        embeddingQuestionArray = np.asarray(embeddingVector,
        dtype = np.float32).reshape(1,-1)
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
                answerEmbed = None
                if isUSE:
                    answerEmbed = embed([answer])
                else:
                    answerEmbed = transformer.encode([answer])
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

        if len(voteOrder) != 1:
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
    beginAnalysis()
