import ijson
import tensorflow_hub as hub
import numpy as np
from bs4 import BeautifulSoup
import sys
import re
import math
from sentence_transformers import SentenceTransformer
import random
import statistics
from scipy import stats as stat
import pickle
import pandas
import lenskit.topn as metrics

# parse input data file and remove duplicates for analysis
# also calls all necessary analysis functions

def fetchEmbeddingDict(fileName, model):
    if model == 'https://tfhub.dev/google/universal-sentence-encoder/4':
        fullFile = '../../data/codeGraph/fullUSE/stackoverflow_embeddings/' + str(fileName)
    elif model == 'bert-base-nli-mean-tokens':
        fullFile = '../../data/codeGraph/fullBERT/stackoverflow_embeddings_bert2/' + str(fileName)
    else:
        fullFile = '../../data/codeGraph/fullBERT/stackoverflow_embeddings_roberta/' + str(fileName)
    try:
        openFile = open(fullFile, 'rb')
    except FileNotFoundError as e:
        return None
    embeddingDict = pickle.load(openFile)
    openFile.close()
    return embeddingDict

def beginAnalysis():
    with open('../../data/codeGraph/stackoverflow_questions_with_answers_1000000.json', 'r') as data:
        properJsonObjects = []
        encounteredPosts = set()
        jsonObjects = ijson.items(data, 'results.bindings.item')
        i = 0
        for jsonObject in jsonObjects:
            objectType = "Class"
            try:
                objectType = jsonObject['class_func_type']['value'].replace('http://purl.org/twc/graph4code/ontology/', '')
            except KeyError as e:
                pass
            if objectType != 'Class':
                continue
            stackUrl = jsonObject['q']['value']
            if stackUrl in encounteredPosts:
                continue
            else:
                encounteredPosts.add(stackUrl)
            properJsonObjects.append(jsonObject)
            i += 1


        USEList = ['https://tfhub.dev/google/universal-sentence-encoder/4']
        for USE in USEList:
            print("Calculating MRR with model", USE)
            print("Calculating MRR with model", USE, file=sys.stderr)
            calculateMRR(properJsonObjects, USE, True)
            print("Calculating NDCG with model", USE)
            print("Calculating NDCG with model", USE, file=sys.stderr)
            calculateNDCG(properJsonObjects, USE, True)
            print("Calculating T statistic with model", USE)
            print("Calculating T statistic with model", USE, file=sys.stderr)
            calculatePairedTTest(properJsonObjects, USE, True)


        '''modelList = ['bert-base-nli-mean-tokens', 'roberta-base-nli-mean-tokens']
        for model in modelList:
            print("Calculating MRR with model", model)
            print("Calculating MRR with model", model, file=sys.stderr)
            calculateMRR(properJsonObjects, model, False)
            print("Calculating NDCG with model", model)
            print("Calculating NDCG with model", model, file=sys.stderr)
            calculateNDCG(properJsonObjects, model, False)
            print("Calculating T statistic with model", model)
            print("Calculating T statistic with model", model, file=sys.stderr)
            calculatePairedTTest(properJsonObjects, model, False)'''

# function to calculate paired t test for linked posts
def calculatePairedTTest(jsonCollect, model, isUSE):
    embed = None
    transformer = None
    coefficients = []
    rPattern = r'https:\/\/stackoverflow\.com\/questions\/\d+'
    
    if isUSE:
        embed = hub.load('https://tfhub.dev/google/universal-sentence-encoder/4')
    else:
        transformer = SentenceTransformer(model)
    
    urlMapping = {}
    urlList = []

    linkedDists = []
    foreignDists = []

    differences = []

    for jsonObject in jsonCollect:
        qUrl = jsonObject['q']['value']
        urlMapping[qUrl] = jsonObject
        urlList.append(qUrl)

    for jsonObject in jsonCollect:
        qUrl = jsonObject['q']['value']
        urlContent = jsonObject['stackoverflow_urls']
        for potentialUrl in urlContent:
            urlMatch = re.search(rPattern, potentialUrl)
            if urlMatch == None:
                continue
            actualUrl = urlMatch.group(0)
            if actualUrl not in urlMapping or qUrl == actualUrl:
                continue
            post2Object = urlMapping[actualUrl]

            post1Url = qUrl
            post2Url = actualUrl

            adjustedPost1Url = qUrl.replace('https://stackoverflow.com/questions/', '')
            adjustedPost2Url = actualUrl.replace('https://stackoverflow.com/questions/', '')

            post1NewEmbed = fetchEmbeddingDict(adjustedPost1Url, model)
            post2NewEmbed = fetchEmbeddingDict(adjustedPost2Url, model)

            if post1NewEmbed == None or post2NewEmbed == None:
                continue

            '''post1Question = jsonObject['content_wo_code']
            post2Question = post2Object['content_wo_code']

            if isUSE:
                post1Embedding = embed([post1Question])
                post2Embedding = embed([post2Question])
            else:
                post1Embedding = transformer.encode([post1Question])
                post2Embedding = transformer.encode([post2Question])

            post1EmbeddingVector = post1Embedding[0]
            post2EmbeddingVector = post2Embedding[0]'''

            post1EmbeddingArray = post1NewEmbed['content'].numpy()[0]
            post2EmbeddingArray = post2NewEmbed['content'].numpy()[0]
            
            linkedDist = np.linalg.norm(post1EmbeddingArray - post2EmbeddingArray)**2
            if linkedDist <= .001:
                continue

            post3Url = random.choice(urlList)
            post4Url = random.choice(urlList)

            while post3Url == post1Url or post3Url == post2Url:
                post3Url = random.choice(urlList)

            while post4Url == post2Url or post4Url == post1Url:
                post4Url = random.choice(urlList)

            adjustedPost3Url = post3Url.replace('https://stackoverflow.com/questions/', '')
            adjustedPost4Url = post4Url.replace('https://stackoverflow.com/questions/', '')

            post3NewEmbed = fetchEmbeddingDict(adjustedPost3Url, model)
            post4NewEmbed = fetchEmbeddingDict(adjustedPost4Url, model)

            if post3NewEmbed == None or post4NewEmbed == None:
                continue

            '''post3Object = urlMapping[post3Url]
            post4Object = urlMapping[post4Url]

            post3Question = post3Object['content_wo_code']
            post4Question = post4Object['content_wo_code']

            if isUSE:
                post3Embedding = embed([post3Question])
                post4Embedding = embed([post4Question])
            else:
                post3Embedding = transformer.encode([post3Question])
                post4Embedding = transformer.encode([post4Question])

            post3EmbeddingVector = post3Embedding[0]
            post4EmbeddingVector = post4Embedding[0]'''

            post3EmbeddingArray = post3NewEmbed['content'].numpy()[0]
            post4EmbeddingArray = post4NewEmbed['content'].numpy()[0]

            post1And3Dist = np.linalg.norm(post1EmbeddingArray - post3EmbeddingArray)**2
            post2And4Dist = np.linalg.norm(post2EmbeddingArray - post4EmbeddingArray)**2

            foreignDistAverage = (post1And3Dist + post2And4Dist)/2
            
            linkedDists.append(linkedDist)
            foreignDists.append(foreignDistAverage)

            difference = foreignDistAverage - linkedDist

            differences.append(difference)

    results = stat.ttest_rel(foreignDists, linkedDists)
    print('Result of T statistic calculation is:', results)


# function to calculate NDCG for question and answer rankings
def calculateNDCG(jsonCollect, model, isUSE):
    embed = None
    transformer = None
    coefficients = []
    if isUSE:
        embed = hub.load('https://tfhub.dev/google/universal-sentence-encoder/4')
    else:
        transformer = SentenceTransformer(model)

    for jsonObject in jsonCollect:
        stackQuestion = jsonObject['content_wo_code']
        stackId = jsonObject['q']['value'].replace('https://stackoverflow.com/questions/', '')
        newEmbed = fetchEmbeddingDict(stackId, model)
        if newEmbed == None:
            continue
        '''if isUSE:
            embeddedQuestion = embed([stackQuestion])
        else:
            embeddedQuestion = transformer.encode([stackQuestion])'''
        embeddingQuestionArray = newEmbed['content'].numpy()[0]
        voteOrder = []
        distanceOrder = []
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
                '''if isUSE:
                    answerEmbed = embed([answer])
                else:
                    answerEmbed = transformer.encode([answer])'''
                answerArray = newEmbed[index].numpy()[0]
                dist = np.linalg.norm(answerArray - embeddingQuestionArray)**2
                voteOrder.append((answerVotes, answer))
                distanceOrder.append((dist, answer))
                voteMap[answer] = answerVotes
                i += 1
            except KeyError as e:
                valid = False
        if not voteOrder:
            continue
        voteOrder.sort()
        voteOrder.reverse()
        distanceOrder.sort()
        if len(voteOrder) != 1:
            i = 1
            workingDCG = 0
            for distanceAnswer in distanceOrder:
                rel = voteMap[distanceAnswer[1]]
                normal = math.log2(i+1)
                totalAdd = rel/normal
                workingDCG += totalAdd
                i += 1
            i = 1
            workingIDCG = 0
            for voteAnswer in voteOrder:
                rel = voteAnswer[0]
                normal = math.log2(i+1)
                totalAdd = rel/normal
                workingIDCG += totalAdd
                i += 1
            nDCG = workingDCG/workingIDCG
            coefficients.append(nDCG)
    fullNDCG = sum(coefficients)/len(coefficients)
    print("Average NDCG:", fullNDCG)


# function to calculate MRR for question and answer rankings            
def calculateMRR(jsonCollect, model, isUSE):
    embed = None
    transformer = None
    recipRanks = []
    if isUSE:
        pass
        #embed = hub.load('https://tfhub.dev/google/universal-sentence-encoder/4')
    else:
        transformer = SentenceTransformer(model)

    for jsonObject in jsonCollect:
        stackQuestion = jsonObject['content_wo_code']
        stackId = jsonObject['q']['value'].replace('https://stackoverflow.com/questions/', '')
        newEmbed = fetchEmbeddingDict(stackId, model)
        if newEmbed == None:
            continue
        #template is newEmbed['(answer/content)'].numpy()[0]
        '''embeddedQuestion = None
        if isUSE:
            embeddedQuestion = embed([stackQuestion])
        else:
            embeddedQuestion = transformer.encode([stackQuestion])
        embeddingVector = embeddedQuestion[0]'''
        embeddingQuestionArray = newEmbed['content'].numpy()[0]
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
                '''soup = BeautifulSoup(answer, 'html.parser')
                for code in soup.find_all('code'):
                    code.decompose()
                answer = soup.get_text()
                answerEmbed = None
                if isUSE:
                    answerEmbed = embed([answer])
                else:
                    answerEmbed = transformer.encode([answer])
                answerVector = answerEmbed[0]'''
                answerArray = newEmbed[index].numpy()[0]
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

        trueDict = {}
        trueDict['item'] = []
        trueDict['rating'] = []
        trueDict['answer'] = []
        trueDict['grouping'] = []
        recommendDict = {}
        recommendDict['item'] = []
        recommendDict['answer'] = []
        recommendDict['grouping'] = []

        idDict = {}


        t = 0
        for entry in voteOrder:
            idDict[entry[1]] = t
            t += 1

        t = 0
        for entry in voteOrder:
            trueDict['answer'].append(entry[1])
            trueDict['rating'].append(entry[0])
            trueDict['item'].append(idDict[entry[1]])
            trueDict['grouping'].append('entry')
            t += 1

        t = 0
        for entry in distanceOrder:
            recommendDict['answer'].append(entry[1])
            recommendDict['item'].append(idDict[entry[1]])
            recommendDict['grouping'].append('entry')
            t += 1

        if len(voteOrder) != 1:
            truthFrame = pandas.DataFrame(data=trueDict)
            recommendFrame = pandas.DataFrame(data=recommendDict)
            print(truthFrame)
            print(recommendFrame)

            analyze = metrics.RecListAnalysis(group_cols=['grouping'])
            analyze.add_metric(metrics.recip_rank)
            analyze.add_metric(metrics.ndcg)

            result = analyze.compute(recommendFrame, truthFrame)

            print(result)

            input()

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