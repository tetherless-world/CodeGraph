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

# parse input data file and remove duplicates for analysis
# also calls all necessary analysis functions

def fetchEmbeddingDict(fileName, model):
    # adjustments must be made here if paths to data need to be adjusted
    # or new sources of data are to be used. This function is called in
    # all of the analysis functions
    if model == 'https://tfhub.dev/google/universal-sentence-encoder/4':
        fullFile = '../../../data/codeGraph/fullUSE/stackoverflow_embeddings/' + str(fileName)
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

def beginAnalysis(stackQandAPath):
    with open(stackQandAPath, 'r') as data:
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

        # uncomment this if analyses are to be performed on BERT or ROBERTA embeddings
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

        # for other sources of data, add calls like the blocks above with the
        # appropriate model name. Additionally, remember to make the requisite
        # changes to fetchEmbeddingDict()

# function to calculate paired t test for linked posts
def calculatePairedTTest(jsonCollect, model, isUSE):
    random.seed(116)
    initialRand = random.getstate()
    embed = None
    transformer = None
    coefficients = []
    rPattern = r'https:\/\/stackoverflow\.com\/questions\/\d+'

  
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

            if isUSE:
                post1EmbeddingArray = post1NewEmbed['content'].numpy()[0]
                post2EmbeddingArray = post2NewEmbed['content'].numpy()[0]
            else:
                post1EmbeddingArray = post1NewEmbed['content']
                post2EmbeddingArray = post2NewEmbed['content']
            
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

            if isUSE:
                post3EmbeddingArray = post3NewEmbed['content'].numpy()[0]
                post4EmbeddingArray = post4NewEmbed['content'].numpy()[0]
            else:
                post3EmbeddingArray = post3NewEmbed['content']
                post4EmbeddingArray = post4NewEmbed['content']

            post1And3Dist = np.linalg.norm(post1EmbeddingArray - post3EmbeddingArray)**2
            post2And4Dist = np.linalg.norm(post2EmbeddingArray - post4EmbeddingArray)**2

            foreignDistAverage = (post1And3Dist + post2And4Dist)/2
            
            linkedDists.append(linkedDist)
            foreignDists.append(foreignDistAverage)

            difference = foreignDistAverage - linkedDist

            differences.append(difference)

    results = stat.ttest_rel(foreignDists, linkedDists)
    random.setstate(initialRand)
    print('Result of T statistic calculation is:', results)


# function to calculate NDCG for question and answer rankings
def calculateNDCG(jsonCollect, model, isUSE):
    embed = None
    transformer = None
    coefficients = []

    for jsonObject in jsonCollect:
        stackQuestion = jsonObject['content_wo_code']
        stackId = jsonObject['q']['value'].replace('https://stackoverflow.com/questions/', '')
        newEmbed = fetchEmbeddingDict(stackId, model)
        if newEmbed == None:
            continue
        if isUSE:
            embeddingQuestionArray = newEmbed['content'].numpy()[0]
        else:
            embeddingQuestionArray = newEmbed['content']
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
                if isUSE:
                    answerArray = newEmbed[index].numpy()[0]
                else:
                    answerArray = newEmbed[index]
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

    for jsonObject in jsonCollect:
        stackQuestion = jsonObject['content_wo_code']
        stackId = jsonObject['q']['value'].replace('https://stackoverflow.com/questions/', '')
        newEmbed = fetchEmbeddingDict(stackId, model)
        if newEmbed == None:
            continue
        if isUSE:
            embeddingQuestionArray = newEmbed['content'].numpy()[0]
        else:
            embeddingQuestionArray = newEmbed['content']
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
                if isUSE:
                    answerArray = newEmbed[index].numpy()[0]
                else:
                    answerArray = newEmbed[index]
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
    stackQandAPath = input("Please enter path to stackoverflow question and answer data")
    beginAnalysis(stackQandAPath)
