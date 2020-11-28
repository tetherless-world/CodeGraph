import ijson
import tensorflow_hub as hub
import numpy as np
from bs4 import BeautifulSoup
import sys
import re
import math, scipy
from sentence_transformers import SentenceTransformer
import random
import statistics
from scipy import stats as stat
import pickle
from utils.util import get_model
# parse input data file and remove duplicates for analysis
# also calls all necessary analysis functions

def fetchEmbeddingDict(fileName, model, embed_dir):
    # adjustments must be made here if paths to data need to be adjusted
    # or new sources of data are to be used. This function is called in
    # all of the analysis functions
    # if model == 'https://tfhub.dev/google/universal-sentence-encoder/4':
    #     fullFile = '../../../data/codeGraph/fullUSE/stackoverflow_embeddings/' + str(fileName)
    # elif model == 'bert-base-nli-mean-tokens':
    #     fullFile = '../../data/codeGraph/fullBERT/stackoverflow_embeddings_bert2/' + str(fileName)
    # else:
    #     fullFile = '../../data/codeGraph/fullBERT/stackoverflow_embeddings_roberta/' + str(fileName)

    fullFile = embed_dir + '/' + str(fileName)
    try:
        openFile = open(fullFile, 'rb')
    except FileNotFoundError as e:
        return None
    embeddingDict = pickle.load(openFile)
    openFile.close()
    return embeddingDict

def beginAnalysis(stackQandAPath, model, embed_type):
    adjustedJsonObjects = []
    with open(stackQandAPath, 'r') as data:
        properJsonObjects = []
        encounteredPosts = set()
        jsonObjects = ijson.items(data, 'item')
        i = 0
        for jsonObject in jsonObjects:
            stackUrl = jsonObject['url']
            if stackUrl in encounteredPosts:
                continue
            else:
                encounteredPosts.add(stackUrl)
            properJsonObjects.append(jsonObject)
            adjustedJsonObjects.append(jsonObject)
            i += 1

    # with open(oldDataPath, 'r') as oldData:
    #     adjustedJsonObjects = []
    #     encounteredPosts = set()
    #     jsonObjects = ijson.items(oldData, 'results.bindings.item')
    #     i = 0
    #     for jsonObject in jsonObjects:
    #         stackUrl = jsonObject['q']['value']
    #         if stackUrl in encounteredPosts:
    #             continue
    #         else:
    #             encounteredPosts.add(stackUrl)
    #         adjustedJsonObjects.append(jsonObject)
    #         i += 1

        print("Calculating MRR with model", embed_type)
        print("Calculating MRR with model", embed_type, file=sys.stderr)
        calculateMRR(properJsonObjects, model, embed_type)
        print("Calculating NDCG with model", embed_type)
        print("Calculating NDCG with model", embed_type, file=sys.stderr)
        calculateNDCG(properJsonObjects, model, embed_type)
        print("Calculating T statistic with model", embed_type)
        print("Calculating T statistic with model", embed_type, file=sys.stderr)
        calculatePairedTTest(adjustedJsonObjects, model, embed_type)

        # USEList = ['https://tfhub.dev/google/universal-sentence-encoder/4']
        # for USE in USEList:
        #     print("Calculating MRR with model", USE)
        #     print("Calculating MRR with model", USE, file=sys.stderr)
        #     calculateMRR(properJsonObjects, USE, True)
        #     print("Calculating NDCG with model", USE)
        #     print("Calculating NDCG with model", USE, file=sys.stderr)
        #     calculateNDCG(properJsonObjects, USE, True)
        #     print("Calculating T statistic with model", USE)
        #     print("Calculating T statistic with model", USE, file=sys.stderr)
        #     calculatePairedTTest(adjustedJsonObjects, USE, True)

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
            calculatePairedTTest(adjustedJsonObjects, model, False)'''

        # for other sources of data, add calls like the blocks above with the
        # appropriate model name. Additionally, remember to make the requisite
        # changes to fetchEmbeddingDict()

# function to calculate paired t test for linked posts
def calculatePairedTTest(jsonCollect, model, embed_type, sample = False):
    random.seed(116)
    initialRand = random.getstate()
    embed = None
    transformer = None
    coefficients = []
    rPattern = r'https:\/\/stackoverflow\.com\/questions\/\d+'
    if sample:
        jsonCollect = jsonCollect[:100]
  
    urlMapping = {}
    urlList = []

    linkedDists = []
    foreignDists = []

    differences = []

    for jsonObject in jsonCollect:
        qUrl = jsonObject['url']
        urlMapping[qUrl] = jsonObject
        urlList.append(qUrl)
    number_posts_with_stackOverflow_links = 0
    num_stackOverflow_links = []
    for jsonObject in jsonCollect:
        qUrl = jsonObject['url']
        all_content = jsonObject['text:']
        answerCollection = jsonObject['answers']
        for answer in answerCollection:
            answerText = answer['text']
            all_content += '  ' + answerText
        urls = re.findall('(https://)([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?', all_content)
        filtered_urls = []
        for url_parts in urls:
            url = ''.join(url_parts)
            if 'stackoverflow.com/questions' in url:
                filtered_urls.append(url)
        # q_urls = [url for url in urls if 'https://stackoverflow.com/questions/' in url]
        # urlContent = jsonObject['stackoverflow_urls']
        urlContent = list(filtered_urls)
        if len(filtered_urls) > 0:
            number_posts_with_stackOverflow_links += 1
            num_stackOverflow_links.append(len(filtered_urls))
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

            # adjustedPost1Url = qUrl.replace('https://stackoverflow.com/questions/', '')
            # adjustedPost2Url = actualUrl.replace('https://stackoverflow.com/questions/', '')

            # post1NewEmbed = fetchEmbeddingDict(adjustedPost1Url, model, embed_dir)
            # post2NewEmbed = fetchEmbeddingDict(adjustedPost2Url, model, embed_dir)
            #
            # if post1NewEmbed == None or post2NewEmbed == None:
            #     continue

            # if model == 'https://tfhub.dev/google/universal-sentence-encoder/4':
            #     post1EmbeddingArray = post1NewEmbed['content'] .numpy()[0]
            #     post2EmbeddingArray = post2NewEmbed['content'].numpy()[0]
            # else:
            # post1EmbeddingArray = post1NewEmbed['content']
            # post2EmbeddingArray = post2NewEmbed['content']
            post1EmbeddingArray = embed_sentences(jsonObject["title"] + '  ' + jsonObject['text:'], model, embed_type)
            post2EmbeddingArray = embed_sentences(post2Object["title"] + '  ' + post2Object['text:'], model, embed_type)

            linkedDist = np.linalg.norm(post1EmbeddingArray - post2EmbeddingArray)**2
            if linkedDist <= .001:
                continue

            post3Url = random.choice(urlList)
            post4Url = random.choice(urlList)

            while post3Url == post1Url or post3Url == post2Url:
                post3Url = random.choice(urlList)

            while post4Url == post2Url or post4Url == post1Url:
                post4Url = random.choice(urlList)

            # adjustedPost3Url = post3Url.replace('https://stackoverflow.com/questions/', '')
            # adjustedPost4Url = post4Url.replace('https://stackoverflow.com/questions/', '')
            #
            # post3NewEmbed = fetchEmbeddingDict(adjustedPost3Url, model, embed_dir)
            # post4NewEmbed = fetchEmbeddingDict(adjustedPost4Url, model, embed_dir)
            #
            # if post3NewEmbed == None or post4NewEmbed == None:
            #     continue
            #
            # # if model == 'https://tfhub.dev/google/universal-sentence-encoder/4':
            # #     post3EmbeddingArray = post3NewEmbed['content'].numpy()[0]
            # #     post4EmbeddingArray = post4NewEmbed['content'].numpy()[0]
            # # else:
            # post3EmbeddingArray = post3NewEmbed['content']
            # post4EmbeddingArray = post4NewEmbed['content']

            # post3EmbeddingArray = post3NewEmbed['content']
            # post4EmbeddingArray = post4NewEmbed['content']
            post3EmbeddingArray = embed_sentences(urlMapping[post3Url]["title"] + '  ' + urlMapping[post3Url]['text:'], model, embed_type)
            post4EmbeddingArray = embed_sentences(urlMapping[post4Url]["title"] + '  ' + urlMapping[post4Url]['text:'], model, embed_type)

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
    print('Number of forum posts with stackoverflow links = ', number_posts_with_stackOverflow_links)
    print('Average number of links per post: ', statistics.mean(num_stackOverflow_links))

# function to calculate NDCG for question and answer rankings
def calculateNDCG(jsonCollect, model, embed_type, sample = False):
    embed = None
    transformer = None
    coefficients = []
    if sample:
        jsonCollect = jsonCollect[:100]
    for jsonObject in jsonCollect:
        stackId = jsonObject['id:']
        # newEmbed = fetchEmbeddingDict(stackId, model, embed_dir)
        # if newEmbed == None:
        #     continue
        # embeddingQuestionArray = newEmbed['content']
        embeddingQuestionArray = embed_sentences(jsonObject["title"] +'  '+ jsonObject['text:'], model, embed_type )
        voteOrder = []
        distanceOrder = []
        voteMap = {}
        answerCollection = jsonObject['answers']
        for answer in answerCollection:
            answerText = answer['text']
            answerID = answer['id']
            answerVotes = int(answer['votes']) if answer['votes'] != '' else 0
            # answerArray = newEmbed[answerID]
            answerArray = embed_sentences(answer["text"], model, embed_type)
            dist = np.linalg.norm(answerArray-embeddingQuestionArray)**2
            voteOrder.append((answerVotes, answerText))
            distanceOrder.append((dist, answerText))
            voteMap[answerText] = answerVotes

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
            if workingIDCG != 0:
                nDCG = workingDCG/workingIDCG
                coefficients.append(nDCG)
    fullNDCG = sum(coefficients)/len(coefficients)
    print('NDCG: standard error of the mean ', stat.sem(coefficients))
    print("Average NDCG:", fullNDCG)

def embed_sentences(sentences, model, embed_type ):
    if embed_type == 'USE':
        sentence_embeddings = model(sentences)
    else:
        sentence_embeddings = model.encode(sentences)
    return sentence_embeddings

# function to calculate MRR for question and answer rankings            
def calculateMRR(jsonCollect, model, embed_type, sample=False):
    embed = None
    transformer = None
    recipRanks = []
    if sample:
        jsonCollect = jsonCollect[:100]
    for jsonObject in jsonCollect:
        stackId = jsonObject['id:']
        # newEmbed = fetchEmbeddingDict(stackId, model, embed_dir)
        # if newEmbed == None:
        #     continue
        embeddingQuestionArray = embed_sentences(jsonObject["title"] +'  '+ jsonObject['text:'], model, embed_type )
        # embeddingQuestionArray = newEmbed['content']
        voteOrder = []
        distanceOrder = []
        valid = True
        answerCollection = jsonObject['answers']
        for answer in answerCollection:
            answerText = answer['text']
            answerID = answer['id']
            answerVotes = int(answer['votes']) if answer['votes'] != '' else 0
            # answerArray = newEmbed[answerID]
            answerArray = embed_sentences(answer["text"], model, embed_type)
            dist = np.linalg.norm(answerArray-embeddingQuestionArray)**2
            voteOrder.append((answerVotes, answerText))
            distanceOrder.append((dist, answerText))
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
    print('MRR: standard error of the mean ', stat.sem(recipRanks))
    print("Mean reciprocal rank is:", meanRecipRank)

if __name__ == "__main__":
    # stackQandAPath = input("Please enter path to stackoverflow question and answer data")
    # oldPath = input("Please enter path to legacy data for paired t test calculation.")
    # stackQandAPath = '/Users/ibrahimabdelaziz/Downloads/stackoverflow_data_ranking_sample.json'
    # model =  ''
    # embed_dir = ''
    stackQandAPath = sys.argv[1]
    embed_type = sys.argv[2]
    model_path = sys.argv[3]
    model = get_model(embed_type, model_path)
    beginAnalysis(stackQandAPath, model, embed_type)
