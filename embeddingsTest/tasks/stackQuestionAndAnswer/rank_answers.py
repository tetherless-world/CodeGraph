import json
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
import os
from pathlib import Path
from scipy.spatial import distance

sample = False
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
    with open(stackQandAPath, 'r', encoding="UTF-8") as data_file:
        properJsonObjects = []
        encounteredPosts = set()
        # jsonObjects = ijson.items(data, 'item')
        data = json.load(data_file)
        i = 0
        for jsonObject in data:
            stackUrl = jsonObject['q_url']
            if stackUrl in encounteredPosts:
                continue
            else:
                encounteredPosts.add(stackUrl)
            properJsonObjects.append(jsonObject)
            adjustedJsonObjects.append(jsonObject)
            i += 1

        folder_name = '/tmp/stackoverflow_embed_'+embed_type
        if not os.path.isdir(folder_name):
            os.makedirs(folder_name)
        print("Calculating MRR with model", embed_type)
        print("Calculating MRR with model", embed_type, file=sys.stderr)
        calculateMRR(properJsonObjects, model, embed_type, folder_name)
        print("Calculating NDCG with model", embed_type)
        print("Calculating NDCG with model", embed_type, file=sys.stderr)
        calculateNDCG(properJsonObjects, model, embed_type, folder_name)
        print("Calculating T statistic with model", embed_type)
        print("Calculating T statistic with model", embed_type, file=sys.stderr)
        calculatePairedTTest(adjustedJsonObjects, model, embed_type)


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
def calculatePairedTTest(jsonCollect, model, embed_type):
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
        qUrl = jsonObject['q_url']
        urlMapping[qUrl] = jsonObject
        urlList.append(qUrl)
    number_posts_with_stackOverflow_links = 0
    num_stackOverflow_links = []
    for idx, jsonObject in enumerate(jsonCollect):
        if idx % 1000 == 0:
            print(f'calculatePairedTTest: finished {idx} out of {len(jsonCollect)}')
        qUrl = jsonObject['q_url']
        all_content = jsonObject['q_text']
        answerCollection = jsonObject['answers']
        for answer in answerCollection:
            answerText = answer['a_text']
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

            post1EmbeddingArray = embed_sentences(jsonObject['q_text'], model, embed_type)
            post2EmbeddingArray = embed_sentences(post2Object["q_text"], model, embed_type)

            linkedDist = np.linalg.norm(post1EmbeddingArray - post2EmbeddingArray)**2
            if linkedDist <= .001:
                continue

            post3Url = random.choice(urlList)
            post4Url = random.choice(urlList)

            while post3Url == post1Url or post3Url == post2Url:
                post3Url = random.choice(urlList)

            while post4Url == post2Url or post4Url == post1Url:
                post4Url = random.choice(urlList)

            post3EmbeddingArray = embed_sentences(urlMapping[post3Url]["q_text"], model, embed_type)
            post4EmbeddingArray = embed_sentences(urlMapping[post4Url]["q_text"], model, embed_type)

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
def calculateNDCG(jsonCollect, model, embed_type, folder_name):
    embed = None
    transformer = None
    coefficients = []
    if sample:
        jsonCollect = jsonCollect[:100]
    for idx, jsonObject in enumerate(jsonCollect):
        if idx % 1000 == 0:
            print(f'calculateNDCG: finished {idx} out of {len(jsonCollect)}')
        stackId = jsonObject['q_id']
        file_path = folder_name + "/q_" + stackId
        embed_dic = {}
        if Path(file_path).is_file():
            with open(file_path, 'rb') as handle:
                embed_dic = pickle.load(handle)
        # newEmbed = fetchEmbeddingDict(stackId, model, embed_dir)
        # if newEmbed == None:
        #     continue
        # embeddingQuestionArray = newEmbed['content']
        if jsonObject["q_text"] in embed_dic:
            embeddingQuestionArray = embed_dic[jsonObject["q_text"]]
        else:
            embeddingQuestionArray = embed_sentences(jsonObject["q_text"], model, embed_type )
            embed_dic[jsonObject["q_text"]] = embeddingQuestionArray
        voteOrder = []
        distanceOrder = []
        voteMap = {}
        answerCollection = jsonObject['answers']
        for answer in answerCollection:
            answerText = answer['a_text']
            # answerID = answer['id']
            answerVotes = int(answer['a_votes']) if answer['a_votes'] != '' else 0
            # answerArray = newEmbed[answerID]
            if answerText in embed_dic:
                answerArray = embed_dic[answerText]
            else:
                answerArray = embed_sentences(answer["text"], model, embed_type)
                embed_dic[answerText] = answerArray
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
        with open(file_path, 'wb') as handle:
            pickle.dump(embed_dic, handle, protocol=pickle.HIGHEST_PROTOCOL)
    fullNDCG = sum(coefficients)/len(coefficients)
    print('NDCG: standard error of the mean ', stat.sem(coefficients))
    print("Average NDCG:", fullNDCG)

def embed_sentences(sentences, model, embed_type ):
    if embed_type == 'USE':
        sentence_embeddings = model.encode([sentences])
    else:
        sentence_embeddings = model.encode(sentences)
    return sentence_embeddings

# function to calculate MRR for question and answer rankings            
def calculateMRR(jsonCollect, model, embed_type, folder_name):
    embed = None
    transformer = None
    recipRanks = []
    if sample:
        jsonCollect = jsonCollect[:100]
    euclid_distances_to_best_answer = []
    euclid_distances_to_worst_answer = []
    cosine_distances_to_best_answer = []
    cosine_distances_to_worst_answer = []

    for idx, jsonObject in enumerate(jsonCollect):
        if idx % 1000 == 0:
            print(f'calculateMRR: finished {idx} out of {len(jsonCollect)}')
        stackId = jsonObject['q_id']
        file_path = folder_name + "/q_"+stackId
        embed_dic = {}
        if Path(file_path).is_file():
            with open(file_path, 'rb') as handle:
                embed_dic = pickle.load(handle)
        # newEmbed = fetchEmbeddingDict(stackId, model, embed_dir)
        # if newEmbed == None:
        #     continue
        if jsonObject['q_text'] in embed_dic:
            embeddingQuestionArray = embed_dic[jsonObject['q_text']]
        else:
            embeddingQuestionArray = embed_sentences(jsonObject['q_text'], model, embed_type )
            embed_dic[jsonObject['q_text']] = embeddingQuestionArray
        # embeddingQuestionArray = newEmbed['content']
        voteOrder = []
        distanceOrder = []
        valid = True
        answerCollection = jsonObject['answers']
        min_vote = float('inf')
        max_vote = float('-inf')
        min_vote_idx = -1
        max_vote_idx = -1

        for idx, answer in enumerate(answerCollection):
            answerVotes = int(answer['a_votes']) if answer['a_votes'] != '' else None
            if answerVotes:
                if answerVotes < min_vote:
                    min_vote = answerVotes
                    min_vote_idx = idx
                if answerVotes > max_vote:
                    max_vote = answerVotes
                    max_vote_idx = idx
        print(f'Min vote {min_vote} has index {min_vote_idx}, Max vote {max_vote} has index {max_vote_idx}')
        for idx, answer in enumerate(answerCollection):
            answerText = answer['a_text']
            # answerID = answer['id']
            answerVotes = int(answer['a_votes']) if answer['a_votes'] != '' else 0
            # answerArray = newEmbed[answerID]
            if answerText in embed_dic:
                answerArray = embed_dic[answerText]
            else:
                answerArray = embed_sentences(answerText, model, embed_type)
                embed_dic[answerText] = answerArray
            dist = np.linalg.norm(answerArray-embeddingQuestionArray)**2
            cosine_dist = distance.cosine(answerArray, embeddingQuestionArray)
            voteOrder.append((answerVotes, answerText))
            distanceOrder.append((dist, answerText))
            if idx == min_vote_idx:
                euclid_distances_to_worst_answer.append(dist)
                cosine_distances_to_worst_answer.append(cosine_dist)

            elif idx == max_vote_idx:
                euclid_distances_to_best_answer.append(dist)
                cosine_distances_to_best_answer.append(cosine_dist)

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

        # if len(embed_dic)!=0:
        with open(file_path, 'wb') as handle:
            pickle.dump(embed_dic, handle, protocol=pickle.HIGHEST_PROTOCOL)

    meanRecipRank = sum(recipRanks)/len(recipRanks)
    print('MRR: standard error of the mean ', stat.sem(recipRanks))
    print("Mean reciprocal rank is:", meanRecipRank)
    print(f"Average distance from question to best answer (highest votes): euclid = {statistics.mean(euclid_distances_to_best_answer)}, "
          f"cosine = {statistics.mean(cosine_distances_to_best_answer)}")
    print(f"Average distance from question to worst answer (lowest votes):euclid = {statistics.mean(euclid_distances_to_worst_answer)}, "
          f"cosine = {statistics.mean(cosine_distances_to_worst_answer)}")

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
