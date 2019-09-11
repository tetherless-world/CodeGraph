#!/usr/bin/env python
# coding: utf-8

# *[CAVEAT: THIS NOTEBOOK TAKES A QUITE SOME TIME TO LOAD DUE TO BOKEH RENDERING THE PLOTS AND IT ALSO TAKES MORE THEN AN HOUR TO RUN - NOTE IF YOU FORK IT]*
# 
# ----
# # Introduction
# 
# This kernel will be an exploration into the target variable and how it is distributed across the structure of the training data to see if any potential information or patterns can be gleaned going forward. Since classical treatments of text data normally comes with the challenges of high dimensionality (using term frequencies or term frequency inverse document frequencies), the plan therefore in this kernel is to visually explore the target variable in some lower dimensional space. 
# 
# We will explore two methods of representing the data in a lower dimensional space, with the first being using the truncated SVD method to linear reduce the dimensions of the term frequency representation of the text data - i.e. a method known as Latent Semantic Analysis (LSA). The second method would be to utilize document embeddings via the Doc2Vec method, which learns a lower dimensional projection for each question. In these lower dimensional spaces, we can finally then utilize the manifold learning method of the t-Distributed Stochastic Neighbor Embedding (t-SNE) technique to further reduce the dimensionality for target variable visualization.
# 
# The kernel is structured as follows:
# 
# 1. Text preprocessing on question text via standard NLP
# 2. T-SNE on LSA feature space: 
# 3. T-SNE on Doc2Vec space: 
# 
# Let's go

# In[ ]:


# Importing the relevant libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer

from nltk.tokenize import word_tokenize, sent_tokenize, TweetTokenizer 
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.corpus import stopwords
from string import punctuation

import re
from functools import reduce

import bokeh.plotting as bp
from bokeh.models import HoverTool, BoxSelectTool
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure, show, output_notebook, reset_output
from bokeh.palettes import d3
import bokeh.models as bmo
from bokeh.io import save, output_file

# init_notebook_mode(connected = True)
# color = sns.color_palette("Set2")
import warnings
warnings.filterwarnings("ignore")

pd.options.mode.chained_assignment = None
pd.options.display.max_columns = 999
pd.options.display.max_rows = 999

# Reading in the train and test datasets and inspecting the first 5 rows of the dataset, we see that each Quora question text comes with a label (the "target" column) consisting of the binary values of either 1 - insincere or 0 - sincere questions.

# In[ ]:


train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
display(train[train.target == 0].head(3))
display(train[train.target == 1].head(3))

# Now, inspecting the class distributions we see that the provided training set and labels are rather imbalanced with about 6% of the data points being insincere, while the remainder 94% being sincere questions.

# In[ ]:


train.target.value_counts()

# For the purposes of speeding up code execution and modeling in the latter stages, I shall re-balance (naively via random sampling) the dataset set by with equal rows of sincere and insincere data points:

# In[ ]:


# Full number of insincere data points
sample_size = 80_810 

# halved the original size as rendering all the data points was causing lag issues
# sample_size = int(sample_size/2)

# Rebalancing the training set
train_rebal = train[train.target == 1].sample(sample_size).append(train[train.target == 0].sample(sample_size)).reset_index()

# ----
# ## 1. Text Processing
# 
# In this section, we arrive at the pre-processing of the question text contained within the training data. The processing applied here are some of the standard NLP steps that one would implement in a text based problem, consisting of:
# 
# * Tokenization
# * Stemming or Lemmatization

# In[ ]:


def remove_stopwords(words):
    """
    Function to remove stopwords from the question text
    """
    stop_words = set(stopwords.words("english"))
    return [word for word in words if word not in stop_words]

def remove_punctuation(text):
    """
    Function to remove punctuation from the question text
    """
    return re.sub(r'[^\w\s]', '', text)

def lemmatize_text(words):
    """
    Function to lemmatize the question text
    """
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(word) for word in words]

def stem_text(words):
    """
    Function to stem the question text
    """
    ps = PorterStemmer()
    return [ps.stem(word) for word in words]

# **Updates - 28/12/18**
# 
# Based on the many effective public LSTM/GRU-variant kernels which employ text pre-processing that cleans numbers, add spaces to punctuation and odd characters as well as expanding out contractions, let us employ these methods as well and see how that affects the T-SNE and Doc2Vec plots.

# In the cell below, I've defined a whole bunch of punctuation, odd characters and contractions so feel free to unhide it should you wish to see the terms explicitly.

# In[ ]:


puncts=['☹', 'Ź', 'Ż', 'ἰ', 'ή', 'Š', '＞', 'ξ','ฉ', 'ั', 'น', 'จ', 'ะ', 'ท', 'ำ', 'ใ', 'ห', '้', 'ด', 'ี', '่', 'ส', 'ุ', 'Π', 'प', 'ऊ', 'Ö', 'خ', 'ب', 'ஜ', 'ோ', 'ட', '「', 'ẽ', '½', '△', 'É', 'ķ', 'ï', '¿', 'ł', '북', '한', '¼', '∆', '≥', '⇒', '¬', '∨', 'č', 'š', '∫', 'ḥ', 'ā', 'ī', 'Ñ', 'à', '▾', 'Ω', '＾', 'ý', 'µ', '?', '!', '.', ',', '"', '#', '$', '%', '\\', "'", '(', ')', '*', '+', '-', '/', ':', ';', '<', '=', '>', '@', '[', ']', '^', '_', '`', '{', '|', '}', '~', '“', '”', '’', 'é', 'á', '′', '…', 'ɾ', '̃', 'ɖ', 'ö', '–', '‘', 'ऋ', 'ॠ', 'ऌ', 'ॡ', 'ò', 'è', 'ù', 'â', 'ğ', 'म', 'ि', 'ल', 'ग', 'ई', 'क', 'े', 'ज', 'ो', 'ठ', 'ं', 'ड', 'Ž', 'ž', 'ó', '®', 'ê', 'ạ', 'ệ', '°', 'ص', 'و', 'ر', 'ü', '²', '₹', 'ú', '√', 'α', '→', 'ū', '—', '£', 'ä', '️', 'ø', '´', '×', 'í', 'ō', 'π', '÷', 'ʿ', '€', 'ñ', 'ç', 'へ', 'の', 'と', 'も', '↑', '∞', 'ʻ', '℅''ι', '•', 'ì', '−', 'л', 'я', 'д', 'ل', 'ك', 'م', 'ق', 'ا', '∈', '∩', '⊆', 'ã', 'अ', 'न', 'ु', 'स', '्', 'व', 'ा', 'र', 'त', '§', '℃', 'θ', '±', '≤', 'उ', 'द', 'य', 'ब', 'ट', '͡', '͜', 'ʖ', '⁴', '™', 'ć', 'ô', 'с', 'п', 'и', 'б', 'о', 'г', '≠', '∂', 'आ', 'ह', 'भ', 'ी', '³', 'च', '...', '⌚', '⟨', '⟩', '∖', '˂', 'ⁿ', '⅔', 'న', 'ీ', 'క', 'ె', 'ం', 'ద', 'ు', 'ా', 'గ', 'ర', 'ి', 'చ', 'র', 'ড়', 'ঢ়', 'સ', 'ં', 'ઘ', 'ર', 'ા', 'જ', '્', 'ય', 'ε', 'ν', 'τ', 'σ', 'ş', 'ś', 'س', 'ت', 'ط', 'ي', 'ع', 'ة', 'د', 'Å', '☺', 'ℇ', '❤', '♨', '✌', 'ﬁ', 'て', '„', 'Ā', 'ត', 'ើ', 'ប', 'ង', '្', 'អ', 'ូ', 'ន', 'ម', 'ា', 'ធ', 'យ', 'វ', 'ី', 'ខ', 'ល', 'ះ', 'ដ', 'រ', 'ក', 'ឃ', 'ញ', 'ឯ', 'ស', 'ំ', 'ព', 'ិ', 'ៃ', 'ទ', 'គ', '¢', 'つ', 'や', 'ค', 'ณ', 'ก', 'ล', 'ง', 'อ', 'ไ', 'ร', 'į', 'ی', 'ю', 'ʌ', 'ʊ', 'י', 'ה', 'ו', 'ד', 'ת', 'ᠠ', 'ᡳ', 'ᠰ', 'ᠨ', 'ᡤ', 'ᡠ', 'ᡵ', 'ṭ', 'ế', 'ध', 'ड़', 'ß', '¸', 'ч',  'ễ', 'ộ', 'फ', 'μ', '⧼', '⧽', 'ম', 'হ', 'া', 'ব', 'ি', 'শ', '্', 'প', 'ত', 'ন', 'য়', 'স', 'চ', 'ছ', 'ে', 'ষ', 'য', '়', 'ট', 'উ', 'থ', 'ক', 'ῥ', 'ζ', 'ὤ', 'Ü', 'Δ', '내', '제', 'ʃ', 'ɸ', 'ợ', 'ĺ', 'º', 'ष', '♭', '़', '✅', '✓', 'ě', '∘', '¨', '″', 'İ', '⃗', '̂', 'æ', 'ɔ', '∑', '¾', 'Я', 'х', 'О', 'з', 'ف', 'ن', 'ḵ', 'Č', 'П', 'ь', 'В', 'Φ', 'ỵ', 'ɦ', 'ʏ', 'ɨ', 'ɛ', 'ʀ', 'ċ', 'օ', 'ʍ', 'ռ', 'ք', 'ʋ', '兰', 'ϵ', 'δ', 'Ľ', 'ɒ', 'î', 'Ἀ', 'χ', 'ῆ', 'ύ', 'ኤ', 'ል', 'ሮ', 'ኢ', 'የ', 'ኝ', 'ን', 'አ', 'ሁ', '≅', 'ϕ', '‑', 'ả', '￼', 'ֿ', 'か', 'く', 'れ', 'ő', '－', 'ș', 'ן', 'Γ', '∪', 'φ', 'ψ', '⊨', 'β', '∠', 'Ó', '«', '»', 'Í', 'க', 'வ', 'ா', 'ம', '≈', '⁰', '⁷', 'ấ', 'ũ', '눈', '치', 'ụ', 'å', '،', '＝', '（', '）', 'ə', 'ਨ', 'ਾ', 'ਮ', 'ੁ', '︠', '︡', 'ɑ', 'ː', 'λ', '∧', '∀', 'Ō', 'ㅜ', 'Ο', 'ς', 'ο', 'η', 'Σ', 'ण']
odd_chars=[ '大','能', '化', '生', '水', '谷', '精', '微', 'ル', 'ー', 'ジ', 'ュ', '支', '那', '¹', 'マ', 'リ', '仲', '直', 'り', 'し', 'た', '主', '席', '血', '⅓', '漢', '髪', '金', '茶', '訓', '読', '黒', 'ř', 'あ', 'わ', 'る', '胡', '南', '수', '능', '广', '电', '总', 'ί', '서', '로', '가', '를', '행', '복', '하', '게', '기', '乡', '故', '爾', '汝', '言', '得', '理', '让', '骂', '野', '比', 'び', '太', '後', '宮', '甄', '嬛', '傳', '做', '莫', '你', '酱', '紫', '甲', '骨', '陳', '宗', '陈', '什', '么', '说', '伊', '藤', '長', 'ﷺ', '僕', 'だ', 'け', 'が', '街', '◦', '火', '团', '表',  '看', '他', '顺', '眼', '中', '華', '民', '國', '許', '自', '東', '儿', '臣', '惶', '恐', 'っ', '木', 'ホ', 'ج', '教', '官', '국', '고', '등', '학', '교', '는', '몇', '시', '간', '업', '니', '本', '語', '上', '手', 'で', 'ね', '台', '湾', '最', '美', '风', '景', 'Î', '≡', '皎', '滢', '杨', '∛', '簡', '訊', '短', '送', '發', 'お', '早', 'う', '朝', 'ش', 'ه', '饭', '乱', '吃', '话', '讲', '男', '女', '授', '受', '亲', '好', '心', '没', '报', '攻', '克', '禮', '儀', '統', '已', '經', '失', '存', '٨', '八', '‛', '字', '：', '别', '高', '兴', '还', '几', '个', '条', '件', '呢', '觀', '《', '》', '記', '宋', '楚', '瑜', '孫', '瀛', '枚', '无', '挑', '剔', '聖', '部', '頭', '合', '約', 'ρ', '油', '腻', '邋', '遢', 'ٌ', 'Ä', '射', '籍', '贯', '老', '常', '谈', '族', '伟', '复', '平', '天', '下', '悠', '堵', '阻', '愛', '过', '会', '俄', '罗', '斯', '茹', '西', '亚', '싱', '관', '없', '어', '나', '이', '키', '夢', '彩', '蛋', '鰹', '節', '狐', '狸', '鳳', '凰', '露', '王', '晓', '菲', '恋', 'に', '落', 'ち', 'ら', 'よ', '悲', '反', '清', '復', '明', '肉', '希', '望', '沒', '公', '病', '配', '信', '開', '始', '日', '商', '品', '発', '売', '分', '子', '创', '意', '梦', '工', '坊', 'ک', 'پ', 'ڤ', '蘭', '花', '羡', '慕', '和', '嫉', '妒', '是', '样', 'ご', 'め', 'な', 'さ', 'い', 'す', 'み', 'ま', 'せ', 'ん', '音', '红', '宝', '书', '封', '柏', '荣', '江', '青', '鸡', '汤', '文', '粵', '拼', '寧', '可', '錯', '殺', '千', '絕', '放', '過', '」', '之', '勢', '请', '国', '知', '识', '产', '权', '局', '標', '點', '符', '號', '新', '年', '快', '乐', '学', '业', '进', '步', '身', '体', '健', '康', '们', '读', '我', '的', '翻', '译', '篇', '章', '欢', '迎', '入', '坑', '有', '毒', '黎', '氏', '玉', '英', '啧', '您', '这', '口', '味', '奇', '特', '也', '就', '罢', '了', '非', '要', '以', '此', '为', '依', '据', '对', '人', '家', '批', '判', '一', '番', '不', '地', '道', '啊', '谢', '六', '佬']
contraction_mapping = {"ain't": "is not", "aren't": "are not","can't": "cannot", "can't've": "cannot have", "'cause": "because", "could've": "could have", "couldn't": "could not", "couldn't've": "could not have","didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hadn't've": "had not have", "hasn't": "has not", "haven't": "have not",  "he'd": "he would", "he'd've": "he would have", "he'll": "he will", "he'll've": "he will have", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",  "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will", "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not","sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as", "this's": "this is","that'd": "that would", "that'd've": "that would have","that's": "that is", "there'd": "there would", "there'd've": "there would have","there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are", "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have","you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have" } 

# In[ ]:


def clean_numbers(x):
    x = re.sub('[0-9]{5,}', ' ##### ', x)
    x = re.sub('[0-9]{4}', ' #### ', x)
    x = re.sub('[0-9]{3}', ' ### ', x)
    x = re.sub('[0-9]{2}', ' ## ', x)
    return x

def punct_add_space(x):
    x = str(x)
    for punct in puncts:
        x = x.replace(punct, f' {punct} ')
    return x  

def odd_add_space(x):
    x = str(x)
    for odd in odd_chars:
        x = x.replace(odd, f' {odd} ')
    return x 

def clean_contractions(text, mapping):
    specials = ["’", "‘", "´", "`"]
    for s in specials:
        text = text.replace(s, "'")
    text = ' '.join([mapping[t] if t in mapping else t for t in text.split(" ")])
    return text

# Now applying these functions sequentially to the question text field:

# In[ ]:


train_rebal["question_text"] = train_rebal["question_text"].apply(lambda x: clean_numbers(x))
train_rebal["question_text"] = train_rebal["question_text"].apply(lambda x: punct_add_space(x))
train_rebal["question_text"] = train_rebal["question_text"].apply(lambda x: odd_add_space(x))
train_rebal["question_text"] = train_rebal["question_text"].apply(lambda x: clean_contractions(x, contraction_mapping))

# In[ ]:


# With the new updated processing - let's leave out the removal of stopwords, punctuation and lemmatization functions.

# Tokenizing the text
train_rebal["question_text"] = train_rebal["question_text"].apply(lambda x: word_tokenize(x))

# Removing stopwords - Leaving Stopwords
# train_rebal["question_text"] = train_rebal["question_text"].apply(lambda x: remove_stopwords(x))

# Lemmatizting
# train_rebal["question_text"] = train_rebal["question_text"].apply(lambda x: lemmatize_text(x))

# Peeking into the newly processed training set, we can see that the resulting text in the "question text" columns are now nicely tokenized and lemmatized with all stopwords removed.

# In[ ]:


train_rebal.head(3)

# ----
# ## 2. T-SNE applied to Latent Semantic (LSA) space
# 
# To start off we look at the sparse representation of text documents via the Term frequency Inverse document frequency method. What this does is create a matrix representation that upweights locally prevalent but globally rare terms - therefore accounting for the occurence bias when using just term frequencies.

# **Tf-idf space**

# In[ ]:


tf_idf_vec = TfidfVectorizer(min_df=3,
                             max_features = 60_000, #100_000,
                             analyzer="word",
                             ngram_range=(1,3), # (1,6)
                             stop_words="english")
tf_idf = tf_idf_vec.fit_transform(list(train_rebal["question_text"].map(lambda tokens: " ".join(tokens))))

# Having obtained our tf-idf matrix - a sparse matrix object, we now apply the TruncatedSVD method to first reduce the dimensionality of the Tf-idf matrix to a decomposed feature space, referred to in the community as the LSA (Latent Semantic Analysis) method.
# 
# LSA has been one of the classical methods in text that have existed for a while allowing "concept" searching of words whereby words which are semantically similar to each other (i.e. have more context) are closer to each other in this space and vice-versa.

# In[ ]:


# Applying the Singular value decomposition
from sklearn.decomposition import TruncatedSVD
svd = TruncatedSVD(n_components=50, random_state=2018)
svd_tfidf = svd.fit_transform(tf_idf)
print("Dimensionality of LSA space: {}".format(svd_tfidf.shape))

# Quickly plotting a scatter plot of the first 3 dimensions of the latent semantic space just to get an initial feel for how the target variables are distributed:

# In[ ]:


# Showing scatter plots 
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(16,12))

# Plot models:
ax = Axes3D(fig) 
ax.scatter(svd_tfidf[:,0],
           svd_tfidf[:,1],
           svd_tfidf[:,2],
           c=train_rebal.target.values,
           cmap=plt.cm.winter_r,
           s=2,
           edgecolor='none',
           marker='o')
plt.title("Semantic Tf-Idf-SVD reduced plot of Sincere-Insincere data distribution")
plt.xlabel("First dimension")
plt.ylabel("Second dimension")
plt.legend()
plt.xlim(0.0, 0.20)
plt.ylim(-0.2,0.4)
plt.show()

# **Takeaways from the plot**
# 
# From the above scatter plots, It is apparent that sincere and insincere question data points overlap quite significantly in the LSA semantic space. The data points visually appear to be evenly distributed and there does not seem to be any clear or obvious pattern in segregating the class labels. However, do keep in mind that this involves only the first three dimensions of the decomposed space whilst we haven't incorporated information from the other 48 dimensions yet and also the fact that the SVD is a linear decomposition technique.

# Perhaps a non-linear technique (T-SNE) yield more insights? This also helps to collapse the information from all 50 dimensions when we apply the T-SNE technique to this LSA reduced space . Here I've used the multicore implementation of T-SNE to speed things up instead of the plain vanilla sklearn version. 

# In[ ]:


# from sklearn.manifold import TSNE

# Importing multicore version of TSNE
from MulticoreTSNE import MulticoreTSNE as TSNE

# In[ ]:


tsne_model = TSNE(n_jobs=4,
                  early_exaggeration=4, # Trying out exaggeration trick
                  n_components=2,
                  verbose=1,
                  random_state=2018,
                  n_iter=500)
tsne_tfidf = tsne_model.fit_transform(svd_tfidf)

# ### Visualization of target variable via Bokeh
# 
# Turning to the target variable visualization in T-SNE reduced concept space, we will use the plotting library Bokeh to suit our purposes:

# In[ ]:


# Putting the tsne information into a dataframe
tsne_tfidf_df = pd.DataFrame(data=tsne_tfidf, columns=["x", "y"])
tsne_tfidf_df["qid"] = train_rebal["qid"].values
tsne_tfidf_df["question_text"] = train_rebal["question_text"].values
tsne_tfidf_df["target"] = train_rebal["target"].values

# In[ ]:


output_notebook()
plot_tfidf = bp.figure(plot_width = 800, plot_height = 700, 
                       title = "T-SNE applied to Tfidf_SVD space",
                       tools = "pan, wheel_zoom, box_zoom, reset, hover, previewsave",
                       x_axis_type = None, y_axis_type = None, min_border = 1)

# colormap = np.array(["#6d8dca", "#d07d3c"])
colormap = np.array(["darkblue", "red"])

# palette = d3["Category10"][len(tsne_tfidf_df["asset_name"].unique())]
source = ColumnDataSource(data = dict(x = tsne_tfidf_df["x"], 
                                      y = tsne_tfidf_df["y"],
                                      color = colormap[tsne_tfidf_df["target"]],
                                      question_text = tsne_tfidf_df["question_text"],
                                      qid = tsne_tfidf_df["qid"],
                                      target = tsne_tfidf_df["target"]))

plot_tfidf.scatter(x = "x", 
                   y = "y", 
                   color="color",
                   legend = "target",
                   source = source,
                   alpha = 0.7)
hover = plot_tfidf.select(dict(type = HoverTool))
hover.tooltips = {"qid": "@qid", 
                  "question_text": "@question_text", 
                  "target":"@target"}

show(plot_tfidf)

# **Takeaways from the plot**
# 
# - It seems that the distribution of the sincere to insincere data points overlap quite substantially in certain regions of the T-SNE plots in concept space, which unfortunately does not allow easy visual discernment between the two classes (even using this non-linear method).
# - This begs the question of how easy therefore, is it in terms of semantic meaning (to a human) to distinguish between an insincere and a sincere question, when we see data from both class labels overlapping quite heavily across each other. Quite a few of the insincere and sincere questions when read aloud do share quite a lot of similarities as well.
# - There is also a popular thread going on in the Discussions/Forums on how the Sincere and Insincere labels were generated - potentially offering the argument that some of the classes could even have been wrongly applied.
# - There does appear to be one region in the T-SNE plot where the sincere class labels visually seem to be quite separated and exist in its own area of the non-linear semantic space. 
# 
# One key point to note in T-SNE plots is that cluster sizes as well as distances from one cluster to another are not meaningful to the overall global geometry of the picture. This can be evinced from the variation in geometry when we for example change the perplexity value (defaulted to 30) to range from 5 all the way to 50 and note how this in turn affects the geometry of the manifold :
# 

# In[ ]:


# Perplexity = 5
tsne_model_5 = TSNE(n_jobs=4, 
                    early_exaggeration=4,
                  perplexity=5,
                  n_components=2,
                  verbose=1,
                  random_state=2018,
                  n_iter=500)
tsne_tfidf_5 = tsne_model_5.fit_transform(svd_tfidf[:50_000,:])
# Creating a Dataframe for Perplexity=5
tsne_tfidf_df_5 = pd.DataFrame(data=tsne_tfidf_5, columns=["x5", "y5"])
tsne_tfidf_df_5["target"] = train_rebal["target"][:50_000].values

# In[ ]:


# Perplexity = 25
tsne_model_25 = TSNE(n_jobs=4, 
                     early_exaggeration=4,
                  perplexity=25,
                  n_components=2,
                  verbose=1,
                  random_state=2018,
                  n_iter=500)
tsne_tfidf_25 = tsne_model_25.fit_transform(svd_tfidf[:50_000,:])
# Creating a Dataframe for Perplexity=5
tsne_tfidf_df_25 = pd.DataFrame(data=tsne_tfidf_25, 
                             columns=["x25", "y25"])
tsne_tfidf_df_25["target"] = train_rebal["target"][:50_000].values

# In[ ]:


# Perplexity = 50
tsne_model_50 = TSNE(n_jobs=4, 
                     early_exaggeration=4,
                  perplexity=200,
                  n_components=2,
                  verbose=1,
                  random_state=2018,
                  n_iter=500)
tsne_tfidf_50 = tsne_model_50.fit_transform(svd_tfidf[:50_000,:])
# Creating a Dataframe for Perplexity=50
tsne_tfidf_df_50 = pd.DataFrame(data=tsne_tfidf_50, 
                                columns=["x50", "y50"])
tsne_tfidf_df_50["target"] = train_rebal["target"][:50_000].values

# In[ ]:


# Showing scatter plots 
plt.figure(figsize=(14,8))
plt.scatter(tsne_tfidf_df_5.x5, 
            tsne_tfidf_df_5.y5, 
            alpha=0.75,
            c=tsne_tfidf_df_5.target,
            cmap=plt.cm.coolwarm)
plt.title("T-SNE plot in SVD space (perplexity=5)")
plt.legend()
plt.show()

# In[ ]:


plt.figure(figsize=(14,8))
plt.scatter(tsne_tfidf_df_25.x25, 
            tsne_tfidf_df_25.y25, 
            c=tsne_tfidf_df_25.target,
            cmap=plt.cm.coolwarm)
plt.title("T-SNE plot in SVD space (perplexity=25)")
plt.legend()
plt.show()

# In[ ]:


plt.figure(figsize=(14,8))
plt.scatter(tsne_tfidf_df_50.x50, 
            tsne_tfidf_df_50.y50, 
            c=tsne_tfidf_df_50.target,
            cmap=plt.cm.coolwarm)
plt.title("T-SNE plot in SVD space (perplexity=50)")
plt.legend()
plt.show()

# ----
# ## 3. T-SNE applied on Doc2Vec embedding
# 
# Pushing forward with our T-SNE visual explorations, we next move away from semantic matrices into the realm of embeddings. Here we will use the Doc2Vec algorithm and much like its very well known counterpart Word2vec involves unsupervised learning of continuous representations for text. Unlike Word2vec which involves finding the representations for words (i.e. word embeddings), Doc2vec modifies the former method and extends it to  sentences and even documents.
# 
# For this notebook, we will be using gensim's Doc2Vec class which inherits from the base Word2Vec class where style of usage and parameters are similar. The only differences lie in the naming terminology of the training method used which are the “distributed memory” or “distributed bag of words” methods.
# 
# According to the Gensim documentation, Doc2Vec requires the input to be an iterable object representing the sentences in the form of two lists, a list of the terms and a list of labels.

# In[ ]:


from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

# In[ ]:


# Storing the question texts in a list
quora_texts = list(train_rebal["question_text"])

# Creating a list of terms and a list of labels to go with it
documents = [TaggedDocument(doc, tags=[str(i)]) for i, doc in enumerate(quora_texts)]

# Finally we can implement the Doc2Vec model as follows

# In[ ]:


max_epochs = 100
alpha=0.025
model = Doc2Vec(documents,
                size=10, 
                min_alpha=0.00025,
                alpha=alpha,
                min_count=1,
#                 window=2, 
                workers=4)

# In[ ]:


# model.build_vocab(documents)

# for epoch in range(max_epochs):
#     print('iteration {0}'.format(epoch))
#     model.train(documents,
#                 total_examples=model.corpus_count,
#                 epochs=model.iter)
#     # decrease the learning rate
#     model.alpha -= 0.0002
#     # fix the learning rate, no decay
#     model.min_alpha = model.alpha

# Fitting a T-SNE model to the dense embeddings and overlaying that with the target visuals, we get:

# In[ ]:


# Creating and fitting the tsne model to the document embeddings
tsne_model = TSNE(n_jobs=4,
                  early_exaggeration=4,
                  n_components=2,
                  verbose=1,
                  random_state=2018,
                  n_iter=300)
tsne_d2v = tsne_model.fit_transform(model.docvecs.vectors_docs)

# Putting the tsne information into sq
tsne_d2v_df = pd.DataFrame(data=tsne_d2v, columns=["x", "y"])
# tsne_tfidf_df.columns = ["x", "y"]
tsne_d2v_df["qid"] = train_rebal["qid"].values
tsne_d2v_df["question_text"] = train_rebal["question_text"].values
tsne_d2v_df["target"] = train_rebal["target"].values

# In[ ]:


output_notebook()
plot_d2v = bp.figure(plot_width = 800, plot_height = 700, 
                       title = "T-SNE applied to Doc2vec document embeddings",
                       tools = "pan, wheel_zoom, box_zoom, reset, hover, previewsave",
                       x_axis_type = None, y_axis_type = None, min_border = 1)

# colormap = np.array(["#6d8dca", "#d07d3c"])
colormap = np.array(["darkblue", "cyan"])

# palette = d3["Category10"][len(tsne_tfidf_df["asset_name"].unique())]
source = ColumnDataSource(data = dict(x = tsne_d2v_df["x"], 
                                      y = tsne_d2v_df["y"],
                                      color = colormap[tsne_d2v_df["target"]],
                                      question_text = tsne_d2v_df["question_text"],
                                      qid = tsne_d2v_df["qid"],
                                      target = tsne_d2v_df["target"]))

plot_d2v.scatter(x = "x", 
                   y = "y", 
                   color="color",
                   legend = "target",
                   source = source,
                   alpha = 0.7)
hover = plot_d2v.select(dict(type = HoverTool))
hover.tooltips = {"qid": "@qid", 
                  "question_text": "@question_text", 
                  "target":"@target"}

show(plot_d2v)

# **Takeaways from the plot**
# 
# The visual overlap between Sincere and Insincere labelled questions are even greater in the Doc2Vec plots - so much so that there doesn't seem to be any obvious manner to segragate the labels via eye-balling if going down the route of document embeddings.

# In[ ]:



