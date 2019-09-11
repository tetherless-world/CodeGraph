#!/usr/bin/env python
# coding: utf-8

# 

# this is folked kernel with japanese comment for education
# 
# # ホーム・クレジット社 債務不履行リスク - データ探索 + 基本モデル
# 
# 融資履歴が少なかったり無かったりするために、多くの人々が融資を受けるのに苦労しています。  
# そして、残念なことに、このような人々は、怪しげな金貸し屋によってしばしばカモにされます。  
# ホームクレジット社は、ポジティブで安全な借入経験を提供することによって、  
# 銀行口座を持たない人々のためのファイナンシャル・インクルージョン (貧困層に正規の金融取引ができるように改善する解決策を提供すること) を広めるために努力しています。  
# この金銭的に不利な人々がポジティブな借入経験を持つことを確実にするために、ホームクレジットは電話や取引情報を含むさまざまな代替データを利用しています。
# そして顧客の返済能力を予測しています。
# 
# ホームクレジット社は現在、これらの予測を行うためにさまざまな統計的方法や機械学習方法を使用していますが、  
# ホームクレジット社は社の持つデータの潜在能力を最大限に発揮するためにKagglersに挑戦を挑みました。  
# このコンペにより、返済能力のある顧客が無事借入できること、そして顧客がより確実に返済完了できるような借入額、完済日、返済スケジュールを提供することが可能となるでしょう。
# 
# これはホーム・クレジット社債務不履行データについてのデータ探索と基本モデルについての簡単なノートブックです。  
# **Contents**   
# 1. Dataset Preparation    
# 2. Exploration - Applications Train  
# &nbsp;&nbsp;&nbsp;&nbsp; 2.1 Snapshot - Application Train    
# &nbsp;&nbsp;&nbsp;&nbsp; 2.2 Distribution of Target Variable    
# &nbsp;&nbsp;&nbsp;&nbsp; 2.3 Gender and Contract Type Distribution and Target Variable    
# &nbsp;&nbsp;&nbsp;&nbsp; 2.4 Own Realty and Own Car  - Distribution with Target Variable  
# &nbsp;&nbsp;&nbsp;&nbsp; 2.5 Suit Type and Income Type    
# &nbsp;&nbsp;&nbsp;&nbsp; 2.6 Family Statue and Housing Type   
# &nbsp;&nbsp;&nbsp;&nbsp; 2.7 Education Type and Income Type   
# &nbsp;&nbsp;&nbsp;&nbsp; 2.8.1 Organization Type and Occupation Type   
# &nbsp;&nbsp;&nbsp;&nbsp; 2.8.2 Walls Material, Foundation and House Type   
# &nbsp;&nbsp;&nbsp;&nbsp; 2.9 Amount Credit Distribution    
# &nbsp;&nbsp;&nbsp;&nbsp; 2.10 Amount Annuity Distribution  
# &nbsp;&nbsp;&nbsp;&nbsp; 2.11 Amount Goods Price   
# &nbsp;&nbsp;&nbsp;&nbsp; 2.12 Amount Region Population Relative    
# &nbsp;&nbsp;&nbsp;&nbsp; 2.13 Days Birth   
# &nbsp;&nbsp;&nbsp;&nbsp; 2.14 Days Employed    
# &nbsp;&nbsp;&nbsp;&nbsp; 2.15 Num Days Registration  
# &nbsp;&nbsp;&nbsp;&nbsp; 2.15 Count of Family Members  
# 3. Exploration - Bureau Data  
# &nbsp;&nbsp;&nbsp;&nbsp; 3.1 Snapshot - Bureau Data    
# 4. Exploration - Bureau Balance Data  
# &nbsp;&nbsp;&nbsp;&nbsp; 4.1 Snapshot - Bureau Balance Data     
# 5. Exploration - Credit Card Balance Data   
# &nbsp;&nbsp;&nbsp;&nbsp; 5.1 Snapshot - Credit Card Balance Data   
# 6. Exploration - POS Cash Balance Data   
# &nbsp;&nbsp;&nbsp;&nbsp; 6.1 Snapshot - POS Cash Balance Data   
# 7. Exploration - Previous Application Data   
# &nbsp;&nbsp;&nbsp;&nbsp; 7.1 Snapshot - Previous Application Data  
# &nbsp;&nbsp;&nbsp;&nbsp; 7.2 Contract Status Distribution - Previous Applications  
# &nbsp;&nbsp;&nbsp;&nbsp; 7.3 Suite Type Distribution - Previous Application    
# &nbsp;&nbsp;&nbsp;&nbsp; 7.4 Client Type Distribution  - Previous Application    
# &nbsp;&nbsp;&nbsp;&nbsp; 7.5 Channel Type Distribution - Previous Applications  
# 7. Exploration - Installation Payments  
# &nbsp;&nbsp;&nbsp;&nbsp; 8.1 Snapshot of Installation Payments  
# 9. Baseline Model  
# &nbsp;&nbsp;&nbsp;&nbsp; 9.1 Dataset Preparation  
# &nbsp;&nbsp;&nbsp;&nbsp; 9.2 Label Encoding     
# &nbsp;&nbsp;&nbsp;&nbsp; 9.3 Validation Sets Preparation    
# &nbsp;&nbsp;&nbsp;&nbsp; 9.4 Model Fitting    
# &nbsp;&nbsp;&nbsp;&nbsp; 9.5 Feature Importance    
# &nbsp;&nbsp;&nbsp;&nbsp; 9.6 Prediction 
# 
# 
# 
# ## 1. Dataset Preparation 

# In[19]:


from plotly.offline import init_notebook_mode, iplot
from wordcloud import WordCloud
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import plotly.plotly as py
from plotly import tools
from datetime import date
import pandas as pd
import numpy as np 
import seaborn as sns
import random 
import warnings
warnings.filterwarnings("ignore")
init_notebook_mode(connected=True)

path = "../input/"

def bar_hor(df, col, title, color, w=None, h=None, lm=0, limit=100, return_trace=False, rev=False, xlb = False):
    """
    横向き棒グラフ作成関数
    df:  
    col: 
    title: 
    color: 
    w=None: 
    h=None: 
    lm=0: 
    limit=100: 
    return_trace=False: 
    rev=False: 
    xlb = False:
    """
    cnt_srs = df[col].value_counts()
    yy = cnt_srs.head(limit).index[::-1] 
    xx = cnt_srs.head(limit).values[::-1] 
    if rev:
        yy = cnt_srs.tail(limit).index[::-1] 
        xx = cnt_srs.tail(limit).values[::-1] 
    if xlb:
        trace = go.Bar(y=xlb, x=xx, orientation = 'h', marker=dict(color=color))
    else:
        trace = go.Bar(y=yy, x=xx, orientation = 'h', marker=dict(color=color))
    if return_trace:
        return trace 
    layout = dict(title=title, margin=dict(l=lm), width=w, height=h)
    data = [trace]
    fig = go.Figure(data=data, layout=layout)
    iplot(fig)

def bar_hor_noagg(x, y, title, color, w=None, h=None, lm=0, limit=100, rt=False):
    trace = go.Bar(y=x, x=y, orientation = 'h', marker=dict(color=color))
    if rt:
        return trace
    layout = dict(title=title, margin=dict(l=lm), width=w, height=h)
    data = [trace]
    fig = go.Figure(data=data, layout=layout)
    iplot(fig)


def bar_ver_noagg(x, y, title, color, w=None, h=None, lm=0, rt = False):
    trace = go.Bar(y=y, x=x, marker=dict(color=color))
    if rt:
        return trace
    layout = dict(title=title, margin=dict(l=lm), width=w, height=h)
    data = [trace]
    fig = go.Figure(data=data, layout=layout)
    iplot(fig)
    
def gp(col, title):
    """
    グループ化棒グラフを表示
    col: 表示する列
    title: 図のタイトル
    """
    df1 = app_train[app_train["TARGET"] == 1]
    df0 = app_train[app_train["TARGET"] == 0]
    a1 = df1[col].value_counts()
    b1 = df0[col].value_counts()

    trace1 = go.Bar(x=a1.index, y=a1.values, name='Target : 1', marker=dict(color="#44ff54"))
    trace2 = go.Bar(x=b1.index, y=b1.values, name='Target : 0', marker=dict(color="#ff4444"))

    data = [trace1, trace2]
    layout = go.Layout(barmode='group', height=300, title = title)

    fig = go.Figure(data=data, layout=layout)
    iplot(fig, filename='grouped-bar')

# ## 2. データ探索: Application (ローン申込書)
# 
# ## 2.1 Application Train の概観
# 
# Application データは全ローン申込書の統計情報からなり、各行が1つのローンを表す。

# In[22]:


app_train = pd.read_csv(path + "application_train.csv") # application train データの読み込み
app_train.head() # 最初の5行を表示

# > 307,511件のローンのデータがあり、列数は122です。
# 
# ## 2.2 目的変数の分布
# 目的変数
# - 1: 支払が困難なクライアント = クライアントが最初のY回の分割払いの内に少なくとも一回でX日以上延滞していた場合
# - 0: それ以外の場合

# In[21]:


# 目的変数の分布
# 横向き棒グラフの表示
bar_hor(app_train, # 表示するデータ 
        "TARGET", # 表示する列名
        "Distribution of Target Variable" , # 図のタイトル
        ["#44ff54", '#ff4444'], # 棒グラフの色
        h=400, # 図の高さ
        w=800, # 図の幅
        lm=100, # ?
        xlb = ['Target : 1','Target : 0'] # 各バーの説明
       )

# > - 目的変数は約282k (85%) が 0 で、わずか24kが 1 です。
# 
# ## 2.3 どの性別、どの契約タイプがローンを申し込んでいるか
# - 性別: クライアントの性別  
# - 契約タイプ: ローンがキャッシュかリボ払いか  
# 
# ### 2.3.1 性別・契約タイプの分布

# In[22]:


tr1 = bar_hor(app_train, "CODE_GENDER", "Distribution of CODE_GENDER Variable" ,"#639af2", w=700, lm=100, return_trace= True) # 性別の棒グラフ
tr2 = bar_hor(app_train, "NAME_CONTRACT_TYPE", "Distribution of NAME_CONTRACT_TYPE Variable" ,"#a4c5f9", w=700, lm=100, return_trace = True) # 契約タイプの棒グラフ

# 図の整形・表示
fig = tools.make_subplots(rows=1, cols=2, print_grid=False, subplot_titles = ['Gender' , 'Contract Type'])
fig.append_trace(tr1, 1, 1);
fig.append_trace(tr2, 1, 2);
fig['layout'].update(height=400, showlegend=False, margin=dict(b=100));
iplot(fig);

# > 性別については、女性が多く (202448) 男性は少ない (105059)。  
# > 契約タイプについてはキャッシュが主でリボ払いは約29kとかなり少ない。
# 
# ### 2.3.2 性別・契約タイプと目的変数との関係

# In[23]:


# 性別・目的変数のグループ化棒グラフ
gp('CODE_GENDER', # 表示する列名
   'Distribution of Target with Gender' # 図のタイトル
  ) 
gp('NAME_CONTRACT_TYPE', 'Distribution of Target with Contract Type') # 契約タイプ・目的変数のグループ化棒グラフ

# ## 2.4. 土地所有・車所有

# In[24]:


tr1 = bar_hor(app_train, "FLAG_OWN_REALTY", "Distribution of FLAG_OWN_REALTY" ,"#639af2", w=700, lm=100, return_trace= True)
tr2 = bar_hor(app_train, "FLAG_OWN_CAR", "Distribution of FLAG_OWN_CAR" ,"#639af2", w=700, lm=100, return_trace = True)

fig = tools.make_subplots(rows=1, cols=2, print_grid=False, subplot_titles = ['Own Realty' , 'Own CAR'])
fig.append_trace(tr1, 1, 1);
fig.append_trace(tr2, 1, 2);
fig['layout'].update(height=400, showlegend=False, margin=dict(b=100));
iplot(fig);


gp('FLAG_OWN_REALTY', 'Distribution of Target with FLAG_OWN_REALTY')
gp('FLAG_OWN_CAR', 'Distribution of Target with FLAG_OWN_CAR Type')

# ## 2.5 同伴者・収入形態
# - 同伴者 (NAME_TYPE_SUITE): 借入申請書提出時に同伴した人物
# 
# ### 2.5.1 同伴者・収入形態の値

# In[23]:


tr1 = bar_hor(app_train, "NAME_TYPE_SUITE", "Distribution of NAME_TYPE_SUITE" ,"#639af2", w=700, lm=100, return_trace= True) # 世帯構成タイプの棒グラフ
tr2 = bar_hor(app_train, "NAME_INCOME_TYPE", "Distribution of NAME_INCOME_TYPE" ,"#a4c5f9", w=700, lm=100, return_trace = True) # 収入タイプの棒グラフ

fig = tools.make_subplots(rows=1, cols=2, print_grid=False, subplot_titles = ['Applicants Suite Type' , 'Applicants Income Type'])
fig.append_trace(tr1, 1, 1);
fig.append_trace(tr2, 1, 2);
fig['layout'].update(height=400, showlegend=False, margin=dict(l=100));
iplot(fig);

# > 同伴者のトップ3は同伴者なし (250k)、家族、夫婦である。
# > 収入形態は8タイプがありトップは:  
#     - Working Class労働階級 (158K)
#     - Pensiner 年金受給者 (55K)同伴者
# 
# ### 2.5.2 同伴者・収入形態と目的変数との関係

# In[26]:


gp('NAME_TYPE_SUITE', 'Target with respect to Suite Type of Applicants')
gp('NAME_INCOME_TYPE', 'Target with respect to Income Type of Applicants')

# ## 2.6. 婚姻状況・住居
# 
# ### 2.6.1 婚姻状況・住居の値

# In[27]:


tr1 = bar_hor(app_train, "NAME_FAMILY_STATUS", "Distribution of NAME_FAMILY_STATUS" ,"#639af2", w=700, lm=100, return_trace= True)
tr2 = bar_hor(app_train, "NAME_HOUSING_TYPE", "Distribution of NAME_HOUSING_TYPE" ,"#a4c5f9", w=700, lm=100, return_trace = True)

fig = tools.make_subplots(rows=1, cols=2, print_grid=False, subplot_titles = ['NAME_FAMILY_STATUS' , 'NAME_HOUSING_TYPE'])
fig.append_trace(tr1, 1, 1);
fig.append_trace(tr2, 1, 2);
fig['layout'].update(height=400, showlegend=False, margin=dict(l=100));
iplot(fig);

# > - 既婚の顧客が最も多く (約196k) 独身がそれに続く。
# > - 住居は多くが "一軒家/アパート" で85%を占め、両親と同居、公営住宅が続く。
# 
# ### 2.6.2 婚姻状況・住居と目的変数との関係

# In[28]:


gp('NAME_FAMILY_STATUS', 'Target and Family Status')
gp('NAME_HOUSING_TYPE', 'Target and Housing Type')

# ## 2.7. 教育
# 
# ### 2.7.1 教育の分布

# In[29]:


tr1 = bar_hor(app_train, "NAME_EDUCATION_TYPE", "Distribution of NAME_EDUCATION_TYPE" ,"#639af2", w=700, lm=100, return_trace= True)
# tr2 = bar_hor(app_train, "NAME_INCOME_TYPE", "Distribution of NAME_FAMILY_STATUS" ,"#a4c5f9", w=700, lm=100, return_trace = True)

fig = tools.make_subplots(rows=1, cols=1, print_grid=False, subplot_titles = ['NAME_EDUCATION_TYPE' ])
fig.append_trace(tr1, 1, 1);
# fig.append_trace(tr2, 1, 2);
fig['layout'].update(height=400, showlegend=False, margin=dict(l=100));
iplot(fig);

# > 多くの割合が中等教育に占められ (218k)、高等教育 (75k) がそれに続く。
# 
# ### 2.7.2 教育と目的変数との関係

# In[30]:


gp('NAME_EDUCATION_TYPE', 'Education Type and Target')

# ## 2.8. 組織・業種
# -  組織: クライアントが働いている組織
# -  業種: クライアントの業種
# 
# ### 2.8.1 組織・業種の分布

# In[34]:


tr1 = bar_hor(app_train, "ORGANIZATION_TYPE", "Distribution of ORGANIZATION_TYPE" ,"#639af2", w=700, lm=100, return_trace= True)
tr2 = bar_hor(app_train, "OCCUPATION_TYPE", "Distribution of OCCUPATION_TYPE" ,"#a4c5f9", w=700, lm=100, return_trace = True)

fig = tools.make_subplots(rows=1, cols=2, print_grid=False, subplot_titles = ['ORGANIZATION_TYPE' , 'OCCUPATION_TYPE'])
fig.append_trace(tr1, 1, 1);
fig.append_trace(tr2, 1, 2);
fig['layout'].update(height=600, showlegend=False, margin=dict(l=200));
iplot(fig);

# > 申請者の中で多い業種は労働者 (55k)、販売員 (32k)、コアスタッフ (28k)  
# > 多い組織は第3種法人が最多で67kを占める
# 
# ### 2.8.2 組織・業種と目的変数との関係

# In[24]:


gp('ORGANIZATION_TYPE', 'Organization Type and Target')
gp('OCCUPATION_TYPE', 'Occupation Type and Target')

# ### 2.8.3. 壁の種類・基礎の種類・家屋の種類の分布
# 変数の意味がよくわかりません…

# In[25]:


tr1 = bar_hor(app_train, "WALLSMATERIAL_MODE", "Distribution of FLAG_OWN_CAR" ,"#a4c5f9", w=700, lm=100, return_trace = True)
tr2 = bar_hor(app_train, "FONDKAPREMONT_MODE", "Distribution of FLAG_OWN_REALTY" ,"#639af2", w=700, lm=100, return_trace= True)
tr3 = bar_hor(app_train, "HOUSETYPE_MODE", "Distribution of FLAG_OWN_CAR" ,"#a4c5f9", w=700, lm=100, return_trace = True)

fig = tools.make_subplots(rows=1, cols=3, print_grid=False, subplot_titles = ['WALLSMATERIAL_MODE' , 'FONDKAPREMONT_MODE', 'HOUSETYPE_MODE'])
fig.append_trace(tr1, 1, 1);
fig.append_trace(tr2, 1, 2);
fig.append_trace(tr3, 1, 3);

fig['layout'].update(height=400, showlegend=False, margin=dict(l=100));
iplot(fig);

# > - 平屋が150kでほとんどを占め、特殊家屋、テラスハウスは1500以下である。
# > - 壁はパネル、石・レンガがほぼ同数で120k近くを占める。
# 
# ### 2.8.4 壁の種類・基礎の種類・家屋の種類と目的変数との関係

# In[37]:


gp('WALLSMATERIAL_MODE', 'Wall Material and Target')
gp('FONDKAPREMONT_MODE', 'Foundation and Target')
gp('HOUSETYPE_MODE', 'House Type and Target')

# ## 2.9. 借金額の分布

# In[27]:


plt.figure(figsize=(12,5))
plt.title("Distribution of AMT_CREDIT")
ax = sns.distplot(app_train["AMT_CREDIT"])

# ## 2.10 年金の分布
# - 年金: ローン年金 (って何?)

# In[26]:


plt.figure(figsize=(12,5))
plt.title("Distribution of AMT_ANNUITY")
ax = sns.distplot(app_train["AMT_ANNUITY"].dropna())

# ## 2.11 商品価格の分布
# - 商品価格: ローンを組む目的である商品の価格

# In[28]:


plt.figure(figsize=(12,5))
plt.title("Distribution of AMT_GOODS_PRICE")
ax = sns.distplot(app_train["AMT_GOODS_PRICE"].dropna())
# ax = sns.distplot(app_train["CNT_CHILDREN"], kde = False)

# ## 2.12 相対地域人口分布の分布
# - 相対地域人口: 正規化されたクライアントが住んでいる地域の人口 (クライアントがより人工の多い地域に住んでいることを表す) 

# In[29]:


plt.figure(figsize=(12,5))
plt.title("Distribution of REGION_POPULATION_RELATIVE")
ax = sns.distplot(app_train["REGION_POPULATION_RELATIVE"])

# ## 2.13 年齢の分布
# - 年齢: クライアントが借入申請日の何日前に生まれたか

# In[43]:


plt.figure(figsize=(12,5))
plt.title("Distribution of DAYS_BIRTH")
ax = sns.distplot(app_train["DAYS_BIRTH"])

# ## 2.14 雇用日数の分布
# - 雇用日数: クライアントが借入申請日の何日前から現在の仕事を始めたか

# In[46]:


plt.figure(figsize=(12,5))
plt.title("Distribution of DAYS_EMPLOYED")
ax = sns.distplot(app_train["DAYS_EMPLOYED"])

# ## 2.15 登録日の分布
# - 登録日: クライアントが借入申請日の何日前に登録情報を更新したか

# In[47]:


plt.figure(figsize=(12,5))
plt.title("Distribution of DAYS_REGISTRATION")
ax = sns.distplot(app_train["DAYS_REGISTRATION"])

# ## 2.16 家族人数

# In[48]:


plt.figure(figsize=(12,5))
plt.title("Distribution of CNT_FAM_MEMBERS")
ax = sns.distplot(app_train["CNT_FAM_MEMBERS"].dropna())

# ## 3. 信用情報機関データのデータ探索
# 
# 信用情報機関によって報告されている顧客の過去の他の金融機関で借入履歴。  
# 顧客の借入申込日以前の借入回数と同じ行数の借入情報が含まれる。  
# 
# ## 3.1 信用情報機関データの概観

# In[30]:


bureau = pd.read_csv(path + "bureau.csv")
bureau.head()

# ## 4. 信用情報機関残高のデータ探索
# 
# 信用情報機関の過去の借入の月間残高。   
# このテーブルには、過去の借入についての各月の残高が1行ずつ記録されています。  
# テーブルの各列には、ある借入のx月 (借入申請からxヶ月前) の債務状況の情報が含まれます。  
# 
# ## 4.1 信用情報機関残高データの概観

# In[31]:


bureau_balance = pd.read_csv(path + "bureau_balance.csv")
bureau_balance.head()

# ## 5. クレジットカード残高のデータ探索
# 
# 借入申請者の持つホーム・クレジット社製クレジットカードの各月の残高情報。  
# このテーブルには、借入申請者の持つホーム・クレジット社製クレジットカード (消費者金融・キャッシュローン) の各月の残高が1行ずつ記録されています。  
# テーブルの各列には、あるクレジットカードのx月 (借入申請からxヶ月前) の債務状況の情報が含まれます。 
# 
# ## 5.1 クレジットカード残高データの概観

# In[11]:


credit_card_balance = pd.read_csv(path + "credit_card_balance.csv")
credit_card_balance.head()

# ## 6. POSキャッシュ残高のデータ探索
# 
# 借入申請者の持つホーム・クレジットに関する過去のPOSとキャッシュローンの各月の残高情報。  
# テーブルの各列には、あるローンのx月 (借入申請からxヶ月前) の債務状況の情報が含まれます。 
# このテーブルには、ローン の各月の残高が1行ずつ記録されています。  
# (訳注: よくわかりませんでした)
# 
# ## 6.1 POSキャッシュ残高データの概観

# In[32]:


pcb = pd.read_csv(path + "POS_CASH_balance.csv")
pcb.head()

# ## 7. 過去の借入申請書のデータ探索
# 
# ## 7.1  過去の借入申請書データの概観

# In[33]:


previous_application = pd.read_csv(path + "previous_application.csv")
previous_application.head()

# ## 7.2 過去の借入申請書の契約状況の分布
# - 契約状況:  受理、拒否...

# In[14]:


# 契約状況 (受理、拒否、申請キャンセル、申請なし (unused offer))の比率の円グラフを表示
t = previous_application['NAME_CONTRACT_STATUS'].value_counts()
labels = t.index
values = t.values

colors = ['#FEBFB3', '#E1396C', '#96D38C', '#D0F9B1']

trace = go.Pie(labels=labels, 
               values=values,
               hoverinfo='', 
               textinfo='',
               textfont=dict(size=12), # フォントサイズ
               marker=dict(colors=colors, # 色設定
                           line=dict(color='#fff', width=2))
              )

layout = go.Layout(title='Name Contract Status in Previous Applications', height=400)
fig = go.Figure(data=[trace], layout=layout)
iplot(fig)

# >-  多くの人が過去に申請が受理されている (62%)。一方で19%がキャンセル、17%が拒否となっている。
# 
# ## 7.3 過去の借入申請書の同伴者の分布

# In[16]:


t = previous_application['NAME_TYPE_SUITE'].value_counts()
labels = t.index
values = t.values

colors = ['#FEBFB3', '#E1396C', '#96D38C', '#D0F9B1']

trace = go.Pie(labels=labels, values=values,
               hoverinfo='', textinfo='',
               textfont=dict(size=12),
               marker=dict(colors=colors,
                           line=dict(color='#fff', width=2)))

layout = go.Layout(title='Suite Type in Previous Application Distribution', height=400)
fig = go.Figure(data=[trace], layout=layout)
iplot(fig)

# >- 過去の申請書の同伴者の多くが同伴者なしであり (60%)、家族がそれに続く (25%)。
# 
# ## 7.4 過去の借入申請書の顧客タイプ

# In[17]:


t = previous_application['NAME_CLIENT_TYPE'].value_counts()
labels = t.index
values = t.values

colors = ['#FEBFB3', '#E1396C', '#96D38C', '#D0F9B1']

trace = go.Pie(labels=labels, values=values,
               hoverinfo='', textinfo='',
               textfont=dict(size=12),
               marker=dict(colors=colors,
                           line=dict(color='#fff', width=2)))

layout = go.Layout(title='Client Type in Previous Applications', height=400)
fig = go.Figure(data=[trace], layout=layout)
iplot(fig)

# >- 過去の借入申請者の74%がリピーターで18%が新規、8%が再登録? (refreshed)である。
# 
# ## 7.5 チャネルタイプ
# - チャネルタイプ: どの方法で借入申請書を受け取ったか

# In[34]:


t = previous_application['CHANNEL_TYPE'].value_counts()
labels = t.index
values = t.values

colors = ['#FEBFB3', '#E1396C', '#96D38C', '#D0F9B1']

trace = go.Pie(labels=labels, values=values,
               hoverinfo='', textinfo='',
               textfont=dict(size=12),
               marker=dict(colors=colors,
                           line=dict(color='#fff', width=2)))

layout = go.Layout(title='Channel Type in Previous Applications', height=400)
fig = go.Figure(data=[trace], layout=layout)
iplot(fig)

# ## 8. 分割支払のデータ探索
# ## 8.1 分割支払データの概観

# In[35]:


installments_payments = pd.read_csv(path + "installments_payments.csv")
installments_payments.head()

# ## 9. ベースライン・モデル
# 
# ### 9.1 前処理

# In[36]:


from sklearn.model_selection import train_test_split 
import lightgbm as lgb

# テストファイル読み込み
app_test = pd.read_csv('../input/application_test.csv')

app_test['is_test'] = 1 
app_test['is_train'] = 0
app_train['is_test'] = 0
app_train['is_train'] = 1

# 目的変数
Y = app_train['TARGET']
train_X = app_train.drop(['TARGET'], axis = 1)

# テストID
test_id = app_test['SK_ID_CURR']
test_X = app_test

# 前処理のためにトレインとテストを連結
data = pd.concat([train_X, test_X], axis=0)

# ### 9.2 カテゴリ変数の処理
# 
# より良い処理をしたければOliverの素晴らしいkernelを見に行ってください: https://www.kaggle.com/ogrellier/good-fun-with-ligthgbm 

# In[45]:


# カテゴリ変数を取得する関数
def _get_categorical_features(df):
    feats = [col for col in list(df.columns) if df[col].dtype == 'object']
    return feats

# カテゴリ変数をファクトライズ (整数に置換)する関数
def _factorize_categoricals(df, cats):
    for col in cats:
        df[col], _ = pd.factorize(df[col])
    return df 

# カテゴリ変数のダミー変数 (二値変数化)を作成する関数
def _get_dummies(df, cats):
    for col in cats:
        df = pd.concat([df, pd.get_dummies(df[col], prefix=col)], axis=1)
    return df 

# カテゴリ変数を取得
data_cats = _get_categorical_features(data)
prev_app_cats = _get_categorical_features(previous_application)
bureau_cats = _get_categorical_features(bureau)

# ダミー変数を取得
previous_application = _get_dummies(previous_application, prev_app_cats)
bureau = _get_dummies(bureau, bureau_cats)

# カテゴリ変数をファクトライズ
data = _factorize_categoricals(data, data_cats)

# ### 9.3 データセットを一箇所にまとめる

# In[ ]:


## 参考: より良い特徴量 : https://www.kaggle.com/ogrellier/good-fun-with-ligthgbm 

### 過去の申請書情報を現在の申請書情報と結びつける

# IDごとに過去の申請回数をカウント
prev_apps_count = previous_application[['SK_ID_CURR', 'SK_ID_PREV']].groupby('SK_ID_CURR').count()
previous_application['SK_ID_PREV'] = previous_application['SK_ID_CURR'].map(prev_apps_count['SK_ID_PREV'])

# IDごとに過去の申請書の特徴量の値の平均を取る
prev_apps_avg = previous_application.groupby('SK_ID_CURR').mean()
prev_apps_avg.columns = ['p_' + col for col in prev_apps_avg.columns]
data = data.merge(right=prev_apps_avg.reset_index(), how='left', on='SK_ID_CURR')

### 信用情報機関を現在の申請書情報と結びつける

# IDごとに信用情報機関の特徴量の値の平均を取る
bureau_avg = bureau.groupby('SK_ID_CURR').mean()
bureau_avg['buro_count'] = bureau[['SK_ID_BUREAU','SK_ID_CURR']].groupby('SK_ID_CURR').count()['SK_ID_BUREAU']
bureau_avg.columns = ['b_' + f_ for f_ in bureau_avg.columns]
data = data.merge(right=bureau_avg.reset_index(), how='left', on='SK_ID_CURR')

# 最終的なtrainとtestデータを用意
ignore_features = ['SK_ID_CURR', 'is_train', 'is_test']
relevant_features = [col for col in data.columns if col not in ignore_features]
trainX = data[data['is_train'] == 1][relevant_features]
testX = data[data['is_test'] == 1][relevant_features]

# ### 9.4 validationデータの作成

# In[47]:


x_train, x_val, y_train, y_val = train_test_split(trainX, Y, test_size=0.2, random_state=18)
lgb_train = lgb.Dataset(data=x_train, label=y_train)
lgb_eval = lgb.Dataset(data=x_val, label=y_val)

# ### 9.5 モデル (Light GBM)の学習

# In[48]:


params = {'task': 'train', 'boosting_type': 'gbdt', 'objective': 'binary', 'metric': 'auc', 
          'learning_rate': 0.01, 'num_leaves': 48, 'num_iteration': 5000, 'verbose': 0 ,
          'colsample_bytree':.8, 'subsample':.9, 'max_depth':7, 'reg_alpha':.1, 'reg_lambda':.1, 
          'min_split_gain':.01, 'min_child_weight':1}
model = lgb.train(params, lgb_train, valid_sets=lgb_eval, early_stopping_rounds=150, verbose_eval=200)

# ### 9.6 特徴量の重要度

# In[49]:


lgb.plot_importance(model, figsize=(12, 50));

# ### 9.7 推定

# In[50]:


preds = model.predict(testX)
sub_lgb = pd.DataFrame()
sub_lgb['SK_ID_CURR'] = test_id
sub_lgb['TARGET'] = preds
sub_lgb.to_csv("lgb_baseline.csv", index=False)
sub_lgb.head()

# In[ ]:



