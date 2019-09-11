#!/usr/bin/env python
# coding: utf-8

# # Reducing Concussions for Punt Plays
# <img src="https://s3.amazonaws.com/nonwebstorage/headstrong/lineup.png" />

# In[ ]:


<style type="text/css">

div.h2 {
    background-color: steelblue; 
    color: white; 
    padding: 8px; 
    padding-right: 300px; 
    font-size: 20px; 
    max-width: 1500px; 
    margin: auto; 
    margin-top: 50px;
}
div.h3 {
    color: steelblue; 
    font-size: 14px; 
    margin-top: 20px; 
    margin-bottom:4px;
}
div.h4 {
    font-size: 15px; 
    margin-top: 20px; 
    margin-bottom: 8px;
}
span.note {
    font-size: 5; 
    color: gray; 
    font-style: italic;
}
span.captiona {
    font-size: 5; 
    color: dimgray; 
    font-style: italic;
    margin-left: 130px;
    vertical-align: top;
}
hr {
    display: block; 
    color: gray
    height: 1px; 
    border: 0; 
    border-top: 1px solid;
}
hr.light {
    display: block; 
    color: lightgray
    height: 1px; 
    border: 0; 
    border-top: 1px solid;
}
table.dataframe th 
{
    border: 1px darkgray solid;
    color: black;
    background-color: white;
}
table.dataframe td 
{
    border: 1px darkgray solid;
    color: black;
    background-color: white;
    font-size: 14px;
    text-align: center;
} 
table.rules th 
{
    border: 1px darkgray solid;
    color: black;
    background-color: white;
    font-size: 14px;
}
table.rules td 
{
    border: 1px darkgray solid;
    color: black;
    background-color: white;
    font-size: 13px;
    text-align: center;
} 
table.rules tr.best
{
    color: green;
}

</style>

# In[ ]:


# import
import warnings
import numpy as np
import pandas as pd
from scipy import stats
import hvplot.pandas
from IPython.display import HTML, Image

# set additional display options for report
pd.set_option("display.max_columns", 100)
th_props = [('font-size', '13px'), ('background-color', 'white'), 
            ('color', '#666666')]
td_props = [('font-size', '15px'), ('background-color', 'white')]
styles = [dict(selector="td", props=td_props), dict(selector="th", 
            props=th_props)]

# This report represents my analysis for the NFL Punt Analytics Competition. It is my opinion that based on the data provided, changing two rules for punt plays could result in up to 8 fewer concussions per year. The two proposed changes are as follows:
# 
#   - *Move the ball forward 10 yards after a Fair Catch.* After fair catch of a scrimmage kick, play would start 10 yards forward from the spot of catch, or the succeeding spot after eforcement of penalties. This would apply when the receiving team elects to put the ball in play by a snap.
# 
#   - *Penalize blindside blocks with forcible contact to any part of a defenseless player's body.*  Defenseless players as described by NFL rules include players receiving a blindside block from a blocker whose path is toward or parallel his own end line. Prohibited contact for punt plays would include blindside blocks with forcible contact to any part of a defenseless player's body, including below the neck.
# 
# The figure below shows the potential reduction in concussions based on 2016-2017 data and associated assumptions.
# <br>
# 
# <span style="display:inline-block; margin-top:30px; margin-left:112px; font-size:12pt; font-weight:bold">Potential Reductions</span>
# <img src="https://s3.amazonaws.com/nonwebstorage/headstrong/redux.png" width="600" height="400">
# 
# 
# <p style="margin-top:40px"> 
# My analysis is thoroughly documented in the following sections:  </p>
# 
#   - <a href='#bg'>Background</a>
#   - <a href='#mt'>Methodology</a>
#   - <a href='#an'>Analysis</a>
#   - <a href='#cn'>Conclusion</a>
#   - <a href='#tm'>Qualifications</a>
#   - <a href='#ap'>Appendix</a>
# 
#   
# You can find the complete code for preparing the attached data files in the following kernels and datasets:
# 
#    - [NFL Data Preparation](https://www.kaggle.com/jpmiller/nfl-data-preparation)
#    - [NFL Punt Play Animation](https://www.kaggle.com/jpmiller/nfl-punt-play-animation)
#    - [NFL Concussion Data](https://www.kaggle.com/jpmiller/nfl-competition-data)
# 
# <p style="margin-top:40px">
# Thank you and happy reading!<br>
# John Miller<br>
# Benbrook Analytics</p>

# <a id='bg'></a>
# <div class="h2">  Background</div>
# 
# <p style="margin-top: 20px">Since 1920 the National Football League has developed the model for a successful modern sports league. The NFL enterprise includes national and international distribution, extensive revenue sharing, competitive excellence, and strong franchises across the country. Last year, 29 of the NFL's 32 teams appeared in the Forbes Top 50 most valuable sports franchises in the world. Football is also regarded as the United States' most popular sport. A 2018 Gallup poll found that among US adults, 37% name football as their favorite sport to watch. The number tops basketaball(11%), baseball(9%) and soccer(7%) by a wide margin. </p>
# 
# Player safety has always been a concern for football. Over the past few years, concussions sustained during play have become one of the most visible issues affecting players. Concussion incidents rose in 2017 even as the NFL implemented several safety measures. The figure below shows the incidence of concussions since 2012. 
# 
# <span class="note"> <i>Hover over the points to see exact numbers.</i> </span>

# In[ ]:


hist_df = pd.DataFrame({'Season': np.arange(2012,2018), 
            'Concussions': [265, 244, 212, 279, 250, 291]})
line_incidents = hist_df.hvplot.line(x='Season', y='Concussions', 
            xlim = (2011.5, 2017.5), ylim=(0,350), 
            title='Concussion Incidents for Full Season (incl Practice)',
            yticks=np.arange(50,350,50).tolist(), 
            xticks=np.arange(2012,2018).tolist(), grid=True)
scat_incidents = hist_df.hvplot.scatter(x='Season', y='Concussions'
            , size=50)
display(line_incidents * scat_incidents,
 HTML('<span class="captiona">' + 'Source: IQVIA' + '<span'))

# <br>
# Kick and punt plays have historically posed the highest risk of concussion to players. During the 2015-2017 seasons, the kickoff represented only six percent of plays but 12 percent of concussions, making the risk four times greater than running or passing plays. In response, the NFL revised its kicking rules for 2018 to reduce risk during kickoffs. 
# 
# The NFL is now increasing the attention given to punt plays. According to NFL executive Jeff Miller, concussion risk during punts is twice that of running or passing plays. In response the NFL is sponsoring this competition as part of an overall effort to make punt plays safer. The goal of the competition is to discover specific rule modifications, supported by data, that may reduce the occurrence of concussions during punt plays.

# <a id='mt'></a>
# <div class="h2">  Methodology </div>
# 
# <p style="margin-top: 20px">I followed a rigorous methodology designed to help solve problems like the one posed in this challenge. The figure below shows the steps and the tools used for each step. The last step is shown for completeness only since it was out of scope for this analysis.</p>
# 
# <span style="display:inline-block; margin-top:30px; margin-left:60px; font-size:12pt; font-weight:bold">Problem Solving Steps and Tools </span>
# <img src="https://s3.amazonaws.com/nonwebstorage/headstrong/problemsolve2.png" width="700">
# 
# In practice, my path through the steps was not completely linear as the figure suggests. I iterated across steps as the analysis progressed.
# 
# I used machine learning for part of the analysis in addition to conventional analytics. I applied a type of modeling called *Unsupervised Machine Learning*. Such models are well-suited for uncovering patterns hidden in complex data. The model works by grouping "like with like", i.e., punt plays of a similar nature as they appear in the data.

# <a id='an'></a>
# <div class="h2">  Analysis </div>
# 
# <div class="h3"> UNDERSTANDING THE PROBLEM </div>
# My first task was to explore the problem presented by the NFL. I researched information on football and sports safety to improve my understanding. Resources included the following items:
# 
#   - Sports articles on punt plays and players
#   - NFL Official Rules
#   - Videos and game highlights
#   - NFL press releases, articles and general statistics
#   - Information on NFL Play Smart, Play Safe to include past and current initiatives
#   - Information from other sports and football leagues addressing concussions  
#   - Articles and research papers on sports-related concussions 
#   
#   
# I also needed to get an independent idea of concussion risk for punt plays relative to other plays. I first analyzed data provided for the competition along with data scraped from nfl.com and concussion data compiled by IQVIA. The table below shows the percentage of concussions for passing and running plays vs. the percentage from punt plays.

# In[ ]:


years = [2016, 2017]
def get_sums(file, collist):
    df = pd.read_csv(file, usecols=['Year'] + collist)
    return [df.loc[df.Year == y, collist].sum().sum() for y in years]


scrims = get_sums('../input/nfl-competition-data/nflcom_scrims.csv', 
                  ['Scrm Plys'])
punts = get_sums('../input/nfl-competition-data/nflcom_punts.csv', 
                  ['Punts', 'Blk'])

punt_concussions = pd.read_csv('../input/NFL-Punt-Analytics-Competition/video_review.csv', 
                  usecols = ['Season_Year']).Season_Year.value_counts().values

kick_conc_pct = 0.12 #source: Kaggle competition Overview
scrimmage_concussions = np.array([217, 235]) * (1-kick_conc_pct) - punt_concussions 
                                                    

pcts_df = pd.DataFrame({'Passes_Runs': (scrimmage_concussions/scrims),
        'Punts': punt_concussions/punts}, index=years)
pcts_df['Risk_Multiple'] = (pcts_df.Punts/pcts_df.Passes_Runs).apply('{:.1f}'.format)
pcts_df['Passes_Runs'] =pcts_df.Passes_Runs.apply('{:.2%}'.format)
pcts_df['Punts'] =pcts_df.Punts.apply('{:.2%}'.format)
display(HTML('<span style="font-weight:bold">' + 'Concussion Percentages by Play Type'\
             + '</span>'),pcts_df) 

# According to these approximate calculations, punt plays were 1.2 to 1.4 times more dangerous than pass and run plays in 2016 and 2017. These numbers include all punt plays to include out of bounds punts and down punts. I show later in the analysis that punt plays carry a higher risk multiple when the ball stays in play.

# I broke the issue down into simpler pieces using a logic tree. The tree helped identify areas that could be addressed with targeted data analysis in search of root causes. I also used the tree to prioritize areas and make the best use of the allotted time. Rather than start with general data exploration looking for patterns, I focused my number crunching in specific areas. The figure below shows the logic tree with numbered circles to indicate each issue's assigned priority. I assigned priorities based on the potential impact and the availability of data for each opportunity.
# 
# 
# <span style="display:inline-block; margin-top:30px; margin-left:84px; font-size:12pt; font-weight:bold">Logic Tree</span>
# <img src="https://s3.amazonaws.com/nonwebstorage/headstrong/issuetree3.png" width="650">
# 
# 

# The following section describes my in-depth analysis of Opportunity Areas 1 and 2. My analysis in Opportunity Area 3 did not result in any recommendations. It is included in the Appendix. 
# <hr>

# <div class="h3"> IDENTIFYING CORRELATIONS AND CAUSES </div>
# 
# <div class="h4"><span style="color:#DD0000"> Opportunity Area 1. </span>  Fewer 'live' returns.</div>
# 
# For this opportunity I looked at the data to see if concussion rates differed by the various outcomes possible for a punt play. For instance, a punt that quickly goes out of bounds typically presents less risk to players than a long return.  I used the following classifications for punt outcomes:
#     
#   - Returned
#   - Fair Catch
#   - Touchback
#   - Downed
#   - Out of Bounds
#   - Not Punted (inlcudes blocks and fakes)
#   
# I classified each of the 6,681 punt plays using the play's text description provided in the data. Below is a sample of play descriptions:

# In[ ]:


descriptions = pd.read_csv('../input/NFL-Punt-Analytics-Competition/play_information.csv', 
                        usecols=['PlayDescription'], nrows=50).PlayDescription.tolist()
for i in range(0,45,15):
    display(HTML('<span style="color:steelblue">' + descriptions[i] + '</span>'))

# One can see from the first description that a single play in the data can actually be two separate plays as experienced by the players. Approximately 200 of 6,681 plays had complex descriptions. I classified these plays in a way that most accurately represented player risk. For example, if an out-of-bounds punt was invalidated by a penalty and then followed by a fair catch, I represented the play as a fair catch. After classifying the plays in this way I compared totals to the events found in the Next Gen Stats (NGS) data. The totals fell within 3% of each other for returns, fair catches, and touchbacks. Downed and out-of-bounds punts differed more, but not to an extent that changed the analysis or conclusions. 

# In[ ]:


#%% get preprocessed play data
plays_all = pd.read_parquet('../input/nfl-data-preparation/plays.parq')
plays_all.set_index(['GameKey', 'PlayID'], inplace=True)


#%% parse text
outcomes = pd.DataFrame({'PlayDescription': plays_all.PlayDescription}, 
                                index=plays_all.index)
punttypes = {"not_punted":    ["no play", "delay of game", 
                               "false start", "blocked", "incomplete"],
             "out_of_bounds": ["out of bounds"], 
             "downed":        ["downed"],        
             "touchback":     ["touchback"],
             "fair_catch":    ["fair catch"],
             "returned":      ["no gain", "for (.*) yard"]
             }       
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    for k,v in punttypes.items():
        outcomes[k] = outcomes.PlayDescription.str.contains('|'.join(v), 
                    case=False, regex=True).astype(int)


#%% correct for mulitple outcomes in one PlayID
outcomes['typesum'] = outcomes[list(punttypes.keys())].sum(axis=1)
outcomes.loc[outcomes.PlayDescription.str.contains("punts", case=False, 
            regex=False), 'not_punted'] = 0
outcomes.loc[~outcomes.PlayDescription.str.contains("punts", case=False, 
            regex=False), 'out_of_bounds':'returned'] = 0
outcomes.loc[~outcomes.PlayDescription.str.contains("punts", case=False, 
            regex=False), 'not_punted'] = 1


outcomes['typesum'] = outcomes[list(punttypes.keys())].sum(axis=1)
outcomes.loc[outcomes.typesum == 0, 'returned'] = 1
                        

outcomes['typesum'] = outcomes[list(punttypes.keys())].sum(axis=1)
outcomes.loc[(outcomes.PlayDescription.str.contains("invalid fair catch", 
            case=False, regex=False)) & (outcomes.returned == 1) & 
            (outcomes.typesum == 2), 'fair_catch'] = 0

outcomes['typesum'] = outcomes[list(punttypes.keys())].sum(axis=1)
outcomes.loc[(outcomes.PlayDescription.str.contains("punts", case=False, 
            regex=False)) & (outcomes.returned == 1) & (outcomes.typesum 
            == 2), 'not_punted':'out_of_bounds'] = 0

outcomes['typesum'] = outcomes[list(punttypes.keys())].sum(axis=1)
outcomes.loc[outcomes.PlayDescription.str.contains("blocked", case=False, 
            regex=False), 'out_of_bounds':'returned'] = 0

outcomes['typesum'] = outcomes[list(punttypes.keys())].sum(axis=1)
outcomes.loc[outcomes.typesum == 0, 'not_punted'] = 1 

outcomes['typesum'] = outcomes[list(punttypes.keys())].sum(axis=1)
outcomes.loc[(outcomes.touchback == 1) & (outcomes.typesum == 2), 
            'out_of_bounds':'downed'] = 0
outcomes.loc[(outcomes.returned == 1) & (outcomes.typesum == 2), 
            'returned'] = 0
outcomes.loc[(outcomes.fair_catch == 1) & (outcomes.typesum == 2), 
            'out_of_bounds':'downed'] = 0
outcomes.loc[(outcomes.downed == 1) & (outcomes.typesum == 2), 
            'out_of_bounds'] = 0

outcomes.drop(['PlayDescription', 'typesum'], axis=1, inplace=True)

plays_all['outcome'] = outcomes.dot(outcomes.columns).values #condense


#%% get yardage for return plays
plays_all['yardage'] = plays_all.PlayDescription.str\
            .extract("for (.{1,3}) yard")
plays_all.loc[plays_all.yardage.isnull(), 'yardage'] = 0
plays_all.loc[plays_all.outcome != "returned", 'yardage'] = 0

# The figures and table below show that returned punts are by far the most common outcome for both the absolute number of concussions and the percent of concussions for that play type. Fair catches, by contrast are low on both counts. 
# 
# 
# <span class="note"> <i>Hover over the bars to see exact numbers.</i> </span>

# In[ ]:


# format data for plotting
crosstable = pd.crosstab(plays_all.outcome, plays_all.concussion).reset_index()\
                    .sort_values(1, ascending=False)
crosstable.columns = ['Play_Outcome','Zero_Concussions', 'Concussions']
crosstable['Pct_of_Type'] = crosstable.Concussions/(crosstable.Concussions\
                + crosstable.Zero_Concussions)*100

bar_concs_all = crosstable.hvplot.bar('Play_Outcome', 'Concussions', ylim=(0,35), rot=45,
                yticks=np.arange(5,40,5).tolist(), width=400, height=300,
                  color="lightgray")
bar_concs_returned = crosstable[crosstable.Play_Outcome == 'returned'].hvplot\
                .bar('Play_Outcome', 'Concussions', title='Punt Concussions 2016-2017', 
                color='#ffa43d')

bar_pcts_all = crosstable.hvplot.bar('Play_Outcome', 'Pct_of_Type', ylim=(0,1.21), 
                rot=45, yticks=np.arange(0,1.4,0.2).tolist(), width=400, height=300,
                color="lightgray")
bar_pcts_returned = crosstable[crosstable.Play_Outcome == 'returned'].hvplot\
                .bar('Play_Outcome', 'Pct_of_Type', title='Punt Concussion Pcts 2016-2017', 
                color='#ffa43d')
                
display(bar_concs_all*bar_concs_returned + bar_pcts_all*bar_pcts_returned)
crosstable['Pct_of_Type'] = (crosstable.Pct_of_Type/100).apply('{:.2%}'.format)
crosstable['Play_Outcome'] = crosstable.Play_Outcome.str.title()
ctable = crosstable.sort_values('Concussions', ascending=False).set_index('Play_Outcome')

display(HTML('<span style="font-weight:bold">' + 'Comparison of Play Outcomes' + '</span>'), 
                ctable)


# The NFL would most likely see fewer concussions if some returned punts are shifted to other outcomes by changing the rules. This is consistent with my findings in the next section that over half of the recorded concussons occurred near the punt returner.
# 
# Another important consideration here was to see yards gained from returned punts. The figure below shows yards gained on each punt when a returner gets control of the ball with the intent to move forward. It does not include muffed catches. The vertical line in the figure indicates the median of all returns.

# In[ ]:


returns = plays_all.loc[(plays_all.outcome == "returned") & 
                        (~plays_all.PlayDescription.str.contains("MUFFS")), 
                        ['outcome', 'yardage', 'concussion', 'Rec_team']]\
                        .sort_values('Rec_team')
returns['yardage'] = returns.yardage.astype(int)
returns_median = returns.yardage.median()

# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib as mpl

sns.set(rc={'axes.facecolor':'darkseagreen'})
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    mpl.rcParams['figure.dpi'] = 600
    mpl.rcParams["figure.figsize"] = [9, 6.5]
    mpl.rcParams['ytick.labelsize'] = 10 
    ax2 = sns.boxplot(x=returns.yardage, y = returns.Rec_team,
                      fliersize=4, whis=20, color="#CCCCCC")
    
    for i,artist in enumerate(ax2.artists):
        artist.set_edgecolor("lightgray")
        for j in range(i*6,i*6+6):
            line = ax2.lines[j]
            line.set_color("gray")
            line.set_mfc("lightgray")
            line.set_mec("lightgray")
    sns.stripplot(x=returns.yardage, y = returns.Rec_team, alpha=0.9, size=3, jitter=True, 
            color="steelblue")
    ax2.set_title('Punt Return Yardage', size=14, loc='left')
    ax2.set_title('2016-2017', size=14, loc='right')
    ax2.set_xlim([-15, 100])
    ax2.xaxis.set_major_locator(ticker.MultipleLocator(10))
    
    ax2.set_xlabel("Yards Gained", size=10)
    ax2.set_ylabel("Receiving Team", size=10)
    plt.axvline(returns_median, linewidth=1,  linestyle='-', color='red')

# The above figure shows that most every team had returns falling largely in the 5-15 yard range. The median return for all teams, all returns was 8 yards. 
# 
# The table below shows the percentiles for five-yard intervals as reflected in the figure. For instance, 63% of all punt returns (meeting the criteria stated above) gained 10 yards or less. Returns over 20 yards, which account for most of the excitement on punt returns, occurred only 10% of the time.

# In[ ]:


yardlist =  [0, 5, 10, 15, 20]
cutpts = [stats.percentileofscore(returns.yardage, yd)/100 for yd in yardlist]

pdf = pd.DataFrame({'Yards_Gained': yardlist, 'Total_Pct_of_Returns': cutpts})
pdf['Yards_Gained'] = pdf.Yards_Gained.astype(str).str.cat(["Yards or Less"]*len(yardlist), sep=" ")
pdf.set_index('Yards_Gained', inplace=True)
pdf['Total_Pct_of_Returns'] = pdf.Total_Pct_of_Returns.apply('{:.0%}'.format)

display(HTML('<span style="font-weight:bold">' + 'Punt Return Percentages by Yards Gained'\
             + '</span>'), pdf)

# #### Recommendation
# 
# The above table shows that nearly 2/3 of all returned punts ended in gains of 10 yards or less. A full 1/3 resulted in gains of less than 5 yards. Based on these findings, I recommend changing the rules so that more punts result in a fair catch or other outcome. My recommendation is to spot the ball 10 yards forward of the point where a fair catch is made, adjusting for any penalties. The  intent is to provide a conservative but meaningful alternative to risking concussion in cases where it is highly likely that the runner will only gain a few yards. 
# 
# This recommendation and its potential impacts are detailed later in the report.
# 
# <hr class="light">

# <div class="h4"><span style="color:#DD0000"> Opportunity Area 2. </span>Less Harmful Contact.</div>
# 
# For this issue I looked at the types of collisions that occurred, where they occurred,  and the players involved. The figure below shows the numerous play flows that result in concussion. Players receiving a concussion appear to the left and primary partners appear to the right. The width of the flowpaths indicates the number of concussions occurring with that combination. For example, offensive linemen from the punting team received 9 concussions while tackling. 8 of those were while the returner was tackled, as one might expect. The 9th was from a defensive lineman blocking the offensive lineman during a tackle.
# 
# <span class="note"> Hover over flowpaths to see exact values. </span>

# In[ ]:


import holoviews as hv
from holoviews import opts

concussion_df = plays_all[plays_all.concussion == 1].copy()
# activities = concussion_df.Player_Activity_Derived.value_counts().reset_index()
# activities.columns = ['Player_Activity', 'Concussions']
# bar_activities = activities.hvplot.bar('Player_Activity', 'Concussions', ylim=(0,23), rot=0,
#                yticks=np.arange(0,25,5).tolist(), width=400, height=300,
#                color='lightgray', title='Concussions by Player Activity')

# with warnings.catch_warnings():
#     warnings.simplefilter("ignore")
#     concussion_df.replace('unspecified', 'Unclear', inplace=True, regex=False)
#     activity_combos = pd.crosstab(concussion_df.Player_Activity_Derived, 
#                     concussion_df.Primary_Partner_Activity_Derived)

#     blocks = activity_combos.loc['Blocked', 'Blocking'] + activity_combos.loc['Blocking', 
#                     'Blocked'] + activity_combos.loc['Blocked', 'Blocked']
#     tackles = activity_combos.loc['Tackled', 'Tackling'] + activity_combos.loc['Tackling',
#                     'Tackled'] + activity_combos.loc['Tackling', 'Tackling']
#     all_others = activity_combos.sum().sum() - tackles - blocks
#     act_types = ['Blocked_Blocking', 'Tackled_Tackling', 'All_Others']
#     acts_combo_df = pd.DataFrame({'Combined_Activity': act_types, 'Concussions': [blocks, 
#                     tackles, all_others]})
#     acts_combo_df.sort_values('Concussions', ascending=False, inplace=True)

    


concussion_df.Type_partner.replace("Unclear", "Unknown", inplace=True)
concussion_df.Type_player.replace("Unclear", "Unknown", inplace=True)
concussion_df['Player_Activity_Derived'] = concussion_df.Player_Activity_Derived + ['_']
concussion_df['Type_player'] = concussion_df.Type_player.str.title() + ['_']
concussion_df['Type_partner'] = concussion_df.Type_partner.str.title()


def make_sankey(colfrom, colto):
    conc_table = concussion_df[~concussion_df.Player_Activity_Derived.isnull()]\
                    .groupby([colfrom, colto])['PlayDescription'].size()\
                    .to_frame().reset_index()
    return conc_table.values

sankey_cols = ['Type_player','Player_Activity_Derived', 
                    'Primary_Partner_Activity_Derived', 'Type_partner']
sankey_list = []
for i in range(3):
    sankey_piece = make_sankey(sankey_cols[i], sankey_cols[i+1])
    sankey_list.append(sankey_piece)

sankey_table = np.concatenate(sankey_list)
c_sankey = hv.Sankey(sankey_table)
display(HTML('<span style="font-weight:bold; margin-left:84px">' \
                 + 'Concussion Roles and Activities' + '</span>'), c_sankey)

# The above diagram shows a variety of scenarios in which concussions occur. One can see by the green box to the left that 20 offensive linemen sustained concussions from tackling and being blocked. Although punt returners received the most concussions as a single-person role, offensive linemen collectively received over half the concussions during the two-season period. Gunners (typically two per play) also collectively received more concussions than most roles.

# I next reviewed the videos for all concussion plays along with NGS animations for select plays. The Next Gen Stats (NGS) data for 2016 and 2017 include the specific postion of each player during the play at intervals of 0.1 seconds, as well as the player's direction of travel and orientaton. The data also include the events that occurred on the field at each time interval. Below is a small excerpt from the 66 Million rows of data for punt plays.
# 

# In[ ]:


from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

ngs = pd.read_parquet('../input/nfl-data-preparation/NGS.parq').sort_values(['GSISID', 
        'Time']).reset_index(drop=True)
ngs_piece = ngs[4281:4285].copy()
ngs_piece['x'] = ngs_piece.x.apply('{:.2f}'.format)
ngs_piece['y'] = ngs_piece.y.apply('{:.2f}'.format)
ngs_piece['dis'] = ngs_piece.dis.apply('{:.3f}'.format)
ngs_piece

# NGS data can be used in many ways to include seeing how plays unfold over time. Below is an animation of an actual punt play in which a lineman from the punting team receives a blindside block from a fast-moving opponent as he closes in on the punt returner. The player receiving the concussion and the partner involved appear as triangles. The play description links to a video of the play for comparison. 
# 
# Full code for creating the animation is located in the [NFL Punt Play Animation](https://www.kaggle.com/jpmiller/nfl-punt-play-animation) kernel.

# In[ ]:


<a href="http://a.video.nfl.com//films/vodzilla/153321/Lechler_55_yd_punt-lG1K51rf-20181119_173634665_5000k.mp4"> 
    (2:57) (Punt formation) S.Lechler punts 48 yards to TEN 16, Center-J.Weeks. A.Jackson pushed ob at TEN 32 for 16 
    yards (J.Jenkins).
</a> 
<img src="https://s3.amazonaws.com/nonwebstorage/headstrong/animation_585_733_3.gif" width="650">

# It's unfortunate there's an injury on the play. On the plus side, however, notice how Jackson (the returner) puts a ninja move on No. 22 as he catches the ball!

# #### Recommendations
# I synthesized the various contact situations into scenarios with the highest potential for harm. For each situation I consider rules changes here in a general way. A detailed discussion and evaluation of recommendations appear later in the report.
# 
# *Offensive linemen and gunners blocked downfield.* As the ball is snapped, gunners work to outmaneuver the jammers and get downfield. Within seconds the ball is punted and offensive linemen also run down field, ahead of the defensive line to tackle the punt returner. Offensive linemen and gunners are at risk as defensive players make their way over and back to protect the returner. Players are usually moving at a high rate of speed and focused on the returner, which amplifies the risk. Players are at risk downfield as long as the ball remains in play.
# 
# I recommend expanding the current rule prohibiting illegal blindside blocks to include contact below the neck. 6 of the 10 downfield blocks leading to concussion occurred when an offensive player was moving toward the returner and got hit from the side or back. In at least 5 of these cases, the block occurred as a member of the receiving team moved toward or parallel to his own end line. However, contact is permitted below the neck and no penalties were assessed. 
# 
#    
# *Offensive linemen blocking at the line of scrimmage.* In punt plays, linebackers from the defense are often placed as linemen on the line of scrimmage. Their job is to aggressively rush the punter and block the punt. The offensive line often inlcudes tight ends and running backs with a job usually staffed by a much larger player. The offensive linemen are at risk during the first several seconds of every punt play.
#   
# I have no recommended rule change specifically related to concussions at the line of scrimmage.
# 
# 
# *Offensive linemen tackling returners.* Although a large proportion of concussions in 2016 and 2017 involved tackling, I have no additional recommendations at this time. The NFL added a rule for 2018 that brings a foul if a player lowers his head to initiate and make contact with his helmet against an opponent. The new rule may very well reduce concussions incurred when tackling.
# 

# <div class="h3"> Evaluating Proposed Changes </div>
# 
# There are many rules changes one can consider to reduce concussions associated with punts. A drastic but effective example would be to declare a dead ball wherever the punt lands, goes out of bounds, or is touched by the receiving team. Clearly this would change the nature of the punt play as we know it. I chose a more conservative approach that also took into account ease of implementation and preserving game integrity. The table below lists the potential rules changes I considered along with a summary evaluation.
# 
# #### Evaluation Summary of Proposed Changes
# <table class="rules">
# <tr>
# <th>Description</th><th>Reduces Concussions</th><th>Reduces Secondary Risks</th><th>Easy to Enforce</th>
# <th>Preserves Game Integrity</th>
# </tr>
#     
# <tr>
# <td>10-yard forward spot after fair catch</td>
# <td>++++</td>
# <td>+++</td>
# <td>+++</td>
# <td>++</td>
# </tr>
# 
# <tr>
# <td>Expand illegal blind-side blocks <br> to include contact below the neck</td>
# <td>++</td>
# <td>+++</td>
# <td>++</td>
# <td>+++</td>
# </tr>
# 
# </table>

# The summary ratings above resulted from the following considerations.
# 
# <b>Change 1. *Move the ball forward 10 yards after a Fair Catch.*</b> After fair catch of a scrimmage kick, play would start 10 yards forward from the spot of catch, or the succeeding spot after eforcement of penalties. This would apply when the receiving team elects to put the ball in play by a snap.
# 
#    - Fair catches had a concussion rate 1/8 that of live returns. Increasing the incentive for a fair catch could reduce risk to both punt returners and offensive linemen downfield. Punt returners may not be as pressured to take a hit in situations where the most likely outcome is at best a 5-10 yard return. Also the play will end sooner which reduces risk to offensive linemen getting hit by defensive linemen running to block for the return. The rule would additionally reduce the risk of tackling which accounted for another 13 concussions, although the new Use of Helmet rule may have already had an impact there.   
#   
#    - Secondary effects should be positive. If the punting team wants to prevent a fair catch they have to kick the ball out of bounds or for a touchback if possible. Both of these outcomes are safer than live returns. 
#     
#    - The rule is fairly easy to enforce. Officials already spot the ball after a fair catch according to the rules. With this change they would simply move the ball forward by 10 yards.
#     
#    - The rule would not present a large change to the game. 2/3 of all returned punts ended in a net gain of 10 yards or less so most returns would see no impact. From the fans' standpoint the change would amount to 2-3 more fair catches per game and possibly even more scoring as the receiving team gets better field position. A returner with room to run or exceptional talent could still run the ball and preserve the rare but exciting possibility of a long return. Also, punting strategies have evolved in recent years as leading punters come up with new ways to gain an advantage over the return team. This change could give them added incentive. 
#   
#   
# <b>Change 2. *Expand illegal blindside blocks to include forcible contact below the blocked player's neck.*</b> Article 7 lists players in a defenseless posture who are protected from prohibitive contact. The list includes players receiving a blindside block from a blocker whose path is toward or parallel his own end line. However, prohibited contact does not include forcible contact below the neck as long as it is not with the helmet and the blocker does not illegally launch. I recommend expanding the definition of prohibited contact for punt plays to include forcible contact to any part of a player's body, provide he meets the criteria for a defenseless player.
#    
#    - Blindside blocks accounted for at least 5 concussions during the 2016-2017 seasons. In most cases an offensive lineman was maneuvering to intercept a returner and was hit by an approaching defensive lineman moving at a higher rate of speed.
#     
#    - Secondary effects on safety should be positive. Blindside blocks are dangerous even when they are legal.
#     
#    - The rule is relatively easy to enforce. There would need to be additional training for officials, coaches and players as with any rule change. There may be a challenge in seeing all possible fouls and calling penalties given the fast action and chaotic nature of the play.
#     
#    - Emphasis on the rule would generally preserve game integrity. A potential downside is that more penalties would be assessed, slowing down the game.

# <a id='cn'></a>
# <div class="h2">  Conclusion </div>
# 
# <p style="margin-top: 20px">In this report I presented my problem solving methodology, analysis, and recommendations. I analyzed three specific areas of opportunity to reduce concussions for punt plays and developed two rules changes. The two changes presented have the potential to eliminate 8 concussions per year. This represents a 40% decrease in concussions from punt plays. The changes would improve player safety, be relatively easy to enforce, and have minimal impact to overall game integrity.</p>

# <a id='tm'></a>
# <div class="h2"> Qualifications </div>
# 
# <p style="margin-top: 20px"> I work as Principal Data Scientist at Benbrook Analytics, helping companies use data to make better decisions. My functional expertise includes predictive modeling, machine learning, artificial intelligence, and statistics. </p>
# 
# I graduated from MIT's Leaders for Global Operations program with Master’s degrees in Business and Engineering. I also have a BS from the United States Military Academy at West Point. Professional certifications include Microsoft Data Science Professional and Lean Six Sigma Master Black Belt.
# 
# I am currently ranked in the top 0.1% of all Kaggle contributors and ranked by Agilience as a Top 250 Authority in Machine Learning, Data Science, and Analytics.

# <a id='ap'></a>
# <div class="h2"> Appendix </div>
#   - Calculations for potential reduction of concussions
#   - Analysis of Opportunity Area 3, Safer Play Strategies
#   
#   <hr>
# 
# <p style="margin-top: 20px">The table below shows calculations and assumptions used to estimate the potential reduction in concussions made possible by the recommended rules changes.</p>
# 
# <img src="https://s3.amazonaws.com/nonwebstorage/headstrong/calcs.png" align="left">
# 
# <hr>

# <div class="h4"><span style="color:#DD0000"> Opportunity Area 3. </span> Safer play strategies.</div>
# *Note: I did not propose any rules changes after analyzing Opportunity Area 3. It is possible that additional exploration could result in insights to play strategy.*
# 
# 
# For this opportunity I looked at the data to see how concussion incidents correlated to the play strategies followed by teams. Special teams strategies have many layers and elements. I focused on the relatively few elements where we had data. I captured the following factors:
# 
#   - Distance from scrimmage line to goal
#   - Score difference  
#   - Formation of each team at line-up
#   - Location of players on the field during the play
#   - Player speeds
#   - Punt distance
# 
# I relied heavily on the NGS data for this part of the analysis. 

# I used the NGS data along with other provided data to analyze the factors listed above. NGS Data was aggregated first for each player, and then for each play. I applied a technique called t-SNE to the aggregated data as a form of unsupervised machine learning. t_SNE compares the factors given and groups like points together in a simple plot.
# 
# The figure below represents a spatial map, or projection, of the various factors onto a 2-D scatter plot. The figure shows  punt plays with different characteristics. There are three groups separate from the main mass, appearing as "islands" of points above, right, and below the main mass of points. These are most likely due to the different scenarios that occur during a punt play. For example, a short punt near the goal line looks different than a 60-yard boomer. Formations, the speed of players, and relative location will typically be different.
# 
# Plays in which a concussion occurred are shown in orange. To gain insight on what separates plays with concussion events from those without, one would need to see areas of the map without orange dots. In this case the dots are found across the map, including within the separated island areas. 
# 
# <span class="note">Hover over points to see play descriptions.</span>

# In[ ]:


x_dist = ngs.x - ngs.x.shift(-1), ngs.y - ngs.y.shift(-1)
ngs['speed'] = 10*np.hypot(ngs.x - ngs.x.shift(-1), ngs.y - ngs.y.shift(-1))

# get player-level agg
aggdict_player = {'Time': ['size'],
                  'x': ['mean', 'max', 'min', 'var'],
                  'y': ['mean', 'max', 'min', 'var'],
                  'speed': ['mean', 'max']}
ngs_agg = ngs.groupby(['GameKey', 'PlayID', 'GSISID']).agg(aggdict_player)
ngs_agg.columns = [n[0] + '_' + n[1] for n in ngs_agg.columns]
ngs_agg


# get play-level agg
aggdict_play = {'x_mean': ['mean', 'var'],
                'x_max': ['max', 'var'],
               'x_min': ['min'],
                'x_var': ['max', 'mean'],
                'y_mean': ['mean', 'var'],
                'y_max': ['max', 'var'],
               'y_min': ['min'],
                'y_var': ['max', 'mean'],
                'speed_mean': ['mean', 'var'],
                'speed_max': ['max', 'var']}
ngs_agg = ngs_agg.groupby(['GameKey', 'PlayID']).agg(aggdict_play)
ngs_agg.columns = [n[0] + '_' + n[1] for n in ngs_agg.columns]

plays_all['points_ahead'] = np.where(plays_all.Poss_Team == 
        plays_all.HomeTeamCode, plays_all.home_score - plays_all.visit_score,
        plays_all.visit_score - plays_all.home_score)

play_cols = ['Type_dlineman_agg',
             'Type_fullback_agg',
             'Type_gunner_agg',
             'Type_jammer_agg',
             'Type_linebacker_agg',
             'Type_olineman_agg',
             'Type_protector_agg',
             'Type_punter_agg', 
             'dist_togoal',
             'concussion',
             'yardage',
             'points_ahead',
             'PlayDescription']

plays_strategy = plays_all[play_cols]

ngs_agg2 =  ngs_agg.join(plays_strategy, how="inner").replace(-99, -1)
ngs_agg2['yardage'] = pd.to_numeric(ngs_agg2.yardage)

targets = ngs_agg2[['concussion', 'PlayDescription']]
ngs_agg2.drop(['concussion', 'PlayDescription'], axis=1, inplace=True)

ngs_scaled = StandardScaler().fit_transform(ngs_agg2.fillna(0).values)

tsne = TSNE(n_components=2, perplexity=30.0, verbose=1, 
        learning_rate=50, random_state=222)

ngs_emb = tsne.fit_transform(ngs_scaled)



# In[ ]:


strategy = pd.DataFrame(ngs_emb, columns=['x', 'y'])
strat_annotated = pd.concat([strategy, ngs_agg2.reset_index(), targets.reset_index(drop=True)], axis=1)

no_conc = strat_annotated[strat_annotated.concussion == 0].hvplot.scatter('x', 'y', 
                     alpha=0.4, size=5, grid=True, hover_cols=['PlayDescription'], 
                     height=550, width=700)
yes_conc = strat_annotated[strat_annotated.concussion == 1].hvplot.scatter('x', 'y', 
                     hover_cols=['PlayDescription'], size=30)
no_conc*yes_conc

# I concluded that based on the factors listed above, there was nothing about specific punt strategies that seemed to make one type of punt play noticably more dangerous than another. Differences in concussion risk appeared to be more related to the outcome of the punt (returned, fair catch, etc.) A more advanced model based on Recurrent Neural Networks may prove better at detecting patterns across each point in the time interval of the play. 
