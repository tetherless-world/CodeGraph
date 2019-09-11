#!/usr/bin/env python
# coding: utf-8

# > Please go through Giba's post and kernel  to underrstand what this leak is all about
# > https://www.kaggle.com/titericz/the-property-by-giba (kernel)
# > https://www.kaggle.com/c/santander-value-prediction-challenge/discussion/61329 (post)
# > 
# > Also, go through this Jiazhen's kernel which finds more columns to exploit leak
# > https://www.kaggle.com/johnfarrell/giba-s-property-extended-result
# > 
# > I just exploit data property in brute force way and then fill in remaining by row non zero means! This should bring everyone on level-playing field.
# > 
# > **Let the competition begin! :D**
# 
# > ### Just some small modifications from [original baseline](https://www.kaggle.com/tezdhar/breaking-lb-fresh-start)~
# >- The leak rows are calculated separately on train/test set
# >- Calculated the leaky values, correctness, for each lag
# >- Hope this can help to do some *lag_selection*
# 
# >### Update leak process codes to Dmitry Frumkin's *fast* [version](https://www.kaggle.com/dfrumkin/a-simple-way-to-use-giba-s-features-v2)
# >- The result of Dmitry's original function and result of Hasan's function seem slightly different
# >- Modified to make the output consistent with Hasan's function (Seems better score)

# Updated this Kernel by Jiazhen Xi with some of the new patterns I found. Using 6 of the 40 those patterns. Hope this helps everyone and provide them with some sort of help in this dying stage of the competition.

# In[ ]:


import warnings
warnings.filterwarnings('ignore')

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))

import lightgbm as lgb
from sklearn.model_selection import *
from sklearn.metrics import mean_squared_error, make_scorer
from scipy.stats import mode, skew, kurtosis, entropy
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt
import seaborn as sns

import dask.dataframe as dd
from dask.multiprocessing import get

from tqdm import tqdm, tqdm_notebook
tqdm.pandas(tqdm_notebook)

# Any results you write to the current directory are saved as output.

# In[ ]:


train = pd.read_csv("../input/santander-value-prediction-challenge/train.csv")
test = pd.read_csv("../input/santander-value-prediction-challenge/test.csv")

transact_cols = [f for f in train.columns if f not in ["ID", "target"]]
y = np.log1p(train["target"]).values

# In[ ]:


test["target"] = train["target"].mean()

# In[ ]:


test.shape

# We take time series columns from [here](https://www.kaggle.com/johnfarrell/giba-s-property-extended-result)

# In[ ]:


all_df = pd.concat([train, test]).reset_index(drop=True)
all_df.columns = all_df.columns.astype(str)
print(all_df.shape)

# In[ ]:



cols = ['f190486d6', '58e2e02e6', 'eeb9cd3aa', '9fd594eec', '6eef030c1', '15ace8c9f', 
        'fb0f5dbfe', '58e056e12', '20aa07010', '024c577b9', 'd6bb78916', 'b43a7cfd5', 
        '58232a6fb', '1702b5bf0', '324921c7b', '62e59a501', '2ec5b290f', '241f0f867', 
        'fb49e4212', '66ace2992', 'f74e8f13d', '5c6487af1', '963a49cdc', '26fc93eb7', 
        '1931ccfdd', '703885424', '70feb1494', '491b9ee45', '23310aa6f', 'e176a204a', 
        '6619d81fc', '1db387535', 
        'fc99f9426', '91f701ba2', '0572565c2', '190db8488', 'adb64ff71', 'c47340d97', 'c5a231d81', '0ff32eb98'
       ]


# Now adding groups suggested by @armamut

# In[ ]:


colgroups = [
    ['f190486d6', '58e2e02e6', 'eeb9cd3aa', '9fd594eec', '6eef030c1', '15ace8c9f', 'fb0f5dbfe', '58e056e12', '20aa07010', '024c577b9', 'd6bb78916', 'b43a7cfd5', '58232a6fb', '1702b5bf0', '324921c7b', '62e59a501', '2ec5b290f', '241f0f867', 'fb49e4212', '66ace2992', 'f74e8f13d', '5c6487af1', '963a49cdc', '26fc93eb7', '1931ccfdd', '703885424', '70feb1494', '491b9ee45', '23310aa6f', 'e176a204a', '6619d81fc', '1db387535', 'fc99f9426', '91f701ba2', '0572565c2', '190db8488', 'adb64ff71', 'c47340d97', 'c5a231d81', '0ff32eb98'],
    ['266525925', '4b6dfc880', '2cff4bf0c', 'a3382e205', '6488c8200', '547d3135b', 'b46191036', '453128993', '2599a7eb7', '2fc60d4d9', '009319104', 'de14e7687', 'aa31dd768', '2b54cddfd', 'a67d02050', '37aab1168', '939cc02f5', '31f72667c', '6f951302c', '54723be01', '4681de4fd', '8bd53906a', '435f27009', 'f82167572', 'd428161d9', '9015ac21d', 'ec4dc7883', '22c7b00ef', 'd4cc42c3d', '1351bf96e', '1e8801477', 'b7d59d3b5', 'a459b5f7d', '580f5ff06', '39b3c553a', '1eec37deb', '692c44993', 'ce8ce671e', '88ef1d9a8', 'bf042d928'],
    ['9d5c7cb94', '197cb48af', 'ea4887e6b', 'e1d0e11b5', 'ac30af84a', 'ba4ceabc5', 'd4c1de0e2', '6d2ece683', '9c42bff81', 'cf488d633', '0e1f6696a', 'c8fdf5cbf', 'f14b57b8f', '3a62b36bd', 'aeff360c7', '64534cc93', 'e4159c59e', '429687d5a', 'c671db79e', 'd79736965', '2570e2ba9', '415094079', 'ddea5dc65', 'e43343256', '578eda8e0', 'f9847e9fe', '097c7841e', '018ab6a80', '95aea9233', '7121c40ee', '578b81a77', '96b6bd42b', '44cb9b7c4', '6192f193d', 'ba136ae3f', '8479174c2', '64dd02e44', '4ecc3f505', 'acc4a8e68', '994b946ad'],
    ['f1eeb56ae', '62ffce458', '497adaff8', 'ed1d5d137', 'faf7285a1', 'd83da5921', '0231f07ed', '7950f4c11', '051410e3d', '39e1796ab', '2e0148f29', '312832f30', '6f113540d', 'f3ee6ba3c', 'd9fc63fa1', '6a0b386ac', '5747a79a9', '64bf3a12a', 'c110ee2b7', '1bf37b3e2', 'fdd07cac1', '0872fe14d', 'ddef5ad30', '42088cf50', '3519bf4a4', 'a79b1f060', '97cc1b416', 'b2790ef54', '1a7de209c', '2a71f4027', 'f118f693a', '15e8a9331', '0c545307d', '363713112', '73e591019', '21af91e9b', '62a915028', '2ab5a56f5', 'a8ee55662', '316b978cd'],
    ['b26d16167', '930f989bf', 'ca58e6370', 'aebe1ea16', '03c589fd7', '600ea672f', '9509f66b0', '70f4f1129', 'b0095ae64', '1c62e29a7', '32a0342e2', '2fc5bfa65', '09c81e679', '49e68fdb9', '026ca57fd', 'aacffd2f4', '61483a9da', '227ff4085', '29725e10e', '5878b703c', '50a0d7f71', '0d1af7370', '7c1af7bbb', '4bf056f35', '3dd64f4c4', 'b9f75e4aa', '423058dba', '150dc0956', 'adf119b9a', 'a8110109e', '6c4f594e0', 'c44348d76', 'db027dbaf', '1fcba48d0', '8d12d44e1', '8d13d891d', '6ff9b1760', '482715cbd', 'f81c2f1dd', 'dda820122'],
    ['c928b4b74', '8e4d0fe45', '6c0e0801a', '02861e414', 'aac52d8d9', '041c5d0c9', 'd7875bb6c', 'e7c0cfd0f', 'd48c08bda', '0c9462c08', '57dd44c29', 'a93118262', '850027e38', 'db3839ab0', '27461b158', '32174174c', '9306da53f', '95742c2bf', '5831f4c76', '1e6306c7c', '06393096a', '13bdd610a', 'd7d314edc', '9a07d7b1f', '4d2671746', '822e49b95', '3c8a3ced0', '83635fb67', '1857fbccf', 'c4972742d', 'b6c0969a2', 'e78e3031b', '36a9a8479', 'e79e5f72c', '092271eb3', '74d7f2dc3', '277ef93fc', 'b30e932ba', '8f57141ec', '350473311'],
    ['06148867b', '4ec3bfda8', 'a9ca6c2f4', 'bb0408d98', '1010d7174', 'f8a437c00', '74a7b9e4a', 'cfd55f2b6', '632fed345', '518b5da24', '60a5b79e4', '3fa0b1c53', 'e769ee40d', '9f5f58e61', '83e3e2e60', '77fa93749', '3c9db4778', '42ed6824a', '761b8e0ec', 'ee7fb1067', '71f5ab59f', '177993dc6', '07df9f30c', 'b1c5346c4', '9a5cd5171', 'b5df42e10', 'c91a4f722', 'd93058147', '20a325694', 'f5e0f4a16', '5edd220bc', 'c901e7df1', 'b02dfb243', 'bca395b73', '1791b43b0', 'f04f0582d', 'e585cbf20', '03055cc36', 'd7f15a3ad', 'ccd9fc164'],
    ['df838756c', '2cb73ede7', '4dcf81d65', '61c1b7eb6', 'a9f61cf27', '1af4d24fa', 'e13b0c0aa', 'b9ba17eb6', '796c218e8', '37f57824c', 'd1e0f571b', 'f9e3b03b7', 'a3ef69ad5', 'e16a20511', '04b88be38', '99e779ee0', '9f7b782ac', '1dd7bca9f', '2eeadde2b', '6df033973', 'cdfc2b069', '031490e77', '5324862e4', '467bee277', 'a3fb07bfd', '64c6eb1cb', '8618bc1fd', '6b795a2bc', '956d228b9', '949ed0965', 'a4511cb0b', 'b64425521', '2e3c96323', '191e21b5f', 'bee629024', '1977eaf08', '5e645a169', '1d04efde3', '8675bec0b', '8337d1adc'],
    ['a1cd7b681', '9b490abb3', 'b10f15193', '05f54f417', 'a7ac690a8', 'ed6c300c2', 'd0803e3a1', 'b1bb8eac3', 'bd1c19973', 'a34f8d443', '84ec1e3db', '24018f832', '82e01a220', '4c2064b00', '0397f7c9b', 'ba42e41fa', '22d7ad48d', '9abffd22c', 'dbfa2b77f', '2c6c62b54', '9fa38def3', 'ecb354edf', '9c3154ae6', '2f26d70f4', '53102b93f', 'a36b95f78', '1fa0f78d0', '19915a6d3', 'c944a48b5', '482b04cba', '2ce77a58f', '86558e595', 'c3f400e36', '20305585c', 'f8ccfa064', 'dd771cb8e', '9aa27017e', 'cd7f0affd', '236cc1ff5', 'a3fc511cd'],
    ['920a04ee2', '93efdb50f', '15ea45005', '78c57d7cd', '91570fb11', 'c5dacc85b', '145c7b018', '590b24ab1', 'c283d4609', 'e8bd579ae', '7298ca1ef', 'ce53d1a35', 'a8f80f111', '2a9fed806', 'feb40ad9f', 'cfd255ee3', '31015eaab', '303572ae2', 'cd15bb515', 'cb5161856', 'a65b73c87', '71d64e3f7', 'ec5fb550f', '4af2493b6', '18b4fa3f5', '3d655b0ed', '5cc9b6615', '88c0ec0a6', '8722f33bb', '5ed0c24d0', '54f26ee08', '04ecdcbb3', 'ade8a5a19', 'd5efae759', 'ac7a97382', 'e1b20c3a6', 'b0fcfeab8', '438b8b599', '43782ef36', 'df69cf626'],
    ['50603ae3d', '48282f315', '090dfb7e2', '6ccaaf2d7', '1bf2dfd4a', '50b1dd40f', '1604c0735', 'e94c03517', 'f9378f7ef', '65266ad22', 'ac61229b6', 'f5723deba', '1ced7f0b4', 'b9a4f06cd', '8132d18b8', 'df28ac53d', 'ae825156f', '936dc3bc4', '5b233cf72', '95a2e29fc', '882a3da34', '2cb4d123e', '0e1921717', 'c83d6b24d', '90a2428a5', '67e6c62b9', '320931ca8', '900045349', 'bf89fac56', 'da3b0b5bb', 'f06078487', '56896bb36', 'a79522786', '71c2f04c9', '1af96abeb', '4b1a994cc', 'dee843499', '645b47cde', 'a8e15505d', 'cc9c2fc87'],
    ['b6daeae32', '3bdee45be', '3d6d38290', '5a1589f1a', '961b91fe7', '29c059dd2', 'cfc1ce276', '0a953f97e', '30b3daec2', 'fb5f5836e', 'c7525612c', '6fa35fbba', '72d34a148', 'dcc269cfe', 'bdf773176', '469630e5c', '23db7d793', 'dc10234ae', '5ac278422', '6cf7866c1', 'a39758dae', '45f6d00da', '251d1aa17', '84d9d1228', 'b98f3e0d7', '66146c12d', 'd6470c4ce', '3f4a39818', 'f16a196c6', 'b8f892930', '6f88afe65', 'ed8951a75', '371da7669', '4b9540ab3', '230a025ca', 'f8cd9ae02', 'de4e75360', '540cc3cd1', '7623d805a', 'c2dae3a5a'],
    ['d0d340214', '34d3715d5', '9c404d218', 'c624e6627', 'a1b169a3a', 'c144a70b1', 'b36a21d49', 'dfcf7c0fa', 'c63b4a070', '43ebb15de', '1f2a670dd', '3f07a4581', '0b1560062', 'e9f588de5', '65d14abf0', '9ed0e6ddb', '0b790ba3a', '9e89978e3', 'ee6264d2b', 'c86c0565e', '4de164057', '87ba924b1', '4d05e2995', '2c0babb55', 'e9375ad86', '8988e8da5', '8a1b76aaf', '724b993fd', '654dd8a3b', 'f423cf205', '3b54cc2cf', 'e04141e42', 'cacc1edae', '314396b31', '2c339d4f2', '3f8614071', '16d1d6204', '80b6e9a8b', 'a84cbdab5', '1a6d13c4a'],
    ['a9819bda9', 'ea26c7fe6', '3a89d003b', '1029d9146', '759c9e85d', '1f71b76c1', '854e37761', '56cb93fd8', '946d16369', '33e4f9a0e', '5a6a1ec1a', '4c835bd02', 'b3abb64d2', 'fe0dd1a15', 'de63b3487', 'c059f2574', 'e36687647', 'd58172aef', 'd746efbfe', 'ccf6632e6', 'f1c272f04', 'da7f4b066', '3a7771f56', '5807de036', 'b22eb2036', 'b77c707ef', 'e4e9c8cc6', 'ff3b49c1d', '800f38b6b', '9a1d8054b', '0c9b00a91', 'fe28836c3', '1f8415d03', '6a542a40a', 'd53d64307', 'e700276a2', 'bb6f50464', '988518e2d', 'f0eb7b98f', 'd7447b2c5'],
    ['87ffda550', '63c094ba4', '2e103d632', '1c71183bb', 'd5fa73ead', 'e078302ef', 'a6b6bc34a', 'f6eba969e', '0d51722ca', 'ce3d7595b', '6c5c8869c', 'dfd179071', '122c135ed', 'b4cfe861f', 'b7c931383', '44d5b820f', '4bcf15776', '51d4053c7', '1fe5d56b9', 'ea772e115', 'ad009c8b9', '68a945b18', '62fb56487', 'c10f31664', 'cbb673163', 'c8d582dd2', '8781e4b91', 'bd6da0cca', 'ca2b906e8', '11e12dbe8', 'bb0ce54e9', 'c0d2348b7', '77deffdf0', 'f97d9431e', 'a09a238d0', '935ca66a9', '9de83dc23', '861076e21', 'f02ecb19c', '166008929'],
    ['f3cf9341c', 'fa11da6df', 'd47c58fe2', '0d5215715', '555f18bd3', '134ac90df', '716e7d74d', 'c00611668', '1bf8c2597', '1f6b2bafa', '174edf08a', 'f1851d155', '5bc7ab64f', 'a61aa00b0', 'b2e82c050', '26417dec4', '53a550111', '51707c671', 'e8d9394a0', 'cbbc9c431', '6b119d8ce', 'f296082ec', 'be2e15279', '698d05d29', '38e6f8d32', '93ca30057', '7af000ac2', '1fd0a1f2a', '41bc25fef', '0df1d7b9a', '88d29cfaf', '2b2b5187e', 'bf59c51c3', 'cfe749e26', 'ad207f7bb', '11114a47a', '341daa7d1', 'a8dd5cea5', '7b672b310', 'b88e5de84']
]

# In[ ]:


colgps = [['f190486d6','58e2e02e6','eeb9cd3aa','9fd594eec','6eef030c1','15ace8c9f','fb0f5dbfe','58e056e12','20aa07010','024c577b9','d6bb78916','b43a7cfd5','58232a6fb','1702b5bf0','324921c7b','62e59a501','2ec5b290f','241f0f867','fb49e4212','66ace2992','f74e8f13d','5c6487af1','963a49cdc','26fc93eb7','1931ccfdd','703885424','70feb1494','491b9ee45','23310aa6f','e176a204a','6619d81fc','1db387535','fc99f9426','91f701ba2','0572565c2','190db8488','adb64ff71','c47340d97','c5a231d81','0ff32eb98']
, ['266525925','4b6dfc880','2cff4bf0c','a3382e205','6488c8200','547d3135b','b46191036','453128993','2599a7eb7','2fc60d4d9','009319104','de14e7687','aa31dd768','2b54cddfd','a67d02050','37aab1168','939cc02f5','31f72667c','6f951302c','54723be01','4681de4fd','8bd53906a','435f27009','f82167572','d428161d9','9015ac21d','ec4dc7883','22c7b00ef','d4cc42c3d','1351bf96e','1e8801477','b7d59d3b5','a459b5f7d','580f5ff06','39b3c553a','1eec37deb','692c44993','ce8ce671e','88ef1d9a8','bf042d928']
, ['9d5c7cb94','197cb48af','ea4887e6b','e1d0e11b5','ac30af84a','ba4ceabc5','d4c1de0e2','6d2ece683','9c42bff81','cf488d633','0e1f6696a','c8fdf5cbf','f14b57b8f','3a62b36bd','aeff360c7','64534cc93','e4159c59e','429687d5a','c671db79e','d79736965','2570e2ba9','415094079','ddea5dc65','e43343256','578eda8e0','f9847e9fe','097c7841e','018ab6a80','95aea9233','7121c40ee','578b81a77','96b6bd42b','44cb9b7c4','6192f193d','ba136ae3f','8479174c2','64dd02e44','4ecc3f505','acc4a8e68','994b946ad']
, ['f1eeb56ae','62ffce458','497adaff8','ed1d5d137','faf7285a1','d83da5921','0231f07ed','7950f4c11','051410e3d','39e1796ab','2e0148f29','312832f30','6f113540d','f3ee6ba3c','d9fc63fa1','6a0b386ac','5747a79a9','64bf3a12a','c110ee2b7','1bf37b3e2','fdd07cac1','0872fe14d','ddef5ad30','42088cf50','3519bf4a4','a79b1f060','97cc1b416','b2790ef54','1a7de209c','2a71f4027','f118f693a','15e8a9331','0c545307d','363713112','73e591019','21af91e9b','62a915028','2ab5a56f5','a8ee55662','316b978cd']
, ['b26d16167','930f989bf','ca58e6370','aebe1ea16','03c589fd7','600ea672f','9509f66b0','70f4f1129','b0095ae64','1c62e29a7','32a0342e2','2fc5bfa65','09c81e679','49e68fdb9','026ca57fd','aacffd2f4','61483a9da','227ff4085','29725e10e','5878b703c','50a0d7f71','0d1af7370','7c1af7bbb','4bf056f35','3dd64f4c4','b9f75e4aa','423058dba','150dc0956','adf119b9a','a8110109e','6c4f594e0','c44348d76','db027dbaf','1fcba48d0','8d12d44e1','8d13d891d','6ff9b1760','482715cbd','f81c2f1dd','dda820122']
, ['c928b4b74','8e4d0fe45','6c0e0801a','02861e414','aac52d8d9','041c5d0c9','d7875bb6c','e7c0cfd0f','d48c08bda','0c9462c08','57dd44c29','a93118262','850027e38','db3839ab0','27461b158','32174174c','9306da53f','95742c2bf','5831f4c76','1e6306c7c','06393096a','13bdd610a','d7d314edc','9a07d7b1f','4d2671746','822e49b95','3c8a3ced0','83635fb67','1857fbccf','c4972742d','b6c0969a2','e78e3031b','36a9a8479','e79e5f72c','092271eb3','74d7f2dc3','277ef93fc','b30e932ba','8f57141ec','350473311']
, ['06148867b','4ec3bfda8','a9ca6c2f4','bb0408d98','1010d7174','f8a437c00','74a7b9e4a','cfd55f2b6','632fed345','518b5da24','60a5b79e4','3fa0b1c53','e769ee40d','9f5f58e61','83e3e2e60','77fa93749','3c9db4778','42ed6824a','761b8e0ec','ee7fb1067','71f5ab59f','177993dc6','07df9f30c','b1c5346c4','9a5cd5171','b5df42e10','c91a4f722','d93058147','20a325694','f5e0f4a16','5edd220bc','c901e7df1','b02dfb243','bca395b73','1791b43b0','f04f0582d','e585cbf20','03055cc36','d7f15a3ad','ccd9fc164']
, ['df838756c','2cb73ede7','4dcf81d65','61c1b7eb6','a9f61cf27','1af4d24fa','e13b0c0aa','b9ba17eb6','796c218e8','37f57824c','d1e0f571b','f9e3b03b7','a3ef69ad5','e16a20511','04b88be38','99e779ee0','9f7b782ac','1dd7bca9f','2eeadde2b','6df033973','cdfc2b069','031490e77','5324862e4','467bee277','a3fb07bfd','64c6eb1cb','8618bc1fd','6b795a2bc','956d228b9','949ed0965','a4511cb0b','b64425521','2e3c96323','191e21b5f','bee629024','1977eaf08','5e645a169','1d04efde3','8675bec0b','8337d1adc']
, ['a1cd7b681','9b490abb3','b10f15193','05f54f417','a7ac690a8','ed6c300c2','d0803e3a1','b1bb8eac3','bd1c19973','a34f8d443','84ec1e3db','24018f832','82e01a220','4c2064b00','0397f7c9b','ba42e41fa','22d7ad48d','9abffd22c','dbfa2b77f','2c6c62b54','9fa38def3','ecb354edf','9c3154ae6','2f26d70f4','53102b93f','a36b95f78','1fa0f78d0','19915a6d3','c944a48b5','482b04cba','2ce77a58f','86558e595','c3f400e36','20305585c','f8ccfa064','dd771cb8e','9aa27017e','cd7f0affd','236cc1ff5','a3fc511cd']
, ['920a04ee2','93efdb50f','15ea45005','78c57d7cd','91570fb11','c5dacc85b','145c7b018','590b24ab1','c283d4609','e8bd579ae','7298ca1ef','ce53d1a35','a8f80f111','2a9fed806','feb40ad9f','cfd255ee3','31015eaab','303572ae2','cd15bb515','cb5161856','a65b73c87','71d64e3f7','ec5fb550f','4af2493b6','18b4fa3f5','3d655b0ed','5cc9b6615','88c0ec0a6','8722f33bb','5ed0c24d0','54f26ee08','04ecdcbb3','ade8a5a19','d5efae759','ac7a97382','e1b20c3a6','b0fcfeab8','438b8b599','43782ef36','df69cf626']
, ['50603ae3d','48282f315','090dfb7e2','6ccaaf2d7','1bf2dfd4a','50b1dd40f','1604c0735','e94c03517','f9378f7ef','65266ad22','ac61229b6','f5723deba','1ced7f0b4','b9a4f06cd','8132d18b8','df28ac53d','ae825156f','936dc3bc4','5b233cf72','95a2e29fc','882a3da34','2cb4d123e','0e1921717','c83d6b24d','90a2428a5','67e6c62b9','320931ca8','900045349','bf89fac56','da3b0b5bb','f06078487','56896bb36','a79522786','71c2f04c9','1af96abeb','4b1a994cc','dee843499','645b47cde','a8e15505d','cc9c2fc87']
, ['b6daeae32','3bdee45be','3d6d38290','5a1589f1a','961b91fe7','29c059dd2','cfc1ce276','0a953f97e','30b3daec2','fb5f5836e','c7525612c','6fa35fbba','72d34a148','dcc269cfe','bdf773176','469630e5c','23db7d793','dc10234ae','5ac278422','6cf7866c1','a39758dae','45f6d00da','251d1aa17','84d9d1228','b98f3e0d7','66146c12d','d6470c4ce','3f4a39818','f16a196c6','b8f892930','6f88afe65','ed8951a75','371da7669','4b9540ab3','230a025ca','f8cd9ae02','de4e75360','540cc3cd1','7623d805a','c2dae3a5a']
, ['d0d340214','34d3715d5','9c404d218','c624e6627','a1b169a3a','c144a70b1','b36a21d49','dfcf7c0fa','c63b4a070','43ebb15de','1f2a670dd','3f07a4581','0b1560062','e9f588de5','65d14abf0','9ed0e6ddb','0b790ba3a','9e89978e3','ee6264d2b','c86c0565e','4de164057','87ba924b1','4d05e2995','2c0babb55','e9375ad86','8988e8da5','8a1b76aaf','724b993fd','654dd8a3b','f423cf205','3b54cc2cf','e04141e42','cacc1edae','314396b31','2c339d4f2','3f8614071','16d1d6204','80b6e9a8b','a84cbdab5','1a6d13c4a']
, ['a9819bda9','ea26c7fe6','3a89d003b','1029d9146','759c9e85d','1f71b76c1','854e37761','56cb93fd8','946d16369','33e4f9a0e','5a6a1ec1a','4c835bd02','b3abb64d2','fe0dd1a15','de63b3487','c059f2574','e36687647','d58172aef','d746efbfe','ccf6632e6','f1c272f04','da7f4b066','3a7771f56','5807de036','b22eb2036','b77c707ef','e4e9c8cc6','ff3b49c1d','800f38b6b','9a1d8054b','0c9b00a91','fe28836c3','1f8415d03','6a542a40a','d53d64307','e700276a2','bb6f50464','988518e2d','f0eb7b98f','d7447b2c5']
, ['87ffda550','63c094ba4','2e103d632','1c71183bb','d5fa73ead','e078302ef','a6b6bc34a','f6eba969e','0d51722ca','ce3d7595b','6c5c8869c','dfd179071','122c135ed','b4cfe861f','b7c931383','44d5b820f','4bcf15776','51d4053c7','1fe5d56b9','ea772e115','ad009c8b9','68a945b18','62fb56487','c10f31664','cbb673163','c8d582dd2','8781e4b91','bd6da0cca','ca2b906e8','11e12dbe8','bb0ce54e9','c0d2348b7','77deffdf0','f97d9431e','a09a238d0','935ca66a9','9de83dc23','861076e21','f02ecb19c','166008929']
, ['f3cf9341c','fa11da6df','d47c58fe2','0d5215715','555f18bd3','134ac90df','716e7d74d','c00611668','1bf8c2597','1f6b2bafa','174edf08a','f1851d155','5bc7ab64f','a61aa00b0','b2e82c050','26417dec4','53a550111','51707c671','e8d9394a0','cbbc9c431','6b119d8ce','f296082ec','be2e15279','698d05d29','38e6f8d32','93ca30057','7af000ac2','1fd0a1f2a','41bc25fef','0df1d7b9a','88d29cfaf','2b2b5187e','bf59c51c3','cfe749e26','ad207f7bb','11114a47a','341daa7d1','a8dd5cea5','7b672b310','b88e5de84']
, ['6bf90e4f5','ff96b95eb','dd2fc27b0','108fe14ef','5dd6254bb','9133963bd','bb9aefe00','c75604498','02c1bd442','31e434a58','8516b48f5','f3250c691','15960e710','d9ce16f1c','ebbd8432c','2e768f672','8c2252655','40e17d035','c8ebd62ea','ddfc3e604','0494ca73d','6bb4be4f2','9ca9e4916','0cb171797','fe56ddf0f','07b31de85','4d5a5e150','fbc6c2b78','fa977f17b','2ad4525cc','166ccc410','cb7ecfc41','72e970835','0eebebc7c','92e056c5c','fcc8443d9','a0fe4bb10','48df886f9','504c7e3bb','1ea08665c']
, ['e05e1751c','96be55d28','f0742e2c4','34b15f335','262e3fc42','3c649dad8','fa422ab84','4b15885d8','56e98e3ad','1da5c1b6b','bbcb92ecf','667132e4b','36d75938f','befca8b7e','f65d1049f','ef38209dc','96d9b7754','4415f4c2b','be83085df','e7962beb9','c4e5eb1f1','b2bc178d8','bc2eb559b','41016a42a','d1a5f5c20','93715fe15','1614f0f84','e2c21c4bc','023bc78d8','aab0aeb4b','6c7a4567c','89e69d1a3','489dde24b','cff75dd09','cba573a9d','831cebed2','57b902085','6eebf3ca4','ad064d609','1ecd56251']
, ['844df03d7','2e7f340f2','22f95560c','2a3c59733','1e403019b','a1d11c496','e429ad370','afac06058','a165f5761','6ab79c2fe','735ea6729','95ba53cf8','9685f5e16','227ac0d56','6879db4be','5da2e6220','89ca53693','dc5a8f1d8','dd0491aad','98d0d2971','324aaa96f','3d4a6baed','2715b2d4a','b7f26c1f7','b0385cee8','007d71f12','be448d5b9','e871db27b','69918e0c1','9d2dea573','43a1bf3e9','adc721d55','db1da2c31','ec1425047','cc462dc0b','b96c4256a','cb5329038','3aab2691c','796855249','cd41bbc4e']
, ['e20edfcb8','842415efb','300d6c1f1','720f83290','069a2c70b','87a91f998','611151826','74507e97f','504e4b156','baa95693d','cb4f34014','5239ceb39','dfdf4b580','81e02e0fa','fe5d62533','bb6260a44','fc9d04cd7','08d1f69ef','98d90a1d1','b4ced4b7a','b6d206324','6456250f1','96f5cf98a','f7c8c6ad3','cc73678bf','5fb85905d','cb71f66af','212e51bf6','d318bea95','b70c62d47','11d86fa6a','9f494676e','42cf36d73','3988d0c5e','1c68ee044','a728310c8','612bf9b47','c18cc7d3d','105233ed9','f08c20722']
, ['b26d16167','930f989bf','ca58e6370','aebe1ea16','03c589fd7','600ea672f','9509f66b0','70f4f1129','b0095ae64','1c62e29a7','32a0342e2','2fc5bfa65','026ca57fd','49e68fdb9','09c81e679','aacffd2f4','61483a9da','29725e10e','227ff4085','5878b703c','50a0d7f71','0d1af7370','7c1af7bbb','4bf056f35','3dd64f4c4','b9f75e4aa','423058dba','150dc0956','adf119b9a','a8110109e','6c4f594e0','c44348d76','1fcba48d0','db027dbaf','8d12d44e1','6ff9b1760','482715cbd','8d13d891d','f81c2f1dd','dda820122']
, ['81de0d45e','543c24e33','18562fc62','0256b6714','d6006ff44','e3a38370e','6a323434b','7c444370b','9657e51e1','8d2d050a2','13f3a3d19','b5c839236','70f3033c6','849125d91','f4b374613','16b532cdc','88219c257','fd1102929','74fb8f14c','699712087','22501b58e','9e9274b24','2c42b0dce','5263c204d','526ed2bec','01f7de15d','2c95e6e31','cdbe394fb','ef1e1fac8','d0f65188c','adf357c9b','b8a716ebf','a3f2345bf','110e4132e','586b23138','680159bab','f1a1562cd','af91c41f0','9f2f1099b','bf0e69e55']
, ['2c7e41e16','5ad1de183','d8cc972fe','5b2471c13','841704460','47e969ba0','fd6e11a24','991bca4be','b5a25e7c9','d77583e88','61a0acefa','0e931cdd3','d9fa0485a','614f8e1eb','1996a153f','289e5ecc3','b7d2baa45','ff8561ce9','601d54a3a','0fb0d19af','7ee833549','05c276b21','0b26c77a9','e8387d928','8fe3c178c','678e2d1dd','d966ac62c','f6c436744','e25a65f3d','f1b626ac2','adf03173b','57276ea06','9282e1543','bdbb0cd24','119230239','e7eb9a66b','6dad99586','4a9abd788','e7e41bbde','ebe9f985f']
, ['1ad24da13','f52a82e7f','8c5025c23','d75793f21','c0b22b847','4cffe31c7','6c2d09fb1','206ba1242','fb42abc0d','62f61f246','1389b944a','d15e80536','fa5044e9e','a0b0a7dbf','4e06c5c6d','1835531cd','1ff6be905','68b647452','c108dbb04','58e8e2c82','f3bfa96d5','f2db09ac3','4e8196700','8cd9be80e','83fc7f74c','dbc48d37c','2028e022d','17e160597','eb8cbd733','addb3f3eb','460744630','9108ee25c','b7950e538','a7da4f282','7f0d863ba','b7492e4eb','24c41bd80','fd7b0fc29','621f71db3','26f222d6d']
, ['f0317ca4f','402b0d650','7e78d546b','71ac2b961','2ad744c57','47abb3cb4','5b8c88c94','293e2698e','2ef8b7f4f','4bdeca0d2','c380056bb','20442bac4','2488e17f5','8a4c53d3e','62c547c8e','8e8736fc8','86f13324d','da52febdb','b0310a768','64e38e7a2','0d866c3d7','05f11f48f','34a2f580b','24bcc2f15','e1e8947d8','8c8616b62','ad1466df8','1ae0db9d5','79e0c374a','f642213a6','9dbb6b717','f8405f8b9','0f7ae26ce','81ec47b4c','ad4e33a4c','a78f85d49','8de6fcbf1','d2ef684ed','9e39c29d0','3ecc09859']
, ['d313c892b','0f5fb7fe7','037a54e89','f4e243e21','df3ac443c','f10717d56','0d8f892fe','09e395f05','d7f1f9e52','bcc18dd40','6760927a0','9e88cfd02','717039eef','62e587225','207871d87','ff08cfbbe','41fb39de4','a72fcabd8','bc3f77679','fd1187d68','7acda93e6','20868afc1','965fa6747','7e0fd6d92','75e55b7a9','c0d363088','08565b519','cfa24e4be','43727fb35','4a6f8b2c1','6035df6d8','acdef5318','4b2316bd5','dd16bb1ff','a46587cda','a8c320153','51ebff825','ac2392a17','21216a0a8','3b8208d28']
, ['1d9078f84','64e483341','a75d400b8','4fe8154c8','29ab304b9','20604ed8f','bd8f989f1','c1b9f4e76','b599b0064','4ead853dc','4824c1e90','58ed8fb53','d26279f1a','402bb0761','c7775aabf','ff65215db','74d7998d4','9884166a7','beb7f98fd','fd99c18b5','d83a2b684','18c35d2ea','0c8063d63','400e9303d','c976a87ad','8a088af55','5f341a818','762cbd0ab','5dca793da','db147ffca','fb5a3097e','8c0a1fa32','01005e5de','47cd6e6e4','a1db86e3b','50e4f96cf','f58fb412c','bb1113dbb','7a7da3079','f514fdb2e']
, ['5030aed26','b850c3e18','9e7c6b515','212efda42','ea5ed6ff7','2d065b147','49ca7ff2e','37c85a274','deabe0f4c','bae4f747c','ca96df1db','05b0f3e9a','eb19e8d63','235b8beac','85fe78c6c','cc507de6c','e0bb9cf0b','80b14398e','9ca0eee11','4933f2e67','fe33df1c4','e03733f56','1d00f511a','e62cdafcf','3aad48cda','92b13ebba','d36ded502','f30ee55dd','1f8754c4e','e75cfcc64','5d8a55e6d','db043a30f','6e29e9500','d251ee3b4','c5aa7c575','c2cabb902','8ab6f5695','73700eaa4','54b1c1bc0','cbd0256fb']
, ['b22288a77','a5f8c7929','330006bce','de104af37','8d81c1c27','d7285f250','123ba6017','3c6980c42','2d3296db7','95cdb3ab7','05527f031','65753f40f','45a400659','1d5df91e2','2a879b4f7','233c7c17c','c3c633f64','fdae76b2c','05d17ab7a','e209569b2','c25078fd7','3fd2b9645','268b047cd','3d350431d','b70c76dff','5fb9cabb1','3f6246360','12122f265','89e7dcacc','fcc17a41d','9e711a568','c5a742ee4','0186620d7','597d78667','4c095683e','b452ba57e','472cd130b','2ce2a1cdb','50c7ea46a','2761e2b76']
, ['f1eeb56ae','62ffce458','497adaff8','ed1d5d137','faf7285a1','d83da5921','0231f07ed','7950f4c11','051410e3d','39e1796ab','2e0148f29','312832f30','6f113540d','f3ee6ba3c','d9fc63fa1','6a0b386ac','5747a79a9','64bf3a12a','c110ee2b7','1bf37b3e2','fdd07cac1','0872fe14d','ddef5ad30','42088cf50','3519bf4a4','a79b1f060','97cc1b416','b2790ef54','1a7de209c','f118f693a','2a71f4027','15e8a9331','73e591019','0c545307d','363713112','21af91e9b','62a915028','2ab5a56f5','a8ee55662','316b978cd']
, ['48b839509','2b8851e90','28f75e1a5','0e3ef9e8f','7ca10e94b','37ac53919','467aa29ce','4b6c549b1','74c5d55dc','44f3640e4','0700acbe1','e431708ff','a0453715a','d1fd0b9c2','097836097','899dbe405','9e3aea49a','525635722','87a2d8324','d421e03fd','faf024fa9','1254b628a','34a4338bc','a19b05919','08e89cc54','a29c9f491','62ea662e7','7ab926448','ff2c9aa8f','5fe6867a4','a0a8005ca','8b710e161','215c4d496','4e5da0e96','b625fe55a','d04e16aed','b6fa5a5fd','55a7e0643','7124d86d9','0a26a3cfe']
, ['3b843ae7e','c8438b12d','d1b9fc443','19a45192a','63509764f','6b6cd5719','b219e3635','4b1d463d7','4baa9ff99','b0868a049','3e3ea106e','043e4971a','25e2bcb45','a2e5adf89','3ac0589c3','413bbe772','e23508558','c1543c985','9dcdc2e63','2dfea2ff3','1f1f641f1','dff08f7d5','75795ea0a','914d2a395','00302fe51','c0032d792','9d709da93','cb72c1f0b','5cf7ac69f','6b1da7278','47b5abbd6','26163ffe1','45bc3b302','902c5cd15','5c208a931','e88913510','e1d6a5347','38ec5d3bb','e3d64fcd7','199d30938']
, ['79e55ef6c','7f72c937f','408d86ce9','7a1e99f69','0f07e3775','736513d36','eb5a2cc20','2b0fc604a','aecd09bf5','91de54e0a','66891582e','8d4d84ddc','20ef8d615','2be024de7','dfde54714','d19110e37','e637e8faf','2d6bd8275','f3b4de254','5cebca53f','c4255588c','23c780950','bc56b26fd','55f4891bb','020a817ab','c4592ac16','542536b93','37fb8b375','0a52be28f','1904ce2ac','bd7bea236','6ae9d58e0','f8ee2386d','25729656f','5b318b659','589a5c62a','64406f348','e157b2c72','60d9fc568','0564ff72c']
, ['1a5a424f8','4d2c7622d','7cd18fa5c','f902239b9','a5fb00d9b','690c387d6','6dae32858','912028f4e','767b6d272','eb4dc2cdf','14fd68c51','5a648a09d','8579b0968','70b564f7b','a04684f1f','464676511','ed6e7fdaf','b7ce8464e','d49b0b346','18976b9f5','6cd62da62','a2d1008bb','8e978ee65','68084ece1','f96fc0e40','fac54bd7e','f8a3bb673','089ff7bcb','09a65c3a5','b200c8b4a','7df9efba5','7be4eb1e5','721c60041','807c7f49d','c2d94313f','e9f57e5c6','49a10e089','f553483a0','1d802b493','438d61d86']
, ['9ddd6d137','5cfc625f1','8984e4066','0ccd6454a','9397535c7','de7063efa','74f3ac6af','6bee3733e','20e2c484e','5adfe7419','03a4ccd7c','ecbd077d0','851697562','ea72c62a1','60cb16e88','73a8a4d75','4c48708d8','bbd16b7a0','3fa6c395f','5d60b9ba7','dba14a5d4','7f9e0d947','a636266f3','6931ed626','76e9423c3','6723b1708','1c4157dfd','c30399758','a2a1975d6','e32ad270b','7194699cd','b74ef4294','d80abf8bc','c436c7e73','d45fd5508','e3846e931','b66bf9d44','e97fa47e4','6f53aee73','02827212f']
, ['75b663d7d','4302b67ec','fc4a873e0','1e9bdf471','8f76eb6e5','3d71c02f0','05c9b6799','86875d9b0','27a7cc0ca','26df61cc3','9ff21281c','3ce93a21b','9f85ae566','3eefaafea','be4729cb7','72f9c4f40','afe8cb696','8c94b6675','ae806420c','63f493dba','5374a601b','5291be544','3690f6c26','acff85649','12a00890f','26c68cede','dd84964c8','fb06e8833','a208e54c7','7de39a7eb','5fe3acd24','e53805953','b719c867c','6c3d38537','86323e98a','2954498ae','3de2a9e0d','1f8a823f2','9cc5d1d8f','d3fbad629']
, ['16bf8b4ec','545d84e13','9a9b4a012','89a26cda9','049e4daae','95837bbfb','0ae364eb9','916ac9986','9b8eff1d7','21a3176c5','abace6b29','1930cefda','db1b70fc8','5a86cabd0','722a0187a','e14727834','b7ae337fe','2d1dd55ed','17a6e2978','42451bcbf','256f1449f','e92c373a6','82c164590','4f325b517','57e01acca','9fe78f046','7fd7c9eae','82e9efdd8','d6be59f6a','849c542c3','3ebf86dd5','befe0f9c4','54481feaa','248db7ce7','29c64fa08','849c464e7','4f0d3819a','24fef0850','1ac1a10d6','55e30b08f']
, ['5d4b3b236','87ee785b5','49ebf51c6','175795887','c8306d5b6','94e568b42','9f62e4134','c23810d14','ce3e5685e','f3ee34336','0704e7155','df443d0ce','c61dd62d9','29d7f1ebd','6cb207ac9','3bfa8340b','83b151006','f402f59ff','8c2f8a59c','ca67f1baa','9be6f99a8','e66856e20','039b8bbc6','5c78454d2','b4c2843b9','9ecfbd198','27859b383','fd7c50a10','c224ec4d9','f8b0c7834','6e4f74c35','5cf6f9d23','b192dacc4','5f7942448','3fe84d157','f357f5ffb','2e278fe94','813536f90','c4d38135f','e86d346f8']
, ['11ad148bd','54d3e247f','2d60e2f7a','c25438f10','e6efe84eb','964037597','0196d5172','47a8de42e','6f460d92f','0656586a4','22eb11620','c3825b569','6aa919e2e','086328cc6','c09edaf01','f9c3438ef','9a33c5c8a','85da130e3','2f09a1edb','76d34bbee','04466547a','3b52c73f5','1cfb3f891','704d68890','aba01a001','f45dd927f','c9160c30b','6a34d32d6','3e3438f04','038cca913','504c22218','56c679323','1938873fd','002d634dc','d37030d36','162989a6d','e4dbe4822','4f45e06b3','ad13147bd','ba480f343']
, ['9437d8b64','caa9883f6','68811ba58','fec5644cf','ef4b87773','ff558c2f2','8d918c64f','0b8e10df6','2d6565ce2','0fe78acfa','b75aa754d','2ab9356a0','4e86dd8f3','348aedc21','d7568383a','856856d94','69900c0d1','02c21443c','5190d6dca','20551fa5b','79cc300c7','8d8276242','da22ed2b8','89cebceab','f171b61af','129fe0263','3a07a8939','5ac7e84c4','aa7223176','e5b2d137a','9bd66acf6','e62c5ac64','4c938629c','57535b55a','a1a0084e3','2a3763e18','474a9ec54','0741f3757','4fe8b17c2','d5754aa08']
, ['4e98771c9','5c0b5d1d4','c95423453','5c20afdb3','29bf806d6','84d4d30b8','3770cb9fa','57412a852','3974799dd','13a2ecd25','3bb7bc789','963c9c0ac','04ef53271','e8d16b5b5','4f0b30912','d5d85bc77','a48a740ef','dacebaeaf','b4a4a4df8','174bec4d1','44c06f79a','7f3479656','890d30d93','b728093e6','211314d56','ee7e4581d','59d2470ed','538df95cd','055232767','8ca717e6d','133714358','f18d3931b','dc6902c31','16be01500','4a9e09bff','f7b2550f2','87ba106d3','366841793','15b0fe826','c6cbb2938']
, ['4d9538272','376474413','0892b3439','75d240f7b','f8de3e357','4bf2b8e7c','8c564ae48','50a900e26','ca4eab5c5','16a9296fd','9bed59a71','683d89bf1','6a3b5a968','60b76e463','736fce873','890163e1a','2c136905e','fbe52b1b2','08d203407','08af3dd45','78c239acf','3da2882fd','e2b4d4ef7','f2520b601','c63090352','10596ddee','b6e38a517','e9c7ccc05','df1ed6b50','b0e45a9f7','3e0e55648','5a88e3d89','fd206ec4d','2135da74a','28dc3cc44','93c1eecb4','06b19b6c4','2bf7dc91d','7b1ddbabf','acee6ff41']
, ['9fa984817','1b681c3f0','3d23e8abd','3be4dad48','b25319cb3','dcfcddf16','b14026520','c5cb7200e','e5ddadc85','07cb6041d','ede70bfea','df6a71cc7','dc60842fb','3a90540ab','6bab7997a','21e0e6ae3','9b39b02c0','c87f4fbfb','35da68abb','5f5cfc3c0','f0aa40974','625525b5d','d7978c11c','2bbcbf526','bc2bf3bcd','169f6dda5','4ceef6dbd','9581ec522','d4e8dd865','542f770e5','bf8150471','b05eae352','3c209d9b6','e5a8e9154','786351d97','b2e1308ae','2b85882ad','dc07f7e11','14c2463ff','14a5969a6']
, ['24b2da056','0f8d7b98e','c30ff7f31','ac0e2ebd0','476d95ef1','bd308fe52','202acf9bd','06be6c2bb','dbc0c19ec','d8296080a','f977e99dc','2191d0a24','7db1be063','9a3a1d59b','1bc285a83','c4d657c5b','a029667de','21bd61954','0e0f8504b','5910a3154','16bf5a9a2','ba852cc7a','685059fcd','21d6a4979','3839f8553','78947b2ad','1435ecf6b','fa1dd6e8c','f016fd549','e9b5b8919','632586103','c25ea08ba','7da54106c','b612f9b7e','395dbfdac','e7c0a50e8','29181e29a','04dc93c58','1beb0ce65','733b3dc47']
, ['ccc7609f4','ca7ea80a3','e509be270','3b8114ab0','a355497ac','27998d0f4','4e22de94f','fa05fd36e','f0d5ffe06','81aafdb57','f1b6cc03f','9af753e9d','567d2715c','857020d0f','3e5dab1e3','99fe351ec','001476ffa','5a5eabaa7','cb5587baa','32cab3140','313237030','0f6386200','b961b0d59','bcfb439ee','9452f2c5f','04a22f489','a4c9ea341','ffdc4bcf8','7e58426a4','298db341e','d7334935b','1a6d866d7','8367dfc36','08984f627','7e3e026f8','5d9f43278','37c10d610','5a88b7f01','99f466457','324e49f36']
, ['4569d5378','f32763afc','9bb02469c','61063fa1c','5fad07863','22f05c895','4a93ad962','fa1efdadd','6ae0787f3','ed0860a34','ffd50f0bf','4ef309fc3','704e2dc55','1b1a893f5','b19e65a65','8d4b52f9a','85dcc913d','92ba988e1','0aab2f918','6d46740f1','783ee6e9a','6610f90f1','13f7f9c70','73361d959','c5c073bb0','a235f5488','fb6da0420','635fbbd2c','78bc2558b','b904b8345','f3b6dabf7','60cd556c9','150504397','d92ea0b2a','ca25aad9f','4e1a8f6eb','c89ae4ce0','f2af9300f','9d435a85b','8d035d41e']
, ['2135fa05a','e8a3423d6','90a438099','7ad6b38bd','60e45b5ee','2b9b1b4e2','d6c82cd68','923114217','b361f589e','ee0b53f05','04be96845','21467a773','47665e3ce','a6229abfb','7dcc40cda','9666bfe76','a89ab46bb','17be6c4e7','9653c119c','cc01687d0','60e9cc05b','ffcec956f','51c250e53','7344de401','a15b2f707','a8e607456','dbb8e3055','2a933bcb8','b77bc4dac','17068424d','58d9f565a','027a2206a','7453eb289','343042ed9','29eddc376','c8fb3c2d8','1c873e4a6','588106548','358dc07d0','282cfe2ad']
, ['a3e023f65','9126049d8','6eaea198c','5244415dd','0616154cc','2165c4b94','fc436be29','9d5af277d','1834f29f5','c6850e7db','6b241d083','56f619761','45319105a','fcda960ae','07746dcda','c906cd268','c24ea6548','829fb34b8','89ebc1b76','22c019a2e','1e16f11f3','94072d7a3','59dfc16da','9886b4d22','0b1741a7f','a682ef110','ac0493670','5c220a143','e26299c3a','68c7cf320','3cea34020','8d8bffbae','afb6b8217','e9a8d043d','1de4d7d62','5780e6ffa','26628e8d8','4c53b206e','99cc87fd7','593cccdab']
, ['53aa182a2','295408598','4e92107c6','b76bf3f19','3305c8063','d3a116347','ac5260727','199caef5d','97ea72529','1d4d5cd4a','8fc7efaf0','225fa9d61','94f3dcaee','4634c8fae','052f633c1','660fdbc58','657dec16b','7fa5bc19f','7207afb67','cda277b2a','e9a473fbb','3eac9a76e','1c554649c','86ffb104c','b14d5014b','8348ea8d3','e3a4596f9','49db469f5','f928893ca','aa610feec','fa2a340da','652142369','947c7c3e8','f81908ca5','c2d200f0e','8160230fd','c99902a93','d3a6362c5','3ee95e3ef','7f8027faf']
, ['0d7692145','62071f7bc','ab515bdeb','c30c6c467','eab76d815','49063a8ed','b6ee6dae6','4cb2946ce','6c27de664','afd87035a','772288e75','44f2f419e','e803a2db0','754ace754','65119177e','c70f77ef2','3a66c353a','4c7768bff','9e4765450','dc8b7d0a8','24141fd90','ba499c6d9','8b1379b36','5a3e3608f','3be3c049e','a0a3c0f1b','457bd191d','4d2ca4d52','6620268ab','7f55b577c','9ad654461','1a1962b67','e059a8594','bc937f79a','989d6e0f5','3b74ac37b','555265925','aa37f9855','32c8b9100','e71a0278c']
, ['36a56d23e','4edc3388d','a00a63886','63be1f619','9e2040e5b','f7faf2d9f','5f11fbe33','992b5c34d','26e998afd','7e1c4f651','f7f553aea','f5538ee5c','711c20509','374b83757','55338de22','f41f0eb2f','bf10af17e','e2979b858','fe0c81eff','d3ed79990','5c0df6ac5','82775fc92','fa9d6b9e5','a8b590c6e','f1c20e3ef','b5c4708ad','c9aaf844f','50a6c6789','fe3fe2667','8761d9bb0','2b6f74f09','84067cfe0','b6403de0b','5755fe831','2bd16b689','d01cc5805','91ace30bd','15e4e8ee5','870e70063','8895ea516']
, ['5b465f819','944e05d50','a2aa0e4e9','4f8b27b6b','a498f253f','c73c31769','616c01612','025dea3b3','83ea288de','f3316966c','2dbeac1de','b4d41b335','47b7b878f','686d60d8a','7210546b2','6dcd9e752','78edb3f13','30992dccd','7f9d59cb3','26144d11f','a970277f9','0aea1fd67','dc528471e','d51d10e38','efa99ed98','48420ad48','7f38dafa6','3a13ed79a','1af4ab267','73445227e','57c4c03f6','971631b2d','7f91dc936','0784536d6','c3c3f66ff','052a76b0f','ffb34b926','9d4f88c7b','442b180b6','948e00a8d']
, ['c13ee1dc9','abb30bd35','d2919256b','66728cc11','eab8abf7a','317ee395d','cc03b5217','38a92f707','467c54d35','e8f065c9d','2ac62cba5','6495d8c77','94cdda53f','1c047a8ce','13f2607e4','28a5ad41a','05cc08c11','b0cdc345e','38f49406e','773180cf6','1906a5c7e','c104aeb2e','8e028d2d2','8e5a41c43','28a785c08','0dc333fa1','03ee30b8e','8b5c0fb4e','67102168f','9fc776466','14a22ab1a','4aafb7383','55741d46d','8e1dfcb94','fd812d7e0','8f940cb1b','6562e2a2c','758a9ab0e','4ea447064','343922109']
, ['ec5764030','fa6e76901','42fdff3a0','6e76d5df3','9562ce5c8','1c486f8dd','2daf6b624','8e1822aa3','cbf236577','fd9968f0d','6bd9d9ae3','ed1f680d4','b41a9fc75','896d1c52d','9d6b84f39','a60974604','5661462ee','e5ac02d3c','186b87c05','0c4bf4863','e17f1f07c','1fba6a5d5','707f193d9','cd8048913','3adf5e2b5','715fa74a4','4f2f6b0b3','8ca08456c','f8b733d3f','dd85a900c','50780ec40','adadb9a96','8485abcab','a60027bb4','6af8b2246','c7ae29e66','77eb013ca','ccb68477c','e7071d5e3','994b4c2ac']
, ['9d4428628','37f11de5d','39549da61','ceba761ec','4c60b70b8','304ebcdbc','823ac378c','4e21c4881','5ee81cb6e','eb4a20186','f6bdb908a','6654ce6d8','65aa7f194','c4de134af','00f844fea','a240f6da7','168c50797','13d6a844f','7acae7ae9','8c61bede6','45293f374','feeb05b3f','a5c62af4a','c46028c0f','1d0aaa90f','22abeffb6','337b3e53b','cde3e280a','c83fc48f2','f99a09543','d6af4ee1a','85ef8a837','64cabb6e7','46c525541','cef9ab060','a31ba11e6','93521d470','3c4df440f','375c6080e','e613715cc']
, ['b1b17b543','bc21e80ff','ab7764ead','84f287070','c3726f249','9616802bb','da5814d9b','b78487210','1084e5813','d9db07d68','9c720c580','941244262','1a9501bae','75de1e5b6','157c8b45f','0cd22b1b5','dd01f3999','b88568883','950f2c435','7ab374cb1','38bbaa62d','4fbcb9f95','155f1b1e5','2b58a21fc','a61ce65a2','f7d385108','aafb4ec55','170655e35','5d80001c0','a513d67d5','d9dc805dd','f0eee77af','9381024b7','607a7b8f0','22b7e449b','c96615af4','a7f94dd85','e7913a5ce','1ea2c906f','2223c664d']
, ['6679fe54f','8f6514df0','5e62457b7','f17ff4efd','ec7f7017f','c02ab7d25','8c309c553','e0b968d7b','22b980fc8','3b6b46221','3e4a6796d','c680e9350','834fb292d','e3d33877c','4052a9419','b95be4138','16517c8b0','219e051b5','a6fbe0987','37d7af8ad','b84b2f72d','775577e6f','4f0c5f900','a68b83290','2a2832b07','ce1f5b02a','a6c9347a7','82c9b4fcd','7f78a36f7','89cffafe9','f49ff3269','aeb3a6ccf','4d6a1439e','c7753cbfc','2123a4f36','5c56fccf1','03bfe48b2','6beb0b35d','ae141696e','9fb38aabe']
, ['a924cf47a','4d294d2cf','a6fd11a84','1f0a4e1f9','e369704a1','51ee03895','daedcafad','7bddf55e1','91fd68481','8c922fa9a','0809c8241','0c49d75af','00b309c64','e506de1e1','afa9b3198','bea06dade','b67c7783e','b261b0abe','090fba3ad','0badd2fa2','6ba70f5f8','4d1f9e4d7','c333aa06c','f98d7054f','6cd2424c4','864b62f7d','0761cbb48','903749e8a','4411325ed','e5587ec32','9f5a3b3c0','a47445036','ce408348f','284d07c28','8706aa459','c47fe5e84','ae3aa1abd','c85a3dcc4','a1f73b0d3','693972ceb']
, ['9c36a77b3','509e911f0','50aaba7f1','9a179ed71','d6a122efd','ed5af35f0','30768bc79','ffd2f9409','1fbbd4edf','9161061c9','e171bccbe','6a055c4fb','d7cdd8aef','61efa1e29','b791ce9aa','1a82869a6','3696a15a7','7b31055f1','37426563f','d168174c7','a76ad8513','82ba7a053','31a3f920c','ba5bbaffc','d3022e2f1','0ccd5ff1c','f1fbe249b','86eb6ec85','18c0b76e9','6d0d72180','75aad4520','22dbe574a','f269ec9c8','38df6c628','5860d7fa9','455f29419','0cad4d7af','dae4d14b4','9d6410ef5','1e1cb47f3']
, ['9a2b0a8be','856225035','709573455','f9db72cff','19a67cb97','616be0c3e','9d478c2ae','9c502dcd9','cf5b8da95','2f7b0f5b5','d50798d34','c612c5f8f','56da2db09','08c089775','59cb69870','7aaefdfd7','37c0a4deb','fb9a4b46d','b4eaa55ea','99f22b12d','304633ac8','4bffaff52','65000b269','4c536ffc0','93a445808','e8b513e29','a2616a980','71aae7896','97d5c39cf','c2acc5633','62d0edc4f','c8d5efceb','e50c9692b','2e1287e41','2baea1172','af1e16c95','01c0495f8','b0c0f5dae','090f3c4f2','33293f845']
, ['8ceddccb8','203c64df6','2087ed398','1ffee02ec','c07f4daba','d8e8397ce','7650524a3','431e67099','b15a468b2','8389fa5f0','eae6fc02f','6d773e96f','46ee7f2c8','5f04745bf','43ef60caa','e0a18e5b6','30609ee5b','f41af7c85','776c262ad','d6bca77b4','abc207b83','dc135562a','54428f346','af0b98ec8','8601a29bc','2dd0e885c','072452760','bf40c722d','dc6676b1f','79c8119ae','4a3baddf6','dd19c0b80','098721511','67db03f3a','608639adb','58a1cb6eb','062f6f3f7','2f8931894','0de698985','7f80a96a9']
, ['372daeab0','9e0c57b34','ec827621a','44d132265','850d3a6f5','05f1b68b8','3b6f67b0e','414b74eaa','18cad608c','440d789c5','615cc4c17','4685cc47b','6cf9184bb','6809065b9','ca04a07ca','39896d3dd','0106dd950','e9c45d66f','06a1c3b47','3855aef1e','7250feb72','ed8ff54b5','ac308c9a3','c7fd9abc6','3dc46e323','964cd68bc','6984f4045','c9eda7d9c','f23b7530c','c593d73e8','d3245937e','5509e2e98','42b407f0d','10f17bd3e','0f81cc1d2','258412544','cbeddb751','76a75bd91','22b3971f5','ff3ebf76b']
, ['4dcf81d65','2cb73ede7','df838756c','61c1b7eb6','1af4d24fa','a9f61cf27','e13b0c0aa','b9ba17eb6','796c218e8','37f57824c','f9e3b03b7','d1e0f571b','a3ef69ad5','e16a20511','04b88be38','1dd7bca9f','2eeadde2b','99e779ee0','9f7b782ac','6df033973','cdfc2b069','031490e77','5324862e4','467bee277','956d228b9','64c6eb1cb','8618bc1fd','a3fb07bfd','6b795a2bc','949ed0965','191e21b5f','a4511cb0b','2e3c96323','b64425521','5e645a169','1977eaf08','bee629024','1d04efde3','8675bec0b','8337d1adc']]

# In[ ]:


pattern_1964666 = pd.read_csv('../input/pattern-found/pattern_1964666.66.csv')
pattern_1166666 = pd.read_csv('../input/pattern-found/pattern_1166666.66.csv')
pattern_812666 = pd.read_csv('../input/pattern-found/pattern_812666.66.csv')
pattern_2002166 = pd.read_csv('../input/pattern-found/pattern_2002166.66.csv')
pattern_3160000 = pd.read_csv('../input/pattern-found/pattern_3160000.csv')
pattern_3255483 = pd.read_csv('../input/pattern-found/pattern_3255483.88.csv')

# In[ ]:


pattern_1964666.drop(['Unnamed: 0','value_count'],axis=1,inplace=True)
pattern_1166666.drop(['Unnamed: 0','value_count'],axis=1,inplace=True)
pattern_812666.drop(['Unnamed: 0','value_count'],axis=1,inplace=True)
pattern_2002166.drop(['Unnamed: 0','value_count'],axis=1,inplace=True)
pattern_3160000.drop(['Unnamed: 0','value_count'],axis=1,inplace=True)
pattern_3255483.drop(['Unnamed: 0','value_count'],axis=1,inplace=True)

# In[ ]:


pattern_1166666.rename(columns={'8.50E+43': '850027e38'},inplace=True)

# In[ ]:


l=[]
l.append(pattern_1964666.columns.values.tolist())
l.append(pattern_1166666.columns.values.tolist())
l.append(pattern_812666.columns.values.tolist())
l.append(pattern_2002166.columns.values.tolist())
l.append(pattern_3160000.columns.values.tolist())
l.append(pattern_3255483.columns.values.tolist())

# In[ ]:


ss = l + colgroups + colgps

# Updating this function on the basis of the hint provided by Paradox [here](http://www.kaggle.com/c/santander-value-prediction-challenge/discussion/61472#363394).

# In[ ]:


def _get_leak(df, cols,extra_feats, lag=0):
    f1 = cols[:((lag+2) * -1)]
    f2 = cols[(lag+2):]
    for ef in extra_feats:
        f1 += ef[:((lag+2) * -1)]
        f2 += ef[(lag+2):]
    
    d1 = df[f1].apply(tuple, axis=1).to_frame().rename(columns={0: 'key'})
    d1.to_csv('extra_d1.csv')
    d2 = df[f2].apply(tuple, axis=1).to_frame().rename(columns={0: 'key'}) 

    d2['pred'] = df[cols[lag]]
#     d2.to_csv('extra_d2.csv')
    #d2 = d2[d2.pred != 0] ### to make output consistent with Hasan's function
    d3 = d2[~d2.duplicated(['key'], keep=False)]
    d4 = d1[~d1.duplicated(['key'], keep=False)]
    d5 = d4.merge(d3, how='inner', on='key')
    
    d6 = d1.merge(d5, how='left', on='key')
    d6.to_csv('extra_d6.csv')
    
    return d1.merge(d5, how='left', on='key').pred.fillna(0)

# In[ ]:


def compiled_leak_result():
    
    max_nlags = len(cols)-2
    train_leak = train[["ID", "target"] + cols]
    train_leak["compiled_leak"] = 0
    train_leak["nonzero_mean"] = train[transact_cols].apply(
        lambda x: np.expm1(np.log1p(x[x!=0]).mean()), axis=1
    )
    scores = []
    leaky_value_counts = []
    leaky_value_corrects = []
    leaky_cols = []
    
    for i in range(max_nlags):
        c = "leaked_target_"+str(i)
        print('Processing lag', i)
        #train_leak[c] = _get_leak(train, cols,l, i)
        train_leak[c] = _get_leak(train, cols,ss, i)
        
        leaky_cols.append(c)
        train_leak = train.join(
            train_leak.set_index("ID")[leaky_cols+["compiled_leak", "nonzero_mean"]], 
            on="ID", how="left"
        )[["ID", "target"] + cols + leaky_cols+["compiled_leak", "nonzero_mean"]]
        zeroleak = train_leak["compiled_leak"]==0
        train_leak.loc[zeroleak, "compiled_leak"] = train_leak.loc[zeroleak, c]
        leaky_value_counts.append(sum(train_leak["compiled_leak"] > 0))
        _correct_counts = sum(train_leak["compiled_leak"]==train_leak["target"])
        leaky_value_corrects.append(_correct_counts*1.0/leaky_value_counts[-1])
        print("Leak values found in train", leaky_value_counts[-1])
        print(
            "% of correct leaks values in train ", 
            leaky_value_corrects[-1]
        )
        tmp = train_leak.copy()
        tmp.loc[zeroleak, "compiled_leak"] = tmp.loc[zeroleak, "nonzero_mean"]
        print('Na count',tmp.compiled_leak.isna().sum())
        scores.append(np.sqrt(mean_squared_error(y, np.log1p(tmp["compiled_leak"]).fillna(14.49))))
        print(
            'Score (filled with nonzero mean)', 
            scores[-1]
        )
    result = dict(
        score=scores, 
        leaky_count=leaky_value_counts,
        leaky_correct=leaky_value_corrects,
    )
    return train_leak, result

# In[ ]:


train_leak, result = compiled_leak_result()

# In[ ]:


result = pd.DataFrame.from_dict(result, orient='columns')
result.T

# In[ ]:


result.to_csv('train_leaky_stat.csv', index=False)

# In[ ]:


train_leak.head()

# In[ ]:


best_score = np.min(result['score'])
best_lag = np.argmin(result['score'])
print('best_score', best_score, '\nbest_lag', best_lag)

# In[ ]:


def rewrite_compiled_leak(leak_df, lag):
    leak_df["compiled_leak"] = 0
    for i in range(lag):
        c = "leaked_target_"+str(i)
        zeroleak = leak_df["compiled_leak"]==0
        leak_df.loc[zeroleak, "compiled_leak"] = leak_df.loc[zeroleak, c]
    return leak_df

# In[ ]:


leaky_cols = [c for c in train_leak.columns if 'leaked_target_' in c]
train_leak = rewrite_compiled_leak(train_leak, best_lag)
train_leak[['ID']+leaky_cols+['compiled_leak']].head()

# In[ ]:


test.head()

# In[ ]:


def compiled_leak_result_test(max_nlags):
    test_leak = test[["ID", "target"] + cols]
    test_leak["compiled_leak"] = 0
    test_leak["nonzero_mean"] = test[transact_cols].apply(
        lambda x: np.expm1(np.log1p(x[x!=0]).mean()), axis=1
    )
    
    scores = []
    leaky_value_counts = []
    # leaky_value_corrects = []
    leaky_cols = []
    
    for i in range(max_nlags):
        c = "leaked_target_"+str(i)
        
        print('Processing lag', i)
        #test_leak[c] = _get_leak(test_leak, cols, i)
        test_leak[c] = _get_leak(test, cols,ss, i)
        
        leaky_cols.append(c)
        test_leak = test.join(
            test_leak.set_index("ID")[leaky_cols+["compiled_leak", "nonzero_mean"]], 
            on="ID", how="left"
        )[["ID", "target"] + cols + leaky_cols+["compiled_leak", "nonzero_mean"]]
        zeroleak = test_leak["compiled_leak"]==0
        test_leak.loc[zeroleak, "compiled_leak"] = test_leak.loc[zeroleak, c]
        leaky_value_counts.append(sum(test_leak["compiled_leak"] > 0))
        #_correct_counts = sum(train_leak["compiled_leak"]==train_leak["target"])
        #leaky_value_corrects.append(_correct_counts/leaky_value_counts[-1])
        print("Leak values found in test", leaky_value_counts[-1])
        #print(
        #    "% of correct leaks values in train ", 
        #    leaky_value_corrects[-1]
        #)
        #tmp = test_leak.copy()
        #tmp.loc[zeroleak, "compiled_leak"] = tmp.loc[zeroleak, "nonzero_mean"]
        #scores.append(np.sqrt(mean_squared_error(y, np.log1p(tmp["compiled_leak"]).fillna(14.49))))
        #print(
        #    'Score (filled with nonzero mean)', 
        #    scores[-1]
        #)
    result = dict(
        # score=scores, 
        leaky_count=leaky_value_counts,
        # leaky_correct=leaky_value_corrects,
    )
    return test_leak, result

# In[ ]:


test_leak, test_result = compiled_leak_result_test(max_nlags=38)


# In[ ]:


test_result = pd.DataFrame.from_dict(test_result, orient='columns')
test_result.T

# In[ ]:


test_result.to_csv('test_leaky_stat.csv', index=False)

# In[ ]:


best_lag = 37

# In[ ]:


test_leak = rewrite_compiled_leak(test_leak, best_lag)
test_leak[['ID']+leaky_cols+['compiled_leak']].head()

# In[ ]:


test_res = test_leak[leaky_cols+['compiled_leak']].replace(0.0, np.nan)
test_res.to_csv('test_leak.csv', index=False)

# In[ ]:


test_leak.loc[test_leak["compiled_leak"]==0, "compiled_leak"] = test_leak.loc[test_leak["compiled_leak"]==0, "nonzero_mean"]

# In[ ]:


#submission
sub = test[["ID"]]
sub["target"] = test_leak["compiled_leak"]
sub.to_csv(f"baseline_sub_lag_{best_lag}.csv", index=False)
print(f"baseline_sub_lag_{best_lag}.csv saved")

# In[ ]:


sub.tail()

# In[ ]:



