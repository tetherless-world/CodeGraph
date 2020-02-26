
import json
from sklearn.decomposition import TruncatedSVD
#import pandas as pd
from sklearn.manifold import TSNE
import pandas as pd
import seaborn as sns
#%matplotlib inline
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import pickle
import time 
X=pickle.load( open( "save.x", "rb" ) )
moduleArr=pickle.load(open("save.moduleArr","rb"))

svd = TruncatedSVD(n_components=4, n_iter=7, random_state=42)
Z = svd.fit_transform(X)

time_start = time.time()
tsne = TSNE(n_components=2, verbose=1, perplexity=7, n_iter=300)
tsne_results = tsne.fit_transform(Z)
#df_subset['tsne-2d-one'] = tsne_results[:,0]
#df_subset['tsne-2d-two'] = tsne_results[:,1]
#plt.figure(figsize=(16,10))
#print(len(moduleArr))
#print(len(tsne_results))
plt.autoscale(enable=True)
df=pd.DataFrame({'x':tsne_results[:,0],'y':tsne_results[:,1],'group':moduleArr[0:-1]})
#sns_plt=sns.regplot(data=df, x="x", y="y", fit_reg=False, marker="+", color="skyblue")
p1=sns.regplot(data=df, x="x", y="y", fit_reg=False, marker="o", color="skyblue", scatter_kws={'s':400})
for line in range(0,df.shape[0]):
         p1.text(df.x[line]+0.2, df.y[line], df.group[line], horizontalalignment='left', size='small', color='black')
figure = p1.get_figure()    

figure.savefig("tSNEout.png")

print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))
