
import json
from sklearn.decomposition import TruncatedSVD
#import pandas as pd
from sklearn.manifold import TSNE

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
print(tsne_results)
print(len(tsne_results))
plt.autoscale(enable=True)
for i in range(len(Z)):
    plt.annotate(s=moduleArr[i],xy=(tsne_results[i,0],tsne_results[i,1]))
print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))
out_png = 'lsaspacialemTNSE.png'
plt.savefig(out_png, dpi=150)

#print("pickle loaded")
#svd = TruncatedSVD()
#Z = svd.fit_transform(X)
#plt.scatter(Z[:,0], Z[:,1])
#for i in range(len(Z)):
#        plt.annotate(s=moduleArr[i], xy=(Z[i,0], Z[i,1]))
#        plt.show()
#        out_png = 'lsaspacialembeddingdocuments.png'
#        plt.savefig(out_png, dpi=150)

