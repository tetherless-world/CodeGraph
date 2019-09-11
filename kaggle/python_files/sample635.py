#!/usr/bin/env python
# coding: utf-8

# **[Voltar para a Página Inicial do Curso](https://www.kaggle.com/c/ml-em-python)**
# 
# # **Árvore de Decisão**
# Antes de entrarmos no assunto de Random Forest, precisamos conhecer as Árvores de Decisão ou Decision Trees. 
# ![Nós e Divisões](https://i1.wp.com/www.vooo.pro/insights/wp-content/uploads/2016/12/RDS-Vooo_insights-Tutorial_arvore_de_decisao_02.jpg?resize=640%2C371&ssl=1)

# Na árvore de decisão o nó raiz em conjunto com os nós de decisão representam os atributos (ou variáveis) e cada nó de término representa uma saída. Sua estrutura em árvore facilita o entendimento, a interpretação e a lógica da tomada de decisão. Funciona da seguinte forma:
# 1. Seleciona o melhor atributo para dividir os registros;
# 2. Faz daquele atributo um nó de decisão e divide o dataset em conjuntos menores de dados (subsets)
# 3. Repete este processo de forma recursiva até que não exista mais atributos para dividir, ou não exista mais registros, ou todos os registros pertençam ao mesmo atributo;

# ![](https://assets.almanaquesos.com/wp-content/uploads/2012/05/541190_267646449990606_165232073565378_577645_1319247076_n.jpg)

# ## Você consegue descobrir se uma pessoa vai ter diabetes?
# Temos um conjunto de dados com 8 características de pessoas indígenas e um resultado que mostra quais tiveram diabetes.
# 1. Modelo de Árvore de Decisão
# 2. Análise Exploratória
# 3. Correlação
# 4. Divisão do Dataset
# 5. Criação do Modelo de Árvore de Decisão
# 6. Características da Árvore
# 7. Importância das Variáveis
# 8. Model Tuning: Poda da ÁRvore
# 9. Model Tuning: Critério de Divisão 
# 10. Exercídio de Árvore de Decisão

# In[ ]:


# Importa as bibliotecas
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics 

# In[ ]:


# Importa o Dataset
indios = pd.read_csv('../input/pima-indians/PimaIndians.csv')
indios.tail()

# ### 1. Modelo de Árvore de Decisão 
# 

# In[ ]:


# Escolher o alvo. Dividir os atributos (variáveis independentes) da target (label)
x = indios[['pregnant', 'glucose', 'diastolic', 'triceps', 'insulin', 'bmi', 'diabetes', 'age']]
y = indios['test']

# In[ ]:


# Vamos criar uma árvore de decisão
modelodt = DecisionTreeClassifier()

# Treina o modelo com árvore de decisão
modelodt = modelodt.fit(x, y)

# Aplica o modelo treinado no dataset para prever o resultado
y_previsao = modelodt.predict(x)

# Avalia a acurácia do modelo de árvore de decisão
metrics.accuracy_score(y, y_previsao)

# ### 2. Análise Exploratória

# In[ ]:


# Análise Exploratória Inicial
indios.describe()

# In[ ]:


# Tem Outliers?
indios.boxplot()

# In[ ]:


# Avalia um dos atributos: quantidade de gravidez
indios.boxplot('pregnant')

# In[ ]:


# Valores Ausentes - Missing Values
indios.isnull().sum().sort_values(ascending = False)

# ### 3. Correlação

# In[ ]:


# Correlação. Cadê a variável test?
indios.corr()

# In[ ]:


# Transforma test em binario: 0 - sem diabentes  ,   1 - com diabetes   
import numpy as np
indios['test'] = pd.Series( np.where(indios.test== 'positif' , 1 , 0 ) , name = 'test' )

# In[ ]:


# Verifica
indios.head()

# In[ ]:


# Correlação. 
indios.corr()

# In[ ]:


# Visualiza a Correlação
import seaborn as sns
sns.set(rc={'figure.figsize':(11,8)})
sns.heatmap(indios.corr(), annot=True, annot_kws={"size": 10})

# ### 4. Divisão do Dataset entre Treino e Teste

# In[ ]:


# Escolher o alvo. Dividir os atributos (variáveis independentes) da target (label)
x = indios[['pregnant', 'glucose', 'diastolic', 'triceps', 'insulin', 'bmi', 'diabetes', 'age']]
y = indios['test']

# In[ ]:


# Divide os dados entre 70% treino e 30% teste
from sklearn.model_selection import train_test_split
x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.3)

# ### 5. Criação do Modelo de Árvore de Decisão

# In[ ]:


# Vamos criar uma árvore de decisão 
modelodt = DecisionTreeClassifier()

# Treina o modelo com árvore de decisão
modelodt = modelodt.fit(x_treino, y_treino)

# Aplica o modelo treinado no dataset de teste para prever o resultado
y_previsao = modelodt.predict(x_teste)

# Avalia a acurácia do modelo de árvore de decisão
metrics.accuracy_score(y_teste, y_previsao)

# ### 6. Características da Árvore

# In[ ]:


# Imprime as primeiras linhas da previsao do resultado
y_previsao[0:5]

# In[ ]:


# Imprime a profundidade da árvore
print(modelodt.tree_.max_depth)

# In[ ]:


# Imprime a quantidade de nós
print(modelodt.tree_.node_count)

# In[ ]:


# Settings: habilitar a Internet

# In[ ]:


# Visualização da Árvore de Decisão
from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus

dot_data = StringIO()
export_graphviz(modelodt, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True, feature_names = x_treino.columns,class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('diabetes.png')
Image(graph.create_png())

# ### 7. Importância das Variáveis

# In[ ]:


# Variáveis mais importantes com Árvore de Decisão
import matplotlib.pyplot as plt

# Cria uma serie de atributos mais importantes e ordena as séries
importances = pd.Series(data=modelodt.feature_importances_, index= x_treino.columns)
importances_sorted = importances.sort_values()

# Desenha um gráfico de barras
importances_sorted.plot(kind='barh', color='lightgreen')
plt.title('Features Importances')
plt.show()

# ### 8. Model Tuning: Poda da Árvore
# * **max_depth:** limita a profundidade máxima da árvore, ou seja, poda a árvore. O valor pode ser um número inteiro que representa a profundidade da árvore ou `None` para nenhum. O valor padrão é `None`, o qual irá expandir ao máximo. Quanto maior a profundidade, maior o overfitting. Quanto menor a profundidade, maior o underfitting.

# In[ ]:


# Imprime a profundidade da árvore
print(modelodt.tree_.max_depth)

# In[ ]:


# Model Tuning
modelodt = DecisionTreeClassifier(max_depth=3)
modelodt = modelodt.fit(x_treino, y_treino)
y_previsao = modelodt.predict(x_teste)
metrics.accuracy_score(y_teste, y_previsao)

# In[ ]:


# Visualização da Árvore de Decisão
from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus

dot_data = StringIO()
export_graphviz(modelodt, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True, feature_names = x_treino.columns,class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('diabetes.png')
Image(graph.create_png())

# Os nós em azul são os casos positivos. Os nós laranja são os casos negativos. Gini é a probabilidade de pertencer à classe. Samples é a quantidade de registros em cada nó.

# ### 9. Model Tuning: Critério de Divisão - Como Selecionar o Melhor Atributo
# A medida de seleção de atributo ou Attribute Selection Measures - ASM é aplicada para realizar a seleção de melhor atributo, usada para dividir os registros do dataset. Esta medida de seleção de atributo faz um ranking de cada variável que explica a sua importância para explicar um dataset. As variáveis de melhor score serão selecionadas como nós de decisão ou atributos de divisão. escolhe a medida de seleção de atributo. Pode ser `gini` para o Gini Index, ou `entropy` para Information Gain. O default é `gini`.
# * **Information Gain:** a entropia mede a impureza dos dados de entrada. Information gain busca diminuir a entropia e mede a diferença entre a entropia antes e depois da divisão do dataset baseado em determinados valores de atributos. Esta medida prefere atributos com uma grande variedade de valores distintos, o que pode levar a um resultado enviesado. Por exemplo: campo ID com uma sequencia numérica. Para aplicar: `criterion = 'entropy'`. Entropia de 0 significa nenhuma impureza, todos os registros são da mesma classe. Ou seja, quanto mais próximo de zero, melhor.
# * **Gini Index:** usa a probabilidade de um registro pertencer à uma classe para dividir os registros. Considera um divisão binária para cada atributo. No caso de variáveis categóricas, o subconjunto de dados que tiver o menor gini index é selecionado como atributo de divisão. Para o caso de variáveis contínuas, a estratégia é selecionar cada par de valores adjacentes como possível ponto de divisão e o ponto com o menor gini index é escolhido como ponto de divisão. Vai de 0 a 1. É usada para avaliar a separação dos dados entre as classes. Seu valor indica o quão boa foi a separação das classes entre dois grupos. Uma separação perfeita resulta num Gini de 0. Enquanto que o pior caso resulta num valor de 0.5. É o valor default. Para aplicar: `criterion = 'gini'`.
# 
# Outro parâmetro é a estratégia de divisão:
# * **splitter:** escolhe a estratégia de divisão, que pode ser `best` ou `random`. O default é `best`.

# In[ ]:


# Vamos criar uma árvore de decisão com critério entropia
modelodt = DecisionTreeClassifier(criterion='entropy', splitter='random')

# Treina o modelo com árvore de decisão
modelodt = modelodt.fit(x_treino, y_treino)

# Aplica o modelo treinado no dataset de teste para prever o resultado
y_previsao = modelodt.predict(x_teste)

# Avalia a acurácia do modelo de árvore de decisão
metrics.accuracy_score(y_teste, y_previsao)

# ### 10. Exercício de Árvore de Decisão
# Vamos descobrir se um paciente terá doenças do coração.

# In[ ]:


# 1. Adicione um Dataset chamado "Heart Disease UCI" clicando no menu em  `+ Add Dataset`
# 2. Importe os dados do heart.csv num dataframe chamado "doenca" e imprima suas primeiras linhas
# 3. Crie uma variável x com os atributos 'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach','exang','oldpeak','slope','ca','thal'
# 4. Crie uma variável y com o 'target'
# 5. Crie as variáveis x_treino, x_teste, y_treino, y_teste, dividindo os dados entre 70% treino e 30% teste
# 6. Importe as bibliotecas de Árvore de Decisão
# 7. Crie um modelo de Árvore de Decisão
# 8. Treine o modelo com árvore de decisão e salve na variavel chamada modelodt (use o fit)
# 9. Aplique o modelo treinado no dataset de teste para prever o resultado e coloque o resultado em y_previsao (use o predict)
# 10. Avalie a acurácia do modelo de árvore de decisão (use metrics.accuracy_score)

# ## Random x Reprodutibilidade

# In[ ]:


# Divisão com Random State
x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.3, random_state=1)

# Vamos criar uma árvore de decisão com Random State
modelodt = DecisionTreeClassifier(random_state=1)

# Treina o modelo com árvore de decisão
modelodt = modelodt.fit(x_treino, y_treino)

# Aplica o modelo treinado no dataset de teste para prever o resultado
y_previsao = modelodt.predict(x_teste)

# Avalia a acurácia do modelo de árvore de decisão
metrics.accuracy_score(y_teste, y_previsao)

# ---
# # Random Forest
# Random Forest ou Árvores Aleatórias é um método de aprendizado de máquina supervisionado que aplica árvores de decisão em vários subconjuntos dos dados, criando uma floresta aleatória, onde seu resultado tem a maioria dos valores dos modelos das árvores de decisão geradas. A métrica de validação é medida pela média da métrica das árvores de decisão. Random Forest reduz a variância. 
# 
# 1. Criar Modelo de Random Forest
# 2. Importância das Variáveis com Random Forest
# 3. Model Tuning de Random Forest
# 4. Exercício de Random Forest

# ### 1. Criar Modelo de Random Forest

# In[ ]:


from sklearn.ensemble import RandomForestClassifier

# Vamos criar uma Random Forest
modelorf = RandomForestClassifier()

# Treina o modelo
modelorf = modelorf.fit(x_treino, y_treino)

# Aplica o modelo treinado no dataset de teste para prever o resultado
y_previsao = modelorf.predict(x_teste)

# Avalia a acurácia do modelo 
metrics.accuracy_score(y_teste, y_previsao)

# ### 2. Importância das Variáveis com RandomForest

# In[ ]:


# Variáveis mais importantes com RandomForest
import matplotlib.pyplot as plt

# Cria uma serie de atributos mais importantes e ordena as séries
importances = pd.Series(data=modelorf.feature_importances_, index= x_treino.columns)
importances_sorted = importances.sort_values()

# Desenha um gráfico de barras
importances_sorted.plot(kind='barh', color='lightgreen')
plt.title('Features Importances')
plt.show()

# ### 3. Model Tuning ou Otimização do Modelo
# * **n_estimators:** quantidade de árvores de decisão
# * **max_depth:** profundidade da árvore de decisão

# In[ ]:


from sklearn.ensemble import RandomForestClassifier

# Random Forest
modelorf = RandomForestClassifier(max_depth = 6, n_estimators=100, random_state=1)
modelorf = modelorf.fit(x_treino, y_treino)
y_previsao = modelorf.predict(x_teste)
print(metrics.accuracy_score(y_teste, y_previsao))

# Extract single tree
estimator_limited = modelorf.estimators_[5]
estimator_limited

# ### 4. Exercício de Random Forest
# Vamos descobrir se um paciente terá doenças do coração com o Random Forest e comparar o resultado com a Árvore de Decisão do Exercício anterior.

# In[ ]:


# 1. Importe as bibliotecas de Random Forest 
# 2. Crie um modelo de Random Forest
# 3. Treine o modelo com Random Forest e salve na variavel chamada modelorf (use o fit)
# 4. Aplique o modelo treinado no dataset de teste para prever o resultado e coloque o resultado em y_previsao (use o predict)
# 5. Avalie a acurácia do modelo de RandomForest (use metrics.accuracy_score). Foi melhor, igual ou pior que o Árvores de Decisão?
# 6. Que parâmetros você alteraria para otimizar o modelo de Random Forest?
# 7. Faça a alteração de parametros para otimizar o modelo (Model Tuning) e avalie o resultado com o resultado sem Model Tuning.

# ## Balanceamento de Classes
# Quando um lado é bem maior do que o outro. Um dataset é considerado desbalanceado se uma categoria é muito mais expressiva do que a outra. Uma árvore pode se considerada enviesada se tiver classes desbalanceadas. Neste caso, é recomendável balancear as classes antes de criar o modelo. 
# ![](https://classeval.files.wordpress.com/2015/06/balanced-or-imbalanced.png?w=450&h=213)

# In[ ]:


# Proporção
indios.test.value_counts()

# ![](https://raw.githubusercontent.com/rafjaa/machine_learning_fecib/master/src/static/img/resampling.png)

# In[ ]:


# Vamos criar uma árvore de decisão com classes balanceadas
modelodt = DecisionTreeClassifier(class_weight='balanced')

# Treina o modelo com árvore de decisão
modelodt = modelodt.fit(x_treino, y_treino)

# Aplica o modelo treinado no dataset de teste para prever o resultado
y_previsao = modelodt.predict(x_teste)

# Avalia a acurácia do modelo de árvore de decisão
metrics.accuracy_score(y_teste, y_previsao)

# ## Curva ROC 
# Avalia a performance do algoritmo, muito usado principalmente no caso de dataset desbalanceado. A ROC mostra o quão bom o modelo criado pode distinguir entre duas coisas (já que é utilizado para classificação). Essas duas coisas pode ser 0 ou 1, ou positivo e negativo. AUC - Area Under the ROC Curve nada mais é que a “área sob a curva”. O valor do AUC varia de 0,0 até 1,0 e o limiar entre a classe é 0,5. Ou seja, acima desse limite, o algoritmo classifica em uma classe e abaixo na outra classe. Quanto maior o AUC, melhor. Quanto mais próximo de 1.0 melhor. 

# In[ ]:


# Curva ROC
from sklearn.metrics import roc_auc_score  

probs = modelodt.predict_proba(x_teste)
probs = probs[:, 1]  
roc_auc_score(y_teste,probs)

# In[ ]:


# Gráfico
import matplotlib.pyplot as plt 
from sklearn.metrics import roc_curve  

y_pred_proba = modelodt.predict_proba(x_teste)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_teste,  y_pred_proba)
auc = metrics.roc_auc_score(y_teste, y_pred_proba)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()

# ## Matriz de Confusão
# ![](https://cdn-images-1.medium.com/max/1600/1*s0aMRNsHq7A3bCA9gX_qXQ.png)

# Num dataset com 50 fotos de gatos e 50 fotos não são de gatos, total de 100. A soma de todos os valores é o tamanho de y = 100.
# 
# * **sensibilidade ou precisão:** são os verdadeiros positivos classificados corretamente. Dentre todas as classificações de classe Positivo que o modelo fez, quantas estão corretas. A precisão pode ser usada em uma situação em que os Falsos Positivos são considerados mais prejudiciais que os Falsos Negativos. 25/50
# * **especificidade:** são os casos negativos identificados corretamente. 40/50
# * **acurácia:** são os casos verdadeiros previstos corretamente. Indica uma performance geral do modelo. Dentre todas as classificações, quantas o modelo classificou corretamente. 65/100
# * **recall:** dentre todas as situações de classe Positivo como valor esperado, quantas estão corretas. O recall pode ser usada em uma situação em que os Falsos Negativos são considerados mais prejudiciais que os Falsos Positivos. 25/35
# * **f1-score:** média harmônica entre precisão e recall. 

# In[ ]:


# Resultado da Matriz de Confusão
metrics.confusion_matrix(y_teste, y_previsao)

# ![](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSjLta8fwnUGWIOdujZ4ADLSoHStF0ZPMQWBMHLdnfF9A5d171Wqg)

# In[ ]:


# precision, sensibilidade ou precisão: TP/(TP+FP)
64/(64+17)

# recall: TP/(TP+FN)
64/(64+18)

# f1-score: 2 * precision * recall / precision + recall
(2*(64/(64+17))*(64/(64+18)))/((64/(64+17))+(64/(64+18)))

# Support: quantidade de itens classificados em cada classe
64 + 18 # negatif
17 + 19 # posifif

# In[ ]:


# Resultado da Matriz de Confusão
print(metrics.classification_report(y_teste, y_previsao))

# In[ ]:


# especificidade: TN/TN+FP
19/(19+17)

# acurácia: TP+TN/TP+FN+TN+FP
(64+19)/(64+18+19+17)

# Outra forma de imprimir a acurácia
metrics.accuracy_score(y_teste, y_previsao)

# ## Regressão com Árvore de Decisão
# Usa Mean Squared Error para fazer as divisões da árvore, e não a entropia ou Gini.

# In[ ]:


from sklearn.tree import DecisionTreeRegressor
# Vamos criar uma árvore de decisão 
modelodt = DecisionTreeRegressor()

# Treina o modelo com árvore de decisão
modelodt = modelodt.fit(x_treino, y_treino)

# Aplica o modelo treinado no dataset de teste para prever o resultado
y_previsao = modelodt.predict(x_teste)

# Avalia a acurácia do modelo de árvore de decisão
#metrics.accuracy_score(y_teste, y_previsao)
print(metrics.r2_score(y_teste, y_previsao))

from sklearn import metrics  
print('Mean Absolute Error:', metrics.mean_absolute_error(y_teste, y_previsao))  
print('Mean Squared Error:', metrics.mean_squared_error(y_teste, y_previsao))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_teste, y_previsao)))  

# É menor que 10% da média dos valores da target?
y.mean()

# ## Regressão com Random Forest

# In[ ]:


from sklearn.ensemble import RandomForestRegressor

# Vamos criar uma Random Forest
modelorf = RandomForestRegressor()

# Treina o modelo
modelorf = modelorf.fit(x_treino, y_treino)

# Aplica o modelo treinado no dataset de teste para prever o resultado
y_previsao = modelorf.predict(x_teste)

# Avalia a acurácia do modelo 
#metrics.accuracy_score(y_teste, y_previsao)

# Avalia a acurácia do modelo 
metrics.r2_score(y_teste, y_previsao)

from sklearn import metrics  
print('Mean Absolute Error:', metrics.mean_absolute_error(y_teste, y_previsao))  
print('Mean Squared Error:', metrics.mean_squared_error(y_teste, y_previsao))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_teste, y_previsao)))  

# ## Conclusão
# Vantagens:
# * Árvores de Decisão são fáceis de interpretar e visualizar
# * Pode capturar facilmente com padrões não-lineares
# * Requer menor pré-processamento dos dados, não precisamos normalizar as colunas
# * Pode ser usado para prever missing values
# 
# Desvantagens:
# * Sensível a outliers, pode sofrer overfitting facilmente
# * Na árvore de decisão, uma pequena variancia pode resultar em diferentes árvores. Este problema pode ser reduzido com o uso de Random Forest.
# * Podem ser enviesadas por datasets desbalanceados. Este problema pode ser resolvido com o balanceamento dos datasets.

# --- 
# ### Resposta dos Exercícios

# In[ ]:


import pandas as pd
doenca = pd.read_csv('../input/heart.csv')
doenca.head()

# Escolher o alvo. Dividir os atributos (variáveis independentes) da target (label)
x = doenca[['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach','exang','oldpeak','slope','ca','thal']]
y = doenca['target']

# Divide os dados entre 70% treino e 30% teste
from sklearn.model_selection import train_test_split
x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.3)

# Árvore de Decisao
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics 

# Vamos criar uma árvore de decisão 
modelodt = DecisionTreeClassifier()

# Treina o modelo com árvore de decisão
modelodt = modelodt.fit(x_treino, y_treino)

# Aplica o modelo treinado no dataset de teste para prever o resultado
y_previsao = modelodt.predict(x_teste)

# Avalia a acurácia do modelo de árvore de decisão
metrics.accuracy_score(y_teste, y_previsao)

# In[ ]:


# Random Forest
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics 

# Vamos criar uma árvore de decisão 
modelorf = RandomForestClassifier(max_depth=4, n_estimators=100, random_state=1, criterion='entropy')

# Treina o modelo com árvore de decisão
modelorf = modelorf.fit(x_treino, y_treino)

# Aplica o modelo treinado no dataset de teste para prever o resultado
y_previsao = modelorf.predict(x_teste)

# Avalia a acurácia do modelo de árvore de decisão
metrics.accuracy_score(y_teste, y_previsao)

# ---
# # Exercícios Complementares:
# 1. [Criar modelo de Árvores de Decisão (Decision Trees)](https://www.kaggle.com/kernels/fork/400771)
# 2. [Avaliar o modelo de Árvore de Decisão (Decision Trees)](https://www.kaggle.com/kernels/fork/1259097)
# 3. [Underfitting e Overfitting](https://www.kaggle.com/kernels/fork/1259126)
# 4. [Random Forest](https://www.kaggle.com/kernels/fork/1259186)
# 
# ### Próxima Aula:
# [Support Vector Machines - SVM](https://www.kaggle.com/debkings/6-m-quinas-de-suporte-de-vetores-svm)
