#!/usr/bin/env python
# coding: utf-8

# **DESTINO ELEGIDO Airbnb**
# 
# Airbnb es un mercado en línea y servicio de hospitalidad, que permite a las personas arrendar o alquilar alojamiento a corto plazo. Para que Airbnb brinde una experiencia personalizada a sus clientes, busca explorar la posibilidad de predecir el país de destino en el que un usuario realizará una reserva. Con esta información, Airbnb puede crear contenido más personalizado con su comunidad, disminuir el tiempo promedio para la primera reserva y mejorar la previsión de la demanda. 
# Estos objetivos proporcionan beneficios mutuos para Airbnb y sus clientes: las recomendaciones personales pueden mejorar el compromiso de los clientes con la plataforma, lo que fomenta las reservas repetidas y las referencias a Airbnb para aquellos los amigos cercanos y familiares.
# En este ejercicio tenemos los datos de los clientes de Airbnb y se busca predecir el primer destino de reserva para los nuevos clientes de Airbnb que viajan desde los Estados Unidos. 
# La variable de respuesta es el destino donde se realiza la reserva. Este puede ser uno de los 12 valores posibles: 'US', 'FR', 'CA', 'GB', 'ES', 'IT', 'PT', 'NL', 'DE', 'AU', ' NDF ', y' otro '. Todas estas corresponden a abreviaturas de dos letras del país, con la excepción de 'NDF', que corresponde a 'No se encontró el destino' e implica que el usuario no ha hecho una reserva.

# **CONJUNTO DE DATOS**
# 
# Para este ejercicio se contara con el siguiente conjuntos de datos: 
# 
# Una lista de usuarios junto con sus datos demográficos. 
# Registros de sesiones web y algunas estadísticas de resumen. Debemos predecir en qué país será el primer destino de la reserva de un nuevo usuario. Todos los usuarios en el conjunto de datos son de los Estados Unidos.
# 
# Hay 12 resultados posibles del país de destino: 'EE. UU.,' FR ',' CA ',' GB ',' ES ',' IT ',' PT ',' NL ',' DE ',' AU ', 'NDF' (no se encontró destino), y 'otro'. 'otro' significa que hubo una reserva, pero es para un país que no está incluido en la lista, mientras que 'NDF' significa que no había una reserva.
# 
# Los conjuntos de entrenamiento y prueba están divididos por fechas. El conjunto de pruebas contiene nuevos usuarios con primeras actividades después del 07/01/2014. En el conjunto de datos de las sesiones, los datos solo se remontan al 01/01/2014, mientras que el conjunto de datos de los usuarios se remonta a 2010.
# 
# ***Conjunto de datos de usuario***
# 
# train_users.csv - El conjunto de entrenamiento de usuarios.
# 
# test_users.csv - el conjunto de prueba de usuarios
# 
# ID: ID de usuario
# date_account_created: la fecha de creación de la cuenta
# timestamp_first_active: timestamp de la primera actividad, tenga en cuenta que puede ser anterior a date_account_created o date_first_booking porque un usuario puede buscar antes de registrarse
# date_first_booking: fecha de la primera reserva
# gender: Genero
# age: Edad
# signup_method
# Signup_flow: la página desde la cual un usuario vino a registrarse.
# language: preferencia de idioma internacional
# affiliate_channel: qué tipo de marketing pagado
# affiliate_provider: donde el marketing es, por ejemplo, google, craigslist, otros
# first_affiliate_tracked: ¿cuál es la primera comercialización con la que el usuario interactuó antes de registrarse?
# signup_app
# first_device_type
# first_browser
# country_destination: esta es la variable objetivo que debes predecir
# 
# ***Conjunto de datos de sesión***
# 
# session.csv - registro de sesiones web para usuarios
# 
# id_usuario: para unirse con la columna 'id' en la tabla de usuarios
# action
# action_type
# action_detail
# device_type
# secs_elapsed
# 
# ***Conjunto de datos de países***
# 
# countries.csv : estadísticas resumidas de los países de destino en este conjunto de datos y sus ubicaciones
# 
# age_gender_bkts.csv - Estadísticas resumidas del grupo de edad de los usuarios, género, país de destino

# In[89]:


# Draw inline

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime

# Set figure aesthetics
sns.set_style("white", {'ytick.major.size': 10.0})
sns.set_context("poster", font_scale=1.1)

# *Exploración de datos*

# In[90]:


# Cargue los datos en DataFrames 
ruta =  '../input/' 
train_users = pd.read_csv (ruta +  'train_users_2.csv' )
test_users = pd.read_csv (ruta +  'test_users.csv' )
sessions = pd.read_csv (ruta +  'sessions.csv' )
countries = pd.read_csv (ruta +  'countries.csv' )
age_gender = pd.read_csv (ruta +  'age_gender_bkts.csv' )

# *Conteo*

# In[91]:


print("Tenemos", train_users.shape[0], "registros en el set de entrenamiento y", 
      test_users.shape[0], "en el set de pruebas.")
print("En total tenemos", train_users.shape[0] + test_users.shape[0], "usuarios.")
print(sessions.shape[0], "Registros de sesión para" , sessions.user_id.nunique() , "usuarios." )
print((train_users.shape[0] + test_users.shape[0] -sessions.user_id.nunique()) , "Usuarios sin registros de sessión." )
print((countries.shape[0]) , "Registros en el Dataset de Países." )
print((age_gender.shape[0]) , "registros en el Dataset edad/genero." )

# *Usuarios*

# In[92]:


# Unimos usuarios de Pruebas y Entrenamiento
users = pd.concat((train_users, test_users), axis=0, ignore_index=True, sort=False)

# Removemos ID's
users.set_index('id',inplace=True)

users.head()

# *Sesiones*

# In[93]:


sessions.head()

# *Paises*

# In[94]:


countries

# *Datos Faltantes*

# Los datos faltantes vienen en la forma **NaN** , pero en los datos anteriores podemos ver que la columna genero tiene algunos valores **-unknown-**. Primero transformamos esos valores en **NaN** y luego resumimos el porcentaje de incógnitas en cada campo.

# In[95]:


users.gender.replace('-unknown-', np.nan, inplace=True)
users.first_browser.replace('-unknown-', np.nan, inplace=True)

# In[96]:


users_nan = (users.isnull().sum() / users.shape[0]) * 100
users_nan[users_nan > 0].drop('country_destination')

# In[97]:


users.age.describe()

# Como podemos ver existe cierta incoherencia en la edad de algunos usuarios. Para sanear esto solo dejaremos las edades entre 18 y 85 años.

# In[98]:


print('Usuarios mayores de 85 años: ' + str(sum(users.age > 85)))
print('Uuarios menores de 18 años: ' + str(sum(users.age < 18)))

# In[99]:


users[users.age > 85]['age'].describe()

# In[100]:


users[users.age < 18]['age'].describe()

# In[101]:


users.loc[users.age > 85, 'age'] = np.nan
users.loc[users.age < 18, 'age'] = np.nan

# *Verificamos la depuración*

# In[102]:


users.age.describe()

# *Tipos de datos*
# 
# En el siguiente paso convertimos cada característica como lo que son.
# Transformamos la fecha y las variables categóricas en los tipos de datos correspondientes.
# 
# Datos categóricos:
# 
# affiliate_channel
# affiliate_provider
# country_destination
# first_affiliate_tracked
# first_browser
# first_device_type
# gender
# language
# signup_app
# signup_method
# 
# Datos Fecha:
# 
# date_account_created
# date_first_booking
# date_first_active

# In[103]:


categorical_features = [
    'affiliate_channel',
    'affiliate_provider',
    'country_destination',
    'first_affiliate_tracked',
    'first_browser',
    'first_device_type',
    'gender',
    'language',
    'signup_app',
    'signup_method'
]

for categorical_feature in categorical_features:
    users[categorical_feature] = users[categorical_feature].astype('category')

# In[104]:


users['date_account_created'] = pd.to_datetime(users['date_account_created'])
users['date_first_booking'] = pd.to_datetime(users['date_first_booking'])
users['date_first_active'] = pd.to_datetime(users['timestamp_first_active'], format='%Y%m%d%H%M%S')

# **Visualizing the Data**

# Genero

# In[105]:


users.gender.value_counts(dropna=False).plot(kind='bar', color='#FD5C64', rot=0)
plt.xlabel('Genero')
sns.despine()

# La gráfica anterior nos ayuda a visualizar la cantidad de datos faltantes para esta función. También podemos notar que hay una ligera diferencia en los conteos entre el género del usuario.
# 
# Lo siguiente que observamos es ver si hay preferencias de género al viajar.

# In[106]:


women = sum(users['gender'] == 'FEMALE')
men = sum(users['gender'] == 'MALE')

female_destinations = users.loc[users['gender'] == 'FEMALE', 'country_destination'].value_counts() / women * 100
male_destinations = users.loc[users['gender'] == 'MALE', 'country_destination'].value_counts() / men * 100

# Bar width
width = 0.4

male_destinations.plot(kind='bar', width=width, color='#4DD3C9', position=0, label='Male', rot=0)
female_destinations.plot(kind='bar', width=width, color='#FFA35D', position=1, label='Female', rot=0)

plt.legend()
plt.xlabel('Pais de Destino')
plt.ylabel('Porcentaje')

sns.despine()
plt.show()

# **País de destino**

# In[107]:


counts =  users.country_destination.value_counts(normalize=True).plot(kind='bar')
plt.xlabel('Pais de Destino')
plt.ylabel('Porcentaje')

# **Edad**

# La gráfica de los datos de edad de los usuarios es la siguiente.

# In[108]:


sns.distplot(users.age.dropna(), color='#FD5C64')
plt.xlabel('Age')
sns.despine()

# Como es de esperar, la edad habitual para viajar es entre los 25 y los 40 años. Queríamos explorar más a fondo si hay diferencias en los patrones de reserva según la edad de los usuarios. Tomamos un rango de división arbitriario de 50 y trazamos el siguiente gráfico.

# In[109]:


age = 50

younger = sum(users.loc[users['age'] < age, 'country_destination'].value_counts())
older = sum(users.loc[users['age'] > age, 'country_destination'].value_counts())

younger_destinations = users.loc[users['age'] < age, 'country_destination'].value_counts() / younger * 100
older_destinations = users.loc[users['age'] > age, 'country_destination'].value_counts() / older * 100

younger_destinations.plot(kind='bar', width=width, color='#63EA55', position=0, label='Jovenes', rot=0)
older_destinations.plot(kind='bar', width=width, color='#4DD3C9', position=1, label='Viejos', rot=0)

plt.legend()
plt.xlabel('Pais de Destino')
plt.ylabel('Porcentaje')

sns.despine()
plt.show()

# Podemos ver que los jóvenes tienden a permanecer en los EE. UU., Y las personas mayores eligen viajar fuera del país.

# **Idioma**
# 
# Exploramos la función de idioma para comprender la distribución y veremos que sería un buen predictor para el país de destino. Podemos visualizar debajo de ese idioma si captura variaciones en el destino de la reserva de los usuarios. Por ejemplo, en el gráfico a continuación, las personas con destino "fr" en el país, "fr" es el segundo destino preferido para la primera reserva después de "US".

# In[110]:


import matplotlib.cm as cm
colors = cm.rainbow(np.linspace(0,1,22))
users[~(users['country_destination'].isin(['NDF']))].groupby(['country_destination' , 'language']).size().unstack().plot(kind='bar', figsize=(20,10),stacked=False,color=colors)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5),
          ncol=1, fancybox=True, shadow=True)
plt.yscale('log')
plt.xlabel('Pais de Destino')
plt.ylabel('Log(Conteo)')

# **Fechas**

# In[111]:


sns.set_style("whitegrid", {'axes.edgecolor': '0'})
sns.set_context("poster", font_scale=1.1)
users.date_account_created.value_counts().plot(kind='line', linewidth=1.2, color='#FD5C64')

# In[112]:


date_first_active = users.date_first_active.apply(lambda x: datetime.datetime(x.year, x.month, x.day))
date_first_active.value_counts().plot(kind='line', linewidth=1.2, color='#FD5C64')

# In[113]:


users['date_account_created'] = pd.to_datetime(users['date_account_created'], errors='ignore')
users['date_first_active'] = pd.to_datetime(users['timestamp_first_active'], format='%Y%m%d%H%M%S')
users['date_first_booking'] = pd.to_datetime(users['date_first_booking'], errors='ignore')

# In[114]:


df = users[~users['country_destination'].isnull()]
df.groupby([df["date_account_created"].dt.year, df["date_account_created"].dt.month])['country_destination'].count().plot(kind="bar",figsize=(20,10))

# In[115]:


import matplotlib.cm as cm
colors = cm.rainbow(np.linspace(0,1,12))
df[df["date_first_booking"].dt.year == 2013].groupby(['country_destination' , df["date_first_booking"].dt.month]).size().unstack().plot(kind='bar', stacked=False,color=colors)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5),
          ncol=1, fancybox=True, shadow=True)
plt.yscale('log')
plt.xlabel('Destination Country by Month 2013')
plt.ylabel('Log(Count)')

# Informacion de Afiliados

# In[116]:


colors = cm.rainbow(np.linspace(0,1,users['affiliate_channel'].nunique()))
users.groupby(['country_destination','affiliate_channel']).size().unstack().plot(kind='bar', stacked=False,color=colors)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5),
          ncol=1, fancybox=True, shadow=True)
plt.yscale('log')
plt.xlabel('Destination Country by affiliate channel')
plt.ylabel('Log(Count)')

# In[117]:


colors = cm.rainbow(np.linspace(0,1,users['affiliate_provider'].nunique()))
users.groupby(['country_destination','affiliate_provider']).size().unstack().plot(kind='bar', stacked=False,color=colors)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5),
          ncol=1, fancybox=True, shadow=True)
plt.yscale('log')
plt.xlabel('Destination Country by affiliate provider')
plt.ylabel('Log(Count)')

# In[118]:


colors = cm.rainbow(np.linspace(0,1,users['first_affiliate_tracked'].nunique()))
users.groupby(['country_destination','first_affiliate_tracked']).size().unstack().plot(kind='bar', stacked=False,color=colors)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5),
          ncol=1, fancybox=True, shadow=True)
plt.yscale('log')
plt.xlabel('Destination Country by first affiliate tracked')
plt.ylabel('Log(Count)')

# **Preprocessing**

# *Edad*

# In[119]:


import numpy as np
import pandas as pd
users.loc[users.age > 85, 'age'] = np.nan
users.loc[users.age < 18, 'age'] = np.nan
users['age'].fillna(-1,inplace=True)
bins = [-1, 0, 4, 9, 14, 19, 24, 29, 34,39,44,49,54,59,64,69,74,79,84,89]
users['age_group'] = np.digitize(users['age'], bins, right=True)

# In[120]:


users.age_group.value_counts().plot(kind='bar')
plt.yscale('log')
plt.xlabel('Age Group')
plt.ylabel('Log(Count)')

# *Fecha*

# In[121]:


df = users[users['country_destination'].isnull()]

# In[122]:


date_account_created = pd.DatetimeIndex(users['date_account_created'])
date_first_active = pd.DatetimeIndex(users['date_first_active'])
date_first_booking = pd.DatetimeIndex(users['date_first_booking'])

# In[123]:


users['time_lag_create'] = (date_first_booking - date_account_created).days
users['time_lag_active'] = (date_first_booking - date_first_active).days
users['time_lag_create'].fillna(365,inplace=True)
users['time_lag_active'].fillna(365,inplace=True)

# In[124]:


import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn.apionly as sns
import importlib
importlib.reload(mpl); importlib.reload(plt); importlib.reload(sns)
ax = sns.boxplot(x="country_destination", y="time_lag_create", showfliers=False,data=users[~(users['country_destination'].isnull())])
#users[~(users['country_destination'].isnull())][['time_lag_create','country_destination']].boxplot(by='country_destination')


# In[125]:


import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn.apionly as sns
import importlib
importlib.reload(mpl); importlib.reload(plt); importlib.reload(sns)
ax = sns.boxplot(x="country_destination", y="time_lag_active", showfliers=False,data=users[~(users['country_destination'].isnull())])
#users[~(users['country_destination'].isnull())][['time_lag_create','country_destination']].boxplot(by='country_destination')

# In[126]:


users[['time_lag_create','time_lag_active']].describe()

# In[127]:


users.loc[users.time_lag_create > 365, 'time_lag_create'] = 365
users.loc[users.time_lag_active > 365, 'time_lag_create'] = 365

# In[128]:


drop_list = [
    'date_account_created', 'date_first_active', 'date_first_booking', 'timestamp_first_active', 'age'
]

users.drop(drop_list, axis=1, inplace=True)

# *Información de Sesión*

# In[129]:


sessions.rename(columns = {'user_id': 'id'}, inplace=True)

# In[130]:


from sklearn import preprocessing
# Create a minimum and maximum processor object
min_max_scaler = preprocessing.MinMaxScaler()

action_count = sessions.groupby(['id'])['action'].nunique()

#action_count = pd.DataFrame(min_max_scaler.fit_transform(action_count.fillna(0)),columns=action_count.columns)
action_type_count = sessions.groupby(['id', 'action_type'])['secs_elapsed'].agg(len).unstack()
action_type_count.columns = action_type_count.columns.map(lambda x: str(x) + '_count')
#action_type_count = pd.DataFrame(min_max_scaler.fit_transform(action_type_count.fillna(0)),columns=action_type_count.columns)
action_type_sum = sessions.groupby(['id', 'action_type'])['secs_elapsed'].agg(sum)

action_type_pcts = action_type_sum.groupby(level=0).apply(lambda x:
                                                 100 * x / float(x.sum())).unstack()
action_type_pcts.columns = action_type_pcts.columns.map(lambda x: str(x) + '_pct')
action_type_sum = action_type_sum.unstack()
action_type_sum.columns = action_type_sum.columns.map(lambda x: str(x) + '_sum')
action_detail_count = sessions.groupby(['id'])['action_detail'].nunique()

#action_detail_count = pd.DataFrame(min_max_scaler.fit_transform(action_detail_count.fillna(0)),columns=action_detail_count.columns)

device_type_sum = sessions.groupby(['id'])['device_type'].nunique()

#device_type_sum = pd.DataFrame(min_max_scaler.fit_transform(device_type_sum.fillna(0)),columns=device_type_sum.columns)

sessions_data = pd.concat([action_count, action_type_count, action_type_sum,action_type_pcts,action_detail_count, device_type_sum],axis=1)
action_count = None
action_type_count = None
action_detail_count = None
device_type_sum = None


#users = users.join(sessions_data, on='id')

# In[131]:


users= users.reset_index().join(sessions_data, on='id')

# **Codificar las características categóricas**

# In[132]:


from sklearn.preprocessing import LabelEncoder
categorical_features = [
    'gender', 'signup_method', 'signup_flow', 'language',
    'affiliate_channel', 'age_group','weekday_account_created','month_account_created','weekday_first_active','month_first_active','hour_first_active',
    'signup_app','affiliate_provider', 'first_affiliate_tracked','first_device_type', 'first_browser'
]
users_sc = users.copy(deep=True)
encode = LabelEncoder()
for j in categorical_features:
    users_sc[j] = encode.fit_transform(users[j].astype('str'))

# *Selección de Caracteristicas*

# In[133]:


colx = users_sc.columns.tolist()
rm_list = ['id','country_destination']
for x in rm_list:
    colx.remove(x)
X = users_sc[~(users_sc['country_destination'].isnull())][colx]
X.fillna(0,inplace=True)
from sklearn.feature_selection import VarianceThreshold
sel = VarianceThreshold(threshold=(0.8))
sel.fit_transform(X)
idxs = sel.get_support(indices=True)
colo = [X.columns.tolist()[i] for i in idxs]
print ('\n'.join(colo))
for y in rm_list:
    colo.append(y)
